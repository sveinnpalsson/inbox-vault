#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-config.toml}"
TARGET_MAX="${TARGET_MAX:-100}"
ENRICH_BATCH_SIZE="${ENRICH_BATCH_SIZE:-${ENRICH_LIMIT:-200}}"
PROFILES_LIMIT="${PROFILES_LIMIT:-200}"
INDEX_BATCH_SIZE="${INDEX_BATCH_SIZE:-500}"
PROFILES_USE_LLM="${PROFILES_USE_LLM:-0}"
REDACTION_MODE="${REDACTION_MODE:-hybrid}"
NO_PAUSE="${NO_PAUSE:-0}"
STRICT_ENRICH="${STRICT_ENRICH:-0}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-10}"
PROGRESS_VERBOSITY="${PROGRESS_VERBOSITY:-detailed}"

RUN_ROOT=".runs/default"
LOG_DIR="$RUN_ROOT/logs"
LOCK_FILE="$RUN_ROOT/run_build_stepwise.lock"
FINAL_SUMMARY_JSON="$RUN_ROOT/final-summary.json"
FINAL_SUMMARY_TXT="$RUN_ROOT/final-summary.txt"
STARTED_AT_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

usage() {
  cat <<'EOF'
Usage: scripts/run_build_stepwise.sh [options]

Options:
  --config <path>             Config path (default: config.toml)
  --target-max <n>            Backfill max messages target (default: 100)
  --enrich-limit <n>          Enrich batch size (deprecated alias)
  --enrich-batch-size <n>     Enrich batch size (default: 200)
  --profiles-limit <n>        Build-profiles batch limit (default: 200)
  --index-batch-size <n>      Index chunk size per pass (default: 500)
  --profiles-use-llm          Enable LLM profile generation (default: off)
  --redaction-mode <mode>     regex|model|hybrid for index-vectors (default: hybrid)
  --no-pause                  Non-interactive mode (skip Press Enter prompts)
  --strict-enrich             Fail if enrich cannot drain all pending rows (default: off)
  --heartbeat-seconds <n>     Heartbeat cadence while batch command runs (default: 10)
  --progress-verbosity <mode> detailed|compact|quiet (default: detailed)
  -h, --help                  Show this help

Environment overrides are supported for the same names:
  CONFIG TARGET_MAX ENRICH_BATCH_SIZE ENRICH_LIMIT PROFILES_LIMIT INDEX_BATCH_SIZE
  PROFILES_USE_LLM REDACTION_MODE NO_PAUSE STRICT_ENRICH HEARTBEAT_SECONDS PROGRESS_VERBOSITY
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --target-max)
      TARGET_MAX="$2"
      shift 2
      ;;
    --enrich-limit)
      ENRICH_BATCH_SIZE="$2"
      shift 2
      ;;
    --enrich-batch-size)
      ENRICH_BATCH_SIZE="$2"
      shift 2
      ;;
    --profiles-limit)
      PROFILES_LIMIT="$2"
      shift 2
      ;;
    --index-batch-size)
      INDEX_BATCH_SIZE="$2"
      shift 2
      ;;
    --profiles-use-llm)
      PROFILES_USE_LLM=1
      shift
      ;;
    --redaction-mode)
      REDACTION_MODE="$2"
      shift 2
      ;;
    --no-pause)
      NO_PAUSE=1
      shift
      ;;
    --strict-enrich)
      STRICT_ENRICH=1
      shift
      ;;
    --heartbeat-seconds)
      HEARTBEAT_SECONDS="$2"
      shift 2
      ;;
    --progress-verbosity)
      PROGRESS_VERBOSITY="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG" >&2
  exit 1
fi

if ! [[ "$TARGET_MAX" =~ ^[0-9]+$ ]]; then
  echo "Expected integer value for --target-max, got: $TARGET_MAX" >&2
  exit 2
fi

for num in "$ENRICH_BATCH_SIZE" "$PROFILES_LIMIT" "$INDEX_BATCH_SIZE"; do
  if ! [[ "$num" =~ ^[0-9]+$ ]]; then
    echo "Expected integer value, got: $num" >&2
    exit 2
  fi
  if [[ "$num" -lt 1 ]]; then
    echo "Expected integer >= 1, got: $num" >&2
    exit 2
  fi
done

if ! [[ "$HEARTBEAT_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "Expected integer value for HEARTBEAT_SECONDS, got: $HEARTBEAT_SECONDS" >&2
  exit 2
fi
if [[ "$HEARTBEAT_SECONDS" -lt 1 ]]; then
  echo "Expected HEARTBEAT_SECONDS >= 1, got: $HEARTBEAT_SECONDS" >&2
  exit 2
fi

if [[ "$REDACTION_MODE" != "regex" && "$REDACTION_MODE" != "model" && "$REDACTION_MODE" != "hybrid" ]]; then
  echo "--redaction-mode must be one of: regex, model, hybrid" >&2
  exit 2
fi

if [[ "$STRICT_ENRICH" != "0" && "$STRICT_ENRICH" != "1" ]]; then
  echo "STRICT_ENRICH must be 0 or 1" >&2
  exit 2
fi

if [[ "$PROGRESS_VERBOSITY" != "detailed" && "$PROGRESS_VERBOSITY" != "compact" && "$PROGRESS_VERBOSITY" != "quiet" ]]; then
  echo "PROGRESS_VERBOSITY must be one of: detailed, compact, quiet" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

acquire_lock() {
  if [[ -f "$LOCK_FILE" ]]; then
    local existing_pid
    existing_pid="$(awk -F= '/^pid=/{print $2}' "$LOCK_FILE" 2>/dev/null || true)"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "Another run_build_stepwise.sh is already running (pid=$existing_pid)." >&2
      echo "Lock file: $LOCK_FILE" >&2
      exit 1
    fi
    echo "Stale lock detected; removing $LOCK_FILE"
    rm -f "$LOCK_FILE"
  fi

  cat >"$LOCK_FILE" <<EOF
pid=$$
started_at_utc=$STARTED_AT_UTC
config=$CONFIG
EOF
}

release_lock() {
  rm -f "$LOCK_FILE"
}

trap release_lock EXIT INT TERM
acquire_lock

pause_if_needed() {
  local stage="$1"
  if [[ "$NO_PAUSE" == "1" ]]; then
    return
  fi
  echo
  read -r -p "Stage '$stage' complete. Press Enter to continue... " _
}

stage_description() {
  local stage="$1"
  case "$stage" in
    validate)
      echo "Quick safety check: confirm required DB tables and contracts exist before running heavy work."
      ;;
    backfill_to_target)
      echo "Backfill fetches older messages from Gmail up to your target cap and stores them encrypted in the local DB."
      ;;
    enrich_pending)
      echo "Enrich reads not-yet-processed messages and adds structured summaries/metadata so later search and profile steps are smarter."
      ;;
    build_profiles)
      echo "Build profiles analyzes message history to refresh contact-level profile cards used for better context."
      ;;
    index_vectors)
      echo "Index vectors normalizes text, applies redaction, generates embeddings, and upserts chunk vectors for semantic search."
      echo "This can be slow because each changed message may trigger multiple embedding calls and DB writes (message + chunks)."
      ;;
    inspect_summary)
      echo "Inspect summary prints final table counts so you can verify ingest/enrich/index coverage at a glance."
      ;;
  esac
}

print_stage_banner() {
  local stage="$1"
  echo
  echo "========== Stage: $stage =========="
  stage_description "$stage"
}

LAST_CHUNK_OP_HINT=""
LAST_CHUNK_ERR_LINE=""
LAST_CHUNK_ELAPSED=0

format_duration() {
  local seconds="${1:-0}"
  if [[ "$seconds" -lt 0 ]]; then
    seconds=0
  fi
  local h=$((seconds / 3600))
  local m=$(((seconds % 3600) / 60))
  local s=$((seconds % 60))
  if [[ "$h" -gt 0 ]]; then
    printf "%dh%02dm%02ds" "$h" "$m" "$s"
  else
    printf "%02dm%02ds" "$m" "$s"
  fi
}

truncate_line() {
  local text="$1"
  local max_len="${2:-160}"
  text="${text//$'\r'/}"
  if [[ "${#text}" -le "$max_len" ]]; then
    printf "%s" "$text"
    return
  fi
  printf "%s..." "${text:0:max_len}"
}

categorize_stderr_hint() {
  local raw="$1"
  local line
  line="$(echo "$raw" | tr '[:upper:]' '[:lower:]')"
  if [[ "$line" == *"embed"* || "$line" == *"/embeddings"* ]]; then
    printf "embedding-call"
  elif [[ "$line" == *"redact"* ]]; then
    printf "redaction"
  elif [[ "$line" == *"upsert"* || "$line" == *"insert into"* || "$line" == *"delete from message_chunk_vectors_v2"* ]]; then
    printf "db-upsert"
  elif [[ "$line" == *"lock"* || "$line" == *"busy"* || "$line" == *"retry"* ]]; then
    printf "lock-retry"
  elif [[ "$line" == *"parse"* && "$line" == *"failed"* ]]; then
    printf "parse-failure"
  elif [[ "$line" == *"http"* || "$line" == *"status"* || "$line" == *"timeout"* ]]; then
    printf "network/http"
  else
    printf "working"
  fi
}

emit_progress_line() {
  local line="$1"
  if [[ "$PROGRESS_VERBOSITY" == "quiet" ]]; then
    return
  fi
  echo "$line"
}

rate_per_min() {
  local completed="$1"
  local elapsed="$2"
  python3 - "$completed" "$elapsed" <<'PY2'
import sys
c = float(sys.argv[1])
e = float(sys.argv[2])
if e <= 0:
    print("n/a")
else:
    print(f"{(c * 60.0 / e):.2f}")
PY2
}

eta_from_progress() {
  local completed="$1"
  local remaining="$2"
  local elapsed="$3"
  if [[ "$completed" -le 0 || "$remaining" -le 0 || "$elapsed" -le 0 ]]; then
    printf "estimating"
    return
  fi
  local eta_seconds=$((remaining * elapsed / completed))
  format_duration "$eta_seconds"
}

render_progress() {
  local stage="$1"
  local description="$2"
  local completed="$3"
  local total="$4"
  local elapsed_seconds="${5:-0}"
  local remaining=$((total - completed))
  if [[ "$remaining" -lt 0 ]]; then
    remaining=0
  fi
  local rate
  local eta
  rate="$(rate_per_min "$completed" "$elapsed_seconds")"
  eta="$(eta_from_progress "$completed" "$remaining" "$elapsed_seconds")"
  emit_progress_line "[$stage] ${description}: completed=${completed}/${total} remaining=${remaining} elapsed=$(format_duration "$elapsed_seconds") rate=${rate}/min eta=${eta}"
}

print_enrich_batch_summary() {
  local batch_number="$1"
  local batch_elapsed="$2"
  local cumulative_elapsed="$3"
  local attempted="$4"
  local succeeded="$5"
  local parse_failed="$6"
  local http_failed="$7"
  local updated_batch="$8"
  local remaining_pending="$9"
  local cumulative_completed="${10}"

  local rate
  local eta
  rate="$(rate_per_min "$cumulative_completed" "$cumulative_elapsed")"
  eta="$(eta_from_progress "$cumulative_completed" "$remaining_pending" "$cumulative_elapsed")"

  emit_progress_line "[enrich_pending][batch ${batch_number}] elapsed_batch=$(format_duration "$batch_elapsed") elapsed_total=$(format_duration "$cumulative_elapsed") attempted=${attempted} succeeded=${succeeded} parse_failed=${parse_failed} http_failed=${http_failed} updated_batch=${updated_batch} remaining_pending=${remaining_pending} rate=${rate}/min eta=${eta}"

  if [[ "$PROGRESS_VERBOSITY" == "detailed" ]]; then
    emit_progress_line "[enrich_pending][batch ${batch_number}] last_op=${LAST_CHUNK_OP_HINT} last_hint=$(truncate_line "$LAST_CHUNK_ERR_LINE" 140)"
  fi
}

print_index_batch_summary() {
  local batch_number="$1"
  local batch_elapsed="$2"
  local cumulative_elapsed="$3"
  local scanned="$4"
  local indexed="$5"
  local unchanged="$6"
  local failed="$7"
  local chunks_indexed="$8"
  local lock_retries="$9"
  local lock_errors="${10}"
  local remaining_pending="${11}"
  local cumulative_completed="${12}"

  local rate
  local eta
  rate="$(rate_per_min "$cumulative_completed" "$cumulative_elapsed")"
  eta="$(eta_from_progress "$cumulative_completed" "$remaining_pending" "$cumulative_elapsed")"

  emit_progress_line "[index_vectors][batch ${batch_number}] elapsed_batch=$(format_duration "$batch_elapsed") elapsed_total=$(format_duration "$cumulative_elapsed") scanned=${scanned} indexed=${indexed} unchanged=${unchanged} failed=${failed} chunks_indexed=${chunks_indexed} lock_retries=${lock_retries} lock_errors=${lock_errors} remaining_pending=${remaining_pending} rate=${rate}/min eta=${eta}"

  if [[ "$PROGRESS_VERBOSITY" == "detailed" ]]; then
    emit_progress_line "[index_vectors][batch ${batch_number}] last_op=${LAST_CHUNK_OP_HINT} last_hint=$(truncate_line "$LAST_CHUNK_ERR_LINE" 140)"
  fi
}

run_chunk_with_progress() {
  local stage="$1"
  local iteration_kind="$2"
  local iteration="$3"
  local known_remaining="$4"
  local chunk_limit="$5"
  local stage_completed_before="$6"
  local stage_total="$7"
  local pending_counter_fn="$8"
  local status_hint="$9"
  local stdout_tmp="${10}"
  local stderr_log="${11}"
  shift 11

  local chunk_stderr
  chunk_stderr="$(mktemp)"

  "$@" >"$stdout_tmp" 2>"$chunk_stderr" &
  local cmd_pid=$!
  local start_epoch
  start_epoch="$(date +%s)"
  local last_emit_epoch="$start_epoch"
  local last_probe_epoch="$start_epoch"
  local last_stderr_line=""
  local last_op_hint="$status_hint"
  local last_emitted_op="$status_hint"
  local stderr_lines_read=0

  while kill -0 "$cmd_pid" 2>/dev/null; do
    sleep 1
    local now_epoch
    now_epoch="$(date +%s)"

    local latest_stderr_line=""
    if [[ -s "$chunk_stderr" ]]; then
      local total_lines
      total_lines="$(wc -l <"$chunk_stderr" | tr -d ' ')"
      if [[ "$total_lines" =~ ^[0-9]+$ ]] && [[ "$total_lines" -gt "$stderr_lines_read" ]]; then
        while IFS= read -r latest_stderr_line; do
          last_stderr_line="$latest_stderr_line"
          if [[ "$latest_stderr_line" == \[index-progress\]* ]]; then
            last_op_hint="indexing"
            if [[ "$PROGRESS_VERBOSITY" == "detailed" ]]; then
              emit_progress_line "$latest_stderr_line"
            elif [[ "$PROGRESS_VERBOSITY" == "compact" && "$latest_stderr_line" == *" done | "* ]]; then
              emit_progress_line "$latest_stderr_line"
            fi
          elif [[ -n "$latest_stderr_line" ]]; then
            last_op_hint="$(categorize_stderr_hint "$latest_stderr_line")"
          fi
        done < <(sed -n "$((stderr_lines_read + 1)),${total_lines}p" "$chunk_stderr")
        stderr_lines_read="$total_lines"
      fi
    fi

    local live_remaining="$known_remaining"
    local batch_done="n/a"
    if [[ -n "$pending_counter_fn" ]] && (( now_epoch - last_probe_epoch >= HEARTBEAT_SECONDS )); then
      local pending_now
      if pending_now="$($pending_counter_fn 2>/dev/null)" && [[ "$pending_now" =~ ^[0-9]+$ ]]; then
        live_remaining="$pending_now"
        local live_completed=$((stage_total - pending_now))
        if [[ "$live_completed" -lt 0 ]]; then
          live_completed=0
        fi
        local done_now=$((live_completed - stage_completed_before))
        if [[ "$done_now" -lt 0 ]]; then
          done_now=0
        fi
        batch_done="$done_now"
      fi
      last_probe_epoch="$now_epoch"
    fi

    if [[ "$PROGRESS_VERBOSITY" == "detailed" ]]; then
      if (( now_epoch - last_emit_epoch >= HEARTBEAT_SECONDS )) || [[ "$last_op_hint" != "$last_emitted_op" ]]; then
        local elapsed=$((now_epoch - start_epoch))
        emit_progress_line "[heartbeat][$stage ${iteration_kind} ${iteration}] elapsed=$(format_duration "$elapsed") limit=${chunk_limit} known_remaining=${live_remaining} batch_done=${batch_done} op=${last_op_hint} hint=$(truncate_line "$last_stderr_line" 120)"
        last_emit_epoch="$now_epoch"
        last_emitted_op="$last_op_hint"
      fi
    elif [[ "$PROGRESS_VERBOSITY" == "compact" ]]; then
      if (( now_epoch - last_emit_epoch >= HEARTBEAT_SECONDS )); then
        local elapsed=$((now_epoch - start_epoch))
        emit_progress_line "[heartbeat][$stage ${iteration_kind} ${iteration}] elapsed=$(format_duration "$elapsed") op=${last_op_hint}"
        last_emit_epoch="$now_epoch"
      fi
    fi
  done

  local cmd_status=0
  if wait "$cmd_pid"; then
    cmd_status=0
  else
    cmd_status=$?
  fi

  cat "$chunk_stderr" >>"$stderr_log"

  LAST_CHUNK_ELAPSED=$(( $(date +%s) - start_epoch ))
  LAST_CHUNK_OP_HINT="$last_op_hint"
  LAST_CHUNK_ERR_LINE="$last_stderr_line"

  rm -f "$chunk_stderr"
  return "$cmd_status"
}

db_count() {
  local query="$1"
  python3 - "$CONFIG" "$query" <<'PY'
import sys
from inbox_vault.config import load_config, resolve_password
from inbox_vault.db import get_conn

cfg = load_config(sys.argv[1])
conn = get_conn(cfg.db.path, resolve_password(cfg.db))
try:
    row = conn.execute(sys.argv[2]).fetchone()
    print(int(row[0]) if row and row[0] is not None else 0)
finally:
    conn.close()
PY
}

count_enrich_pending() {
  db_count "SELECT COUNT(*) FROM messages m LEFT JOIN message_enrichment e ON e.msg_id = m.msg_id WHERE e.msg_id IS NULL"
}

count_index_pending() {
  python3 - "$CONFIG" <<'PY'
import sys
from inbox_vault.config import load_config, resolve_password
from inbox_vault.db import get_conn
from inbox_vault.vectors import count_pending_vector_updates

cfg = load_config(sys.argv[1])
conn = get_conn(cfg.db.path, resolve_password(cfg.db))
try:
    print(count_pending_vector_updates(conn, cfg))
finally:
    conn.close()
PY
}

estimate_index_batch_window() {
  local limit="$1"
  python3 - "$CONFIG" "$limit" <<'PY'
import sys
from inbox_vault.config import load_config, resolve_password
from inbox_vault.db import get_conn, vector_index_source_rows
from inbox_vault.vectors import _build_chunks, _normalize_for_indexing, _pending_message_ids_for_index

cfg = load_config(sys.argv[1])
limit = max(1, int(sys.argv[2]))
conn = get_conn(cfg.db.path, resolve_password(cfg.db))
try:
    pending_ids = _pending_message_ids_for_index(conn, cfg, limit=limit)
    pending_id_set = set(pending_ids)
    if not pending_id_set:
        print("0\t0")
        raise SystemExit(0)

    pending_chunks = 0
    rows = vector_index_source_rows(conn, limit=None)
    max_chars = max(1, int(cfg.indexing.max_index_chars))
    for msg_id, _acct, _thread_id, subject, _snippet, body_text, _labels_json in rows:
        if msg_id not in pending_id_set:
            continue
        normalized_subject = _normalize_for_indexing(
            subject,
            strip_zero_width=cfg.indexing.strip_zero_width,
            collapse_whitespace=cfg.indexing.collapse_whitespace,
            max_chars=max_chars,
        )
        normalized_body = _normalize_for_indexing(
            body_text,
            strip_zero_width=cfg.indexing.strip_zero_width,
            collapse_whitespace=cfg.indexing.collapse_whitespace,
            max_chars=max_chars,
        )
        pending_chunks += len(
            _build_chunks(
                msg_id,
                subject=normalized_subject,
                body_text=normalized_body,
                chunk_chars=cfg.retrieval.chunk_chars,
                overlap_chars=cfg.retrieval.chunk_overlap_chars,
            )
        )

    print(f"{len(pending_ids)}\t{pending_chunks}")
finally:
    conn.close()
PY
}

snapshot_account_message_stats() {
  python3 - "$CONFIG" <<'PY'
import json
import sys

from inbox_vault.config import load_config, resolve_password
from inbox_vault.db import get_conn

cfg = load_config(sys.argv[1])
conn = get_conn(cfg.db.path, resolve_password(cfg.db))
try:
    rows = conn.execute(
        """
        SELECT account_email,
               COUNT(*) AS total_messages,
               MIN(COALESCE(date_iso, '')) AS oldest_date,
               MAX(COALESCE(date_iso, '')) AS newest_date
        FROM messages
        GROUP BY account_email
        """
    ).fetchall()

    by_account = {
        str(email): {
            "account_email": str(email),
            "total_messages": int(total or 0),
            "oldest_date": str(oldest or ""),
            "newest_date": str(newest or ""),
        }
        for email, total, oldest, newest in rows
    }

    out = []
    for acct in cfg.accounts:
        email = acct.email
        out.append(
            by_account.get(
                email,
                {
                    "account_email": email,
                    "total_messages": 0,
                    "oldest_date": "",
                    "newest_date": "",
                },
            )
        )

    print(json.dumps(out))
finally:
    conn.close()
PY
}

print_backfill_account_summary() {
  local before_json="$1"
  local after_json="$2"
  python3 - "$before_json" "$after_json" <<'PY'
import json
import sys
from datetime import datetime

before = {row["account_email"]: row for row in json.loads(sys.argv[1])}
after = {row["account_email"]: row for row in json.loads(sys.argv[2])}


def fmt_date(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return "n/a"
    for pattern in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(raw, pattern)
            return dt.strftime("%b %d, %Y")
        except ValueError:
            continue
    return raw

header = (
    "[backfill_to_target] Per-account results\n"
    "account_email                         before    delta    total    oldest                newest"
)
print(header)

for email in sorted(set(before) | set(after)):
    b = before.get(email, {})
    a = after.get(email, {})
    b_total = int(b.get("total_messages", 0) or 0)
    a_total = int(a.get("total_messages", 0) or 0)
    delta = a_total - b_total
    oldest = fmt_date(a.get("oldest_date") or b.get("oldest_date") or "")
    newest = fmt_date(a.get("newest_date") or b.get("newest_date") or "")
    print(f"{email:<36} {b_total:>7} {delta:>8} {a_total:>8}  {oldest:<20} {newest:<20}")
PY
}

run_backfill_stage() {
  local stage="backfill_to_target"
  local stage_log="$LOG_DIR/${stage}.json"
  local stdout_tmp
  local stderr_log
  local finished_at
  stdout_tmp="$(mktemp)"
  stderr_log="$LOG_DIR/${stage}.stderr.log"

  local before_snapshot
  before_snapshot="$(snapshot_account_message_stats)"

  print_stage_banner "$stage"
  echo "Command: ${IV_CMD[*]} backfill --max-messages $TARGET_MAX"

  local cmd_status=0
  if "${IV_CMD[@]}" backfill --max-messages "$TARGET_MAX" >"$stdout_tmp" 2>"$stderr_log"; then
    cmd_status=0
  else
    cmd_status=$?
  fi
  finished_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  python3 - "$stage" "$cmd_status" "$finished_at" "$stdout_tmp" "$stderr_log" "$stage_log" <<'PY'
import json
import pathlib
import sys

stage = sys.argv[1]
status = int(sys.argv[2])
finished_at = sys.argv[3]
stdout_path = pathlib.Path(sys.argv[4])
stderr_path = pathlib.Path(sys.argv[5])
out_path = pathlib.Path(sys.argv[6])

stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")

payload = {
    "stage": stage,
    "ok": status == 0,
    "exit_code": status,
    "finished_at_utc": finished_at,
    "stderr": stderr_text.strip(),
}
if stdout_text.strip():
    try:
        payload["result"] = json.loads(stdout_text)
    except Exception:
        payload["stdout_raw"] = stdout_text
else:
    payload["result"] = None

out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

  local after_snapshot
  after_snapshot="$(snapshot_account_message_stats)"
  print_backfill_account_summary "$before_snapshot" "$after_snapshot"
  echo "[${stage}] machine_log=$stage_log"

  rm -f "$stdout_tmp"

  if [[ "$cmd_status" -ne 0 ]]; then
    echo "Stage '$stage' failed. See: $stage_log" >&2
    exit "$cmd_status"
  fi

  pause_if_needed "$stage"
}

run_json_cmd() {
  local stage="$1"
  shift

  local stage_log="$LOG_DIR/${stage}.json"
  local stdout_tmp
  local stderr_log
  local finished_at
  stdout_tmp="$(mktemp)"
  stderr_log="$LOG_DIR/${stage}.stderr.log"

  print_stage_banner "$stage"
  echo "Command: $*"

  local cmd_status=0
  if "$@" >"$stdout_tmp" 2>"$stderr_log"; then
    cmd_status=0
  else
    cmd_status=$?
  fi
  finished_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  python3 - "$stage" "$cmd_status" "$finished_at" "$stdout_tmp" "$stderr_log" "$stage_log" <<'PY'
import json
import pathlib
import sys

stage = sys.argv[1]
status = int(sys.argv[2])
finished_at = sys.argv[3]
stdout_path = pathlib.Path(sys.argv[4])
stderr_path = pathlib.Path(sys.argv[5])
out_path = pathlib.Path(sys.argv[6])

stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")

payload = {
    "stage": stage,
    "ok": status == 0,
    "exit_code": status,
    "finished_at_utc": finished_at,
    "stderr": stderr_text.strip(),
}

if stdout_text.strip():
    try:
        payload["result"] = json.loads(stdout_text)
    except Exception:
        payload["stdout_raw"] = stdout_text
else:
    payload["result"] = None

out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

  python3 - "$stage_log" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
stage = payload.get("stage")
ok = payload.get("ok")
result = payload.get("result") or {}

print(f"[{stage}] status={'ok' if ok else 'failed'}")
if stage == "validate":
    checks = [k for k, v in result.items() if k != "ok" and v]
    print(f"[{stage}] passed_checks={len(checks)}")
elif stage == "build_profiles":
    updated = int(result.get("updated", 0) or 0)
    diag = result.get("diagnostics") or {}
    attempted = int(diag.get("attempted", 0) or 0)
    print(f"[{stage}] profiles_updated={updated} attempted={attempted}")
elif stage == "backfill_to_target":
    ingest = result.get("ingest") or {}
    print(
        f"[{stage}] ingested={int(ingest.get('ingested', 0) or 0)} "
        f"skipped_existing={int(ingest.get('skipped_existing', 0) or 0)} failed={int(ingest.get('failed', 0) or 0)}"
    )
print(f"[{stage}] machine_log={path}")
PY

  rm -f "$stdout_tmp"

  if [[ "$cmd_status" -ne 0 ]]; then
    echo "Stage '$stage' failed. See: $stage_log" >&2
    exit "$cmd_status"
  fi

  pause_if_needed "$stage"
}

run_enrich_pending_stage() {
  local stage="enrich_pending"
  local stage_log="$LOG_DIR/${stage}.json"
  local stderr_log="$LOG_DIR/${stage}.stderr.log"

  print_stage_banner "$stage"
  : >"$stderr_log"

  local initial_pending
  initial_pending="$(count_enrich_pending)"
  local current_pending="$initial_pending"
  local total_updated=0
  local iteration=0
  local chunks_json='[]'
  local stalled_no_progress=0
  local stage_started_epoch
  stage_started_epoch="$(date +%s)"

  render_progress "$stage" "enrich pending" 0 "$initial_pending" 0

  while [[ "$current_pending" -gt 0 ]]; do
    iteration=$((iteration + 1))

    local stdout_tmp
    stdout_tmp="$(mktemp)"
    local chunk_status=0
    local completed_before=$((initial_pending - current_pending))
    if [[ "$completed_before" -lt 0 ]]; then
      completed_before=0
    fi

    if run_chunk_with_progress "$stage" "batch" "$iteration" "$current_pending" "$ENRICH_BATCH_SIZE" "$completed_before" "$initial_pending" "count_enrich_pending" "llm/enrich" "$stdout_tmp" "$stderr_log" "${IV_CMD[@]}" enrich --limit "$ENRICH_BATCH_SIZE"; then
      chunk_status=0
    else
      chunk_status=$?
    fi

    local chunk_json
    chunk_json="$(cat "$stdout_tmp")"
    rm -f "$stdout_tmp"

    local chunk_updated=0
    local chunk_attempted=0
    local chunk_succeeded=0
    local chunk_parse_failed=0
    local chunk_http_failed=0
    IFS=$'\t' read -r chunk_updated chunk_attempted chunk_succeeded chunk_parse_failed chunk_http_failed < <(python3 - "$chunk_json" <<'PY'
import json
import sys

raw = sys.argv[1].strip()
if not raw:
    print("0\t0\t0\t0\t0")
    raise SystemExit(0)
try:
    payload = json.loads(raw)
except Exception:
    print("0\t0\t0\t0\t0")
    raise SystemExit(0)
d = payload.get("diagnostics") or {}
vals = [
    int(payload.get("updated", 0) or 0),
    int(d.get("attempted", 0) or 0),
    int(d.get("succeeded", 0) or 0),
    int(d.get("parse_failed", 0) or 0),
    int(d.get("http_failed", 0) or 0),
]
print("\t".join(str(v) for v in vals))
PY
)

    current_pending="$(count_enrich_pending)"
    local completed=$((initial_pending - current_pending))
    if [[ "$completed" -lt 0 ]]; then
      completed=0
    fi

    local now_epoch
    now_epoch="$(date +%s)"
    local elapsed_seconds=$((now_epoch - stage_started_epoch))

    render_progress "$stage" "enrich pending (batch $iteration, limit=$ENRICH_BATCH_SIZE)" "$completed" "$initial_pending" "$elapsed_seconds"
    print_enrich_batch_summary "$iteration" "$LAST_CHUNK_ELAPSED" "$elapsed_seconds" "$chunk_attempted" "$chunk_succeeded" "$chunk_parse_failed" "$chunk_http_failed" "$chunk_updated" "$current_pending" "$completed"

    total_updated=$((total_updated + chunk_updated))

    chunks_json="$(python3 - "$chunks_json" "$iteration" "$chunk_status" "$chunk_updated" "$current_pending" "$chunk_json" <<'PY'
import json
import sys

chunks = json.loads(sys.argv[1])
iteration = int(sys.argv[2])
status = int(sys.argv[3])
updated = int(sys.argv[4])
remaining = int(sys.argv[5])
raw = sys.argv[6].strip()
result = None
if raw:
    try:
        result = json.loads(raw)
    except Exception:
        result = {"raw": raw}
chunks.append(
    {
        "iteration": iteration,
        "exit_code": status,
        "updated": updated,
        "remaining_pending": remaining,
        "result": result,
    }
)
print(json.dumps(chunks))
PY
)"

    if [[ "$chunk_status" -ne 0 ]]; then
      break
    fi

    if [[ "$chunk_updated" -eq 0 && "$current_pending" -gt 0 ]]; then
      stalled_no_progress=1
      echo "No enrich progress in latest batch; stopping loop to avoid infinite retries." | tee -a "$stderr_log" >&2
      break
    fi
  done

  local finished_at
  finished_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  python3 - "$stage" "$finished_at" "$initial_pending" "$current_pending" "$total_updated" "$chunks_json" "$stderr_log" "$stage_log" "$stalled_no_progress" "$STRICT_ENRICH" <<'PY'
import json
import pathlib
import sys

stage = sys.argv[1]
finished_at = sys.argv[2]
initial_pending = int(sys.argv[3])
remaining_pending = int(sys.argv[4])
updated_total = int(sys.argv[5])
chunks = json.loads(sys.argv[6])
stderr_text = pathlib.Path(sys.argv[7]).read_text(encoding="utf-8", errors="replace").strip()
out_path = pathlib.Path(sys.argv[8])
stalled_no_progress = int(sys.argv[9]) == 1
strict_enrich = int(sys.argv[10]) == 1

ok = remaining_pending == 0
warning = None
proceed = ok

if not ok and stalled_no_progress:
    warning = (
        f"enrich stalled with {remaining_pending} pending rows; rerun resumes from pending only"
    )
    if strict_enrich:
        proceed = False
    else:
        ok = True
        proceed = True

payload = {
    "stage": stage,
    "ok": ok,
    "exit_code": 0 if ok else 1,
    "proceed": proceed,
    "finished_at_utc": finished_at,
    "stderr": stderr_text,
    "result": {
        "initial_pending": initial_pending,
        "remaining_pending": remaining_pending,
        "completed": max(0, initial_pending - remaining_pending),
        "updated_total": updated_total,
        "batch_size": None,
        "stalled_no_progress": stalled_no_progress,
        "strict_enrich": strict_enrich,
        "chunks": chunks,
    },
}
if warning:
    payload["warning"] = warning
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

  # Fill batch_size directly in JSON without reparsing bash arrays.
  python3 - "$stage_log" "$ENRICH_BATCH_SIZE" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
batch_size = int(sys.argv[2])
payload = json.loads(path.read_text(encoding="utf-8"))
payload.setdefault("result", {})["batch_size"] = batch_size
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

  python3 - "$stage_log" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
result = payload.get("result") or {}
print(
    "[enrich_pending] "
    f"completed={int(result.get('completed', 0) or 0)}/"
    f"{int(result.get('initial_pending', 0) or 0)} "
    f"remaining={int(result.get('remaining_pending', 0) or 0)} "
    f"updated_total={int(result.get('updated_total', 0) or 0)}"
)
warning = payload.get("warning")
if warning:
    print(f"[enrich_pending] warning={warning}")
print(f"[enrich_pending] machine_log={sys.argv[1]}")
PY

  if [[ "$(python3 - "$stage_log" <<'PY'
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(1 if payload.get("ok") else 0)
PY
)" != "1" ]]; then
    echo "Stage '$stage' failed to drain pending enrich work. See: $stage_log" >&2
    exit 1
  fi

  pause_if_needed "$stage"
}

run_index_vectors_stage() {
  local stage="index_vectors"
  local stage_log="$LOG_DIR/${stage}.json"
  local stderr_log="$LOG_DIR/${stage}.stderr.log"

  print_stage_banner "$stage"
  : >"$stderr_log"

  local initial_pending
  initial_pending="$(count_index_pending)"
  local current_pending="$initial_pending"
  local iteration=0
  local chunks_json='[]'
  local stage_started_epoch
  stage_started_epoch="$(date +%s)"

  render_progress "$stage" "index vectors" 0 "$initial_pending" 0

  while [[ "$current_pending" -gt 0 ]]; do
    iteration=$((iteration + 1))

    local chunk_limit="$INDEX_BATCH_SIZE"

    local stdout_tmp
    stdout_tmp="$(mktemp)"
    local chunk_status=0
    local completed_before=$((initial_pending - current_pending))
    if [[ "$completed_before" -lt 0 ]]; then
      completed_before=0
    fi

    local est_window_messages=0
    local est_window_chunks=0
    IFS=$'\t' read -r est_window_messages est_window_chunks < <(estimate_index_batch_window "$chunk_limit")
    emit_progress_line "[index_vectors][batch ${iteration} preflight] limit=${chunk_limit} est_messages=${est_window_messages} est_chunks=${est_window_chunks}"

    if run_chunk_with_progress "$stage" "batch" "$iteration" "$current_pending" "$chunk_limit" "$completed_before" "$initial_pending" "count_index_pending" "embeddings/index" "$stdout_tmp" "$stderr_log" "${IV_CMD[@]}" index-vectors --pending-only --limit "$chunk_limit" --redaction-mode "$REDACTION_MODE"; then
      chunk_status=0
    else
      chunk_status=$?
    fi

    local chunk_json
    chunk_json="$(cat "$stdout_tmp")"
    rm -f "$stdout_tmp"

    local chunk_scanned=0
    local chunk_indexed=0
    local chunk_unchanged=0
    local chunk_failed=0
    local chunk_chunks_indexed=0
    local chunk_lock_retries=0
    local chunk_lock_errors=0
    IFS=$'\t' read -r chunk_scanned chunk_indexed chunk_unchanged chunk_failed chunk_chunks_indexed chunk_lock_retries chunk_lock_errors < <(python3 - "$chunk_json" <<'PY'
import json
import sys

raw = sys.argv[1].strip()
if not raw:
    print("0\t0\t0\t0\t0\t0\t0")
    raise SystemExit(0)
try:
    payload = json.loads(raw)
except Exception:
    print("0\t0\t0\t0\t0\t0\t0")
    raise SystemExit(0)
vals = [
    int(payload.get("scanned", 0) or 0),
    int(payload.get("indexed", 0) or 0),
    int(payload.get("unchanged", 0) or 0),
    int(payload.get("failed", 0) or 0),
    int(payload.get("chunks_indexed", 0) or 0),
    int(payload.get("lock_retries", 0) or 0),
    int(payload.get("lock_errors", 0) or 0),
]
print("\t".join(str(v) for v in vals))
PY
)

    current_pending="$(count_index_pending)"
    local completed=$((initial_pending - current_pending))
    if [[ "$completed" -lt 0 ]]; then
      completed=0
    fi

    local now_epoch
    now_epoch="$(date +%s)"
    local elapsed_seconds=$((now_epoch - stage_started_epoch))

    render_progress "$stage" "index vectors (chunk $iteration, limit=$chunk_limit)" "$completed" "$initial_pending" "$elapsed_seconds"
    print_index_batch_summary "$iteration" "$LAST_CHUNK_ELAPSED" "$elapsed_seconds" "$chunk_scanned" "$chunk_indexed" "$chunk_unchanged" "$chunk_failed" "$chunk_chunks_indexed" "$chunk_lock_retries" "$chunk_lock_errors" "$current_pending" "$completed"

    chunks_json="$(python3 - "$chunks_json" "$iteration" "$chunk_limit" "$chunk_status" "$chunk_indexed" "$current_pending" "$chunk_json" <<'PY'
import json
import sys

chunks = json.loads(sys.argv[1])
iteration = int(sys.argv[2])
chunk_limit = int(sys.argv[3])
status = int(sys.argv[4])
indexed = int(sys.argv[5])
remaining = int(sys.argv[6])
raw = sys.argv[7].strip()
result = None
if raw:
    try:
        result = json.loads(raw)
    except Exception:
        result = {"raw": raw}
chunks.append(
    {
        "iteration": iteration,
        "limit": chunk_limit,
        "exit_code": status,
        "indexed": indexed,
        "remaining_pending": remaining,
        "result": result,
    }
)
print(json.dumps(chunks))
PY
)"

    if [[ "$chunk_status" -ne 0 ]]; then
      break
    fi

    if [[ "$current_pending" -eq 0 ]]; then
      break
    fi

    if [[ "$chunk_indexed" -eq 0 && "$current_pending" -gt 0 ]]; then
      echo "No index progress in pending-only batch (limit=$chunk_limit, remaining_pending=$current_pending); stopping loop." | tee -a "$stderr_log" >&2
      break
    fi
  done

  local finished_at
  finished_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  python3 - "$stage" "$finished_at" "$initial_pending" "$current_pending" "$INDEX_BATCH_SIZE" "$chunks_json" "$stderr_log" "$stage_log" <<'PY'
import json
import pathlib
import sys

stage = sys.argv[1]
finished_at = sys.argv[2]
initial_pending = int(sys.argv[3])
remaining_pending = int(sys.argv[4])
index_batch_size = int(sys.argv[5])
chunks = json.loads(sys.argv[6])
stderr_text = pathlib.Path(sys.argv[7]).read_text(encoding="utf-8", errors="replace").strip()
out_path = pathlib.Path(sys.argv[8])

ok = remaining_pending == 0
payload = {
    "stage": stage,
    "ok": ok,
    "exit_code": 0 if ok else 1,
    "finished_at_utc": finished_at,
    "stderr": stderr_text,
    "result": {
        "initial_pending": initial_pending,
        "remaining_pending": remaining_pending,
        "completed": max(0, initial_pending - remaining_pending),
        "index_batch_size": index_batch_size,
        "chunks": chunks,
    },
}
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

  python3 - "$stage_log" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
result = payload.get("result") or {}
print(
    "[index_vectors] "
    f"completed={int(result.get('completed', 0) or 0)}/"
    f"{int(result.get('initial_pending', 0) or 0)} "
    f"remaining={int(result.get('remaining_pending', 0) or 0)}"
)
print(f"[index_vectors] machine_log={sys.argv[1]}")
PY

  if [[ "$(python3 - "$stage_log" <<'PY'
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(1 if payload.get("ok") else 0)
PY
)" != "1" ]]; then
    echo "Stage '$stage' failed to drain pending index work. See: $stage_log" >&2
    exit 1
  fi

  pause_if_needed "$stage"
}

run_inspect_summary() {
  local stage="inspect_summary"
  local stage_log="$LOG_DIR/${stage}.json"

  print_stage_banner "$stage"

  python3 - "$CONFIG" "$stage_log" <<'PY'
import json
import pathlib
import sys

from inbox_vault.config import load_config, resolve_password
from inbox_vault.db import get_conn

cfg_path = sys.argv[1]
out_path = pathlib.Path(sys.argv[2])

cfg = load_config(cfg_path)
password = resolve_password(cfg.db)
conn = get_conn(cfg.db.path, password)


def count(query: str) -> int:
    row = conn.execute(query).fetchone()
    return int(row[0]) if row else 0

summary = {
    "stage": "inspect_summary",
    "ok": True,
    "db_path": cfg.db.path,
    "counts": {
        "messages": count("SELECT COUNT(*) FROM messages"),
        "raw_messages": count("SELECT COUNT(*) FROM raw_messages"),
        "message_enrichment": count("SELECT COUNT(*) FROM message_enrichment"),
        "contact_profiles": count("SELECT COUNT(*) FROM contact_profiles"),
        "message_vectors_v2": count("SELECT COUNT(*) FROM message_vectors_v2"),
        "message_chunk_vectors_v2": count("SELECT COUNT(*) FROM message_chunk_vectors_v2"),
    },
}

conn.close()
out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(
    f"[inspect_summary] messages={summary['counts']['messages']} "
    f"enrichments={summary['counts']['message_enrichment']} "
    f"vectors={summary['counts']['message_vectors_v2']} chunks={summary['counts']['message_chunk_vectors_v2']}"
)
print(f"[inspect_summary] machine_log={out_path}")
PY

  pause_if_needed "$stage"
}

IV_CMD=(inbox-vault --config "$CONFIG")
PROFILE_ARGS=(--limit "$PROFILES_LIMIT")
if [[ "$PROFILES_USE_LLM" == "1" ]]; then
  PROFILE_ARGS+=(--use-llm)
fi

run_json_cmd validate "${IV_CMD[@]}" validate
run_backfill_stage
run_enrich_pending_stage
run_json_cmd build_profiles "${IV_CMD[@]}" build-profiles "${PROFILE_ARGS[@]}"
run_index_vectors_stage
run_inspect_summary

FINISHED_AT_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

python3 - "$LOG_DIR" "$CONFIG" "$TARGET_MAX" "$ENRICH_BATCH_SIZE" "$PROFILES_LIMIT" "$INDEX_BATCH_SIZE" "$PROFILES_USE_LLM" "$REDACTION_MODE" "$STRICT_ENRICH" "$STARTED_AT_UTC" "$FINISHED_AT_UTC" "$FINAL_SUMMARY_JSON" "$FINAL_SUMMARY_TXT" <<'PY'
import json
import pathlib
import sys

log_dir = pathlib.Path(sys.argv[1])
config_path = sys.argv[2]
target_max = int(sys.argv[3])
enrich_batch_size = int(sys.argv[4])
profiles_limit = int(sys.argv[5])
index_batch_size = int(sys.argv[6])
profiles_use_llm = sys.argv[7] == "1"
redaction_mode = sys.argv[8]
strict_enrich = sys.argv[9] == "1"
started_at = sys.argv[10]
finished_at = sys.argv[11]
summary_json = pathlib.Path(sys.argv[12])
summary_txt = pathlib.Path(sys.argv[13])

stage_files = [
    "validate.json",
    "backfill_to_target.json",
    "enrich_pending.json",
    "build_profiles.json",
    "index_vectors.json",
    "inspect_summary.json",
]

stages = {}
for name in stage_files:
    p = log_dir / name
    if p.exists():
        stages[name.removesuffix(".json")] = json.loads(p.read_text(encoding="utf-8"))

summary = {
    "ok": True,
    "started_at_utc": started_at,
    "finished_at_utc": finished_at,
    "config": config_path,
    "parameters": {
        "target_max": target_max,
        "enrich_batch_size": enrich_batch_size,
        "profiles_limit": profiles_limit,
        "index_batch_size": index_batch_size,
        "profiles_use_llm": profiles_use_llm,
        "redaction_mode": redaction_mode,
        "strict_enrich": strict_enrich,
    },
    "logs_dir": str(log_dir),
    "stages": stages,
}

summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

inspect_counts = stages.get("inspect_summary", {}).get("counts", {})
enrich_stage = stages.get("enrich_pending", {})
enrich_result = enrich_stage.get("result", {})
remaining_pending = enrich_result.get("remaining_pending")
if remaining_pending is None:
    enrich_status_line = "enrich_status=unknown"
elif int(remaining_pending) == 0:
    enrich_status_line = "enrich_status=complete (remaining_pending=0)"
else:
    enrich_status_line = f"enrich_status=partial (remaining_pending={remaining_pending})"

lines = [
    "run_build_stepwise complete",
    f"config: {config_path}",
    f"target_max={target_max} enrich_batch_size={enrich_batch_size} profiles_limit={profiles_limit}",
    f"index_batch_size={index_batch_size} profiles_use_llm={profiles_use_llm} redaction_mode={redaction_mode} strict_enrich={strict_enrich}",
    enrich_status_line,
    "rerun_note: reruns resume from pending-only enrich/index work; no full recompute",
    f"messages={inspect_counts.get('messages', 'n/a')} enrichments={inspect_counts.get('message_enrichment', 'n/a')}",
    f"vectors={inspect_counts.get('message_vectors_v2', 'n/a')} chunks={inspect_counts.get('message_chunk_vectors_v2', 'n/a')}",
    f"logs: {log_dir}",
    f"summary_json: {summary_json}",
]
summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

echo
echo "Done. Final summary: $FINAL_SUMMARY_TXT"
