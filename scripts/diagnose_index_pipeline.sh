#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-config.toml}"
SAMPLE_SIZE="${SAMPLE_SIZE:-10}"
INDEX_BATCH_SIZE="${INDEX_BATCH_SIZE:-5}"
REDACTION_MODE="${REDACTION_MODE:-regex}"
MAX_INDEX_CHARS_LIST="${MAX_INDEX_CHARS_LIST:-2000,3000,5000,8000}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"

usage() {
  cat <<'EOF'
Usage: scripts/diagnose_index_pipeline.sh [options]

Quick-fail indexing/embedding diagnostics over tiny batches.
Each max-index-chars trial runs incremental index-vectors passes and classifies failures.

Options:
  --config <path>                 Config path (default: config.toml)
  --sample-size <n>               Total indexing scope per trial (default: 10)
  --index-batch-size <n>          Increment step for --limit within a trial (default: 5)
  --redaction-mode <mode>         regex|model|hybrid (default: regex)
  --max-index-chars-list <csv>    Trial values (default: 2000,3000,5000,8000)
  --heartbeat-seconds <n>         Optional heartbeat cadence while command runs
  -h, --help                      Show this help

Environment overrides:
  CONFIG SAMPLE_SIZE INDEX_BATCH_SIZE REDACTION_MODE MAX_INDEX_CHARS_LIST HEARTBEAT_SECONDS TIMEOUT_SECONDS
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --sample-size)
      SAMPLE_SIZE="$2"
      shift 2
      ;;
    --index-batch-size)
      INDEX_BATCH_SIZE="$2"
      shift 2
      ;;
    --redaction-mode)
      REDACTION_MODE="$2"
      shift 2
      ;;
    --max-index-chars-list)
      MAX_INDEX_CHARS_LIST="$2"
      shift 2
      ;;
    --heartbeat-seconds)
      HEARTBEAT_SECONDS="$2"
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

for n in "$SAMPLE_SIZE" "$INDEX_BATCH_SIZE" "$TIMEOUT_SECONDS"; do
  if ! [[ "$n" =~ ^[0-9]+$ ]] || [[ "$n" -lt 1 ]]; then
    echo "Expected positive integer, got: $n" >&2
    exit 2
  fi
done

if [[ "$REDACTION_MODE" != "regex" && "$REDACTION_MODE" != "model" && "$REDACTION_MODE" != "hybrid" ]]; then
  echo "--redaction-mode must be one of: regex, model, hybrid" >&2
  exit 2
fi

if [[ -n "$HEARTBEAT_SECONDS" ]]; then
  if ! [[ "$HEARTBEAT_SECONDS" =~ ^[0-9]+$ ]] || [[ "$HEARTBEAT_SECONDS" -lt 1 ]]; then
    echo "--heartbeat-seconds must be a positive integer" >&2
    exit 2
  fi
fi

TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
RUN_ROOT=".runs/default/diagnostics/$TIMESTAMP"
mkdir -p "$RUN_ROOT"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT INT TERM

split_csv() {
  python3 - "$1" <<'PY'
import sys
raw = sys.argv[1]
items = [x.strip() for x in raw.split(',') if x.strip()]
for item in items:
    print(item)
PY
}

run_with_timeout() {
  local out_log="$1"
  local err_log="$2"
  shift 2

  timeout "$TIMEOUT_SECONDS" "$@" >>"$out_log" 2>>"$err_log" &
  local pid=$!

  if [[ -n "$HEARTBEAT_SECONDS" ]]; then
    while kill -0 "$pid" 2>/dev/null; do
      sleep "$HEARTBEAT_SECONDS"
      if kill -0 "$pid" 2>/dev/null; then
        echo "[heartbeat] still running (pid=$pid)" | tee -a "$err_log" >/dev/null
      fi
    done
  fi

  wait "$pid"
}

build_trial_config() {
  local base_cfg="$1"
  local out_cfg="$2"
  python3 - "$base_cfg" "$out_cfg" <<'PY'
import pathlib
import re
import sys

src = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
out_path = pathlib.Path(sys.argv[2])


def set_value(text: str, section: str, key: str, value_literal: str) -> str:
    section_pat = re.compile(rf"(?ms)^\[{re.escape(section)}\]\n(.*?)(?=^\[|\Z)")
    m = section_pat.search(text)
    if not m:
        append = f"\n[{section}]\n{key} = {value_literal}\n"
        return text + append

    body = m.group(1)
    key_pat = re.compile(rf"(?m)^{re.escape(key)}\s*=.*$")
    if key_pat.search(body):
      body_new = key_pat.sub(f"{key} = {value_literal}", body)
    else:
      body_new = body
      if body_new and not body_new.endswith("\n"):
          body_new += "\n"
      body_new += f"{key} = {value_literal}\n"

    return text[: m.start(1)] + body_new + text[m.end(1) :]

text = src
text = set_value(text, "embeddings", "fallback", '"none"')
text = set_value(text, "embeddings", "max_retries", "0")
text = set_value(text, "embeddings", "timeout_seconds", "4")
out_path.write_text(text, encoding="utf-8")
PY
}

parse_trial_result() {
  local out_log="$1"
  local err_log="$2"
  local exit_code="$3"
  python3 - "$out_log" "$err_log" "$exit_code" <<'PY'
import json
import pathlib
import re
import sys

out_text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
err_text = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")
exit_code = int(sys.argv[3])
combined = (out_text + "\n" + err_text).lower()

indexed = 0
failed = 0
for m in re.finditer(r'"indexed"\s*:\s*(\d+)', out_text):
    indexed = int(m.group(1))
for m in re.finditer(r'"failed"\s*:\s*(\d+)', out_text):
    failed = int(m.group(1))

if re.search(r'timeout:.*(no such file|failed to run command)', combined):
    outcome = "parse failures"
elif re.search(r'(maximum context length|context length|too many tokens|token.{0,40}(too large|exceed|limit)|input.{0,40}too long|requested \d+ tokens)', combined):
    outcome = "embedding input too large"
elif re.search(r'(connection refused|timed out|timeout|temporar(y|ily) unavailable|name or service not known|retryable status from embedding endpoint|embedding generation failed without explicit error|failed to establish a new connection|502|503|504)', combined):
    outcome = "embedding endpoint unavailable/timeout"
elif re.search(r'(database is locked|database is busy|sqlite lock retries exhausted|lock retries exhausted|lock_errors\"\s*:\s*[1-9])', combined):
    outcome = "lock errors"
elif re.search(r'(jsondecodeerror|parse failed|invalid \[|valueerror|toml)', combined):
    outcome = "parse failures"
elif exit_code == 0:
    outcome = f"success (indexed={indexed})"
else:
    outcome = "parse failures"

print(json.dumps({"outcome": outcome, "indexed": indexed, "failed": failed}, separators=(",", ":")))
PY
}

print_header() {
  echo "Quick indexing diagnostics"
  echo "- config: $CONFIG"
  echo "- sample_size: $SAMPLE_SIZE"
  echo "- index_batch_size: $INDEX_BATCH_SIZE"
  echo "- redaction_mode: $REDACTION_MODE"
  echo "- max_index_chars_list: $MAX_INDEX_CHARS_LIST"
  echo "- timeout_seconds: $TIMEOUT_SECONDS"
  if [[ -n "$HEARTBEAT_SECONDS" ]]; then
    echo "- heartbeat_seconds: $HEARTBEAT_SECONDS"
  fi
  echo "- logs: $RUN_ROOT"
  echo
}

print_header

RESULTS_TSV="$RUN_ROOT/summary.tsv"
: >"$RESULTS_TSV"

echo -e "max_index_chars\texit_code\tindexed\tfailed\toutcome\tout_log\terr_log" >>"$RESULTS_TSV"

trial_num=0
while IFS= read -r trial_chars; do
  [[ -z "$trial_chars" ]] && continue
  if ! [[ "$trial_chars" =~ ^[0-9]+$ ]] || [[ "$trial_chars" -lt 1 ]]; then
    echo "Skipping invalid max-index-chars value: $trial_chars" >&2
    continue
  fi

  trial_num=$((trial_num + 1))
  trial_tag="trial-${trial_num}-${trial_chars}"
  out_log="$RUN_ROOT/${trial_tag}.out.log"
  err_log="$RUN_ROOT/${trial_tag}.err.log"
  tmp_cfg="$TMP_DIR/${trial_tag}.toml"

  build_trial_config "$CONFIG" "$tmp_cfg"

  echo "[$trial_tag] max_index_chars=$trial_chars"
  echo "[$trial_tag] temp config: $tmp_cfg" >>"$err_log"

  limit="$INDEX_BATCH_SIZE"
  trial_exit=0
  while [[ "$limit" -le "$SAMPLE_SIZE" ]]; do
    echo "[$trial_tag] run index-vectors --limit $limit --max-index-chars $trial_chars" >>"$err_log"
    if run_with_timeout "$out_log" "$err_log" \
      inbox-vault --config "$tmp_cfg" index-vectors \
      --limit "$limit" \
      --redaction-mode "$REDACTION_MODE" \
      --max-index-chars "$trial_chars"; then
      :
    else
      trial_exit=$?
      break
    fi

    if [[ "$limit" -eq "$SAMPLE_SIZE" ]]; then
      break
    fi
    limit=$((limit + INDEX_BATCH_SIZE))
    if [[ "$limit" -gt "$SAMPLE_SIZE" ]]; then
      limit="$SAMPLE_SIZE"
    fi
  done

  parsed_json="$(parse_trial_result "$out_log" "$err_log" "$trial_exit")"
  outcome="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["outcome"])' "$parsed_json")"
  indexed="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["indexed"])' "$parsed_json")"
  failed="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["failed"])' "$parsed_json")"

  echo -e "${trial_chars}\t${trial_exit}\t${indexed}\t${failed}\t${outcome}\t${out_log}\t${err_log}" >>"$RESULTS_TSV"
done < <(split_csv "$MAX_INDEX_CHARS_LIST")

python3 - "$RESULTS_TSV" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines(), delimiter="\t"))

print("\nSummary")
print(f"{'max_chars':>9}  {'exit':>4}  {'indexed':>7}  {'failed':>6}  outcome")
print("-" * 78)
for r in rows:
    print(f"{r['max_index_chars']:>9}  {r['exit_code']:>4}  {r['indexed']:>7}  {r['failed']:>6}  {r['outcome']}")

recommended = None
success_clean = [r for r in rows if r['outcome'].startswith('success') and int(r['failed']) == 0 and int(r['exit_code']) == 0]
success_any = [r for r in rows if r['outcome'].startswith('success') and int(r['exit_code']) == 0]
if success_clean:
    recommended = min(success_clean, key=lambda r: int(r['max_index_chars']))
    reason = "first clean success (failed=0)"
elif success_any:
    recommended = min(success_any, key=lambda r: int(r['max_index_chars']))
    reason = "smallest successful trial"
else:
    reason = "no successful trial; inspect trial logs"

print("\nRecommended settings")
if recommended:
    print(f"- max_index_chars: {recommended['max_index_chars']} ({reason})")
else:
    print(f"- max_index_chars: keep low (e.g. 2000-3000); {reason}")
print("- embeddings.fallback: none (diagnostic mode only)")
print("- embeddings.max_retries: 0 (diagnostic mode only)")
print("- embeddings.timeout_seconds: 4 (diagnostic mode only)")
print(f"\nDetailed logs: {path.parent}")
PY
