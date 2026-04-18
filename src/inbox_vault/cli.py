from __future__ import annotations

import argparse
import json
import socket
import sys
import urllib.parse
from datetime import datetime, timedelta, timezone

from .config import load_config, resolve_password
from .consolidation import run_consolidation
from .db import (
    clear_contact_profiles,
    get_conn,
    ingest_triage_summary,
    vector_level_counts,
)
from .enrich import enrich_pending
from .evals import bootstrap_eval_template, run_retrieval_eval
from .ingest import backfill, backfill_attachment_inventory, repair, update
from .profiles import build_profiles
from .redaction import (
    REDACTION_POLICY_VERSION,
    is_persistent_redaction_value_allowed,
    redact_text,
)
from .stress import run_isolated_stress
from .vectors import (
    INDEX_LEVEL_AUTO,
    INDEX_LEVEL_FULL,
    INDEX_LEVEL_REDACTED,
    count_pending_vector_updates,
    index_vectors,
    search_vectors,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inbox-vault",
        description="Encrypted Gmail sync + retrieval tooling with privacy-safe defaults.",
        epilog=(
            "Examples:\n"
            "  inbox-vault latest --from-date 2026-03-01 --to-date 2026-03-08 --clearance redacted\n"
            "  inbox-vault search \"invoice follow-up\" --from-date 2026-03-01 --to-date 2026-03-08\n"
            "  inbox-vault message 190b5f6b1f8a7c2d --clearance full"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default=None, help="Path to config TOML (default: ./config.toml)"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_backfill = sub.add_parser(
        "backfill", help="One-time full import from Gmail into encrypted DB"
    )
    p_backfill.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Global backfill cap (total messages across all configured accounts)",
    )
    p_backfill.add_argument(
        "--enrich", action="store_true", help="Run local-LLM enrichment after ingest"
    )
    p_backfill.add_argument(
        "--build-profiles", action="store_true", help="Build contact profiles after ingest"
    )
    p_backfill.add_argument(
        "--profiles-use-llm", action="store_true", help="Use local LLM for profile generation"
    )
    p_backfill.add_argument(
        "--index-vectors",
        action="store_true",
        help="Run vector indexing after ingest (or use config [indexing].auto_index_after_ingest).",
    )
    p_backfill.add_argument(
        "--no-index-vectors",
        action="store_true",
        help="Skip auto vector indexing for this run even if enabled in config.",
    )
    backfill_index_scope = p_backfill.add_mutually_exclusive_group()
    backfill_index_scope.add_argument(
        "--index-pending-only",
        action="store_true",
        help="When indexing via backfill, index only rows pending vector updates.",
    )
    backfill_index_scope.add_argument(
        "--index-all",
        action="store_true",
        help="When indexing via backfill, scan all candidate rows (not pending-only).",
    )
    p_backfill.add_argument(
        "--index-limit",
        type=int,
        default=None,
        help="Optional per-run cap when indexing via backfill (default from [indexing].auto_index_limit).",
    )

    p_update = sub.add_parser(
        "update",
        help="Sync new messages and run enrichment/indexing pipeline (main entry point)",
    )
    p_update.add_argument(
        "--backfill",
        type=int,
        default=None,
        metavar="N",
        help="Instead of incremental sync, backfill up to N historical messages.",
    )
    p_update.add_argument(
        "--enrich",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run pending enrichment after ingest (default: enabled; use --no-enrich to skip).",
    )
    p_update.add_argument("--build-profiles", action="store_true")
    p_update.add_argument("--profiles-use-llm", action="store_true")
    p_update.add_argument(
        "--index-vectors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run vector indexing after ingest (default: enabled; use --no-index-vectors to skip).",
    )
    update_index_scope = p_update.add_mutually_exclusive_group()
    update_index_scope.add_argument(
        "--index-pending-only",
        action="store_true",
        help="When indexing, process only rows pending vector updates.",
    )
    update_index_scope.add_argument(
        "--index-all",
        action="store_true",
        help="When indexing, scan all candidate rows (not pending-only).",
    )
    p_update.add_argument(
        "--index-limit",
        type=int,
        default=None,
        help="Optional per-run cap for indexing (default from [indexing].auto_index_limit).",
    )

    p_repair = sub.add_parser(
        "repair",
        help=(
            "Repair backlog: optional bounded historical ingest plus pending enrichment/vector indexing catch-up"
        ),
    )
    p_repair.add_argument(
        "--backfill-limit",
        type=int,
        default=0,
        help=(
            "Global cap for historical missing-message ingest during repair "
            "(0 keeps repair processing-only with no Gmail historical ingest)."
        ),
    )
    p_repair.add_argument(
        "--commit-every",
        type=int,
        default=100,
        help="Commit repair ingest progress every N scanned historical IDs.",
    )
    p_repair.add_argument(
        "--enrich",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run pending enrichment repair (default: enabled; use --no-enrich to skip).",
    )
    p_repair.add_argument(
        "--index-vectors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run pending vector/redaction catch-up (default: enabled; use --no-index-vectors to skip).",
    )
    repair_index_scope = p_repair.add_mutually_exclusive_group()
    repair_index_scope.add_argument(
        "--index-pending-only",
        action="store_true",
        help="When indexing via repair, index only rows pending vector updates.",
    )
    repair_index_scope.add_argument(
        "--index-all",
        action="store_true",
        help="When indexing via repair, scan all candidate rows (not pending-only).",
    )
    p_repair.add_argument(
        "--index-limit",
        type=int,
        default=None,
        help="Optional per-run cap when indexing via repair (default from [indexing].auto_index_limit).",
    )

    p_attachment_backfill = sub.add_parser(
        "backfill-attachments",
        help="Refresh attachment metadata for already-ingested messages without enrichment/indexing",
    )
    p_attachment_backfill.add_argument("--account-email", default=None)
    p_attachment_backfill.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional global cap for messages refreshed in this run.",
    )
    p_attachment_backfill.add_argument(
        "--all",
        action="store_true",
        help="Refresh all already-ingested messages in scope, not just those missing attachment inventory.",
    )
    p_attachment_backfill.add_argument(
        "--commit-every",
        type=int,
        default=100,
        help="Commit attachment inventory progress every N processed messages.",
    )

    p_enrich = sub.add_parser("enrich", help="Enrich unprocessed emails using local LLM")
    p_enrich.add_argument("--limit", type=int, default=200)

    p_profiles = sub.add_parser("build-profiles", help="Build or update contact profiles")
    p_profiles.add_argument("--use-llm", action="store_true")
    p_profiles.add_argument("--limit", type=int, default=200)
    p_profiles.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear existing contact_profiles before rebuilding in the same run",
    )

    p_index = sub.add_parser(
        "index-vectors", help="Build/update semantic vectors for indexed messages"
    )
    p_index.add_argument("--account-email", default=None, help="Optional account email filter")
    p_index.add_argument("--limit", type=int, default=None)
    p_index.add_argument(
        "--force", action="store_true", help="Recompute embeddings even if unchanged"
    )
    p_index.add_argument(
        "--pending-only",
        action="store_true",
        help="Index only rows currently pending vector updates (limit applies to pending rows, not raw scan rows).",
    )
    p_index.add_argument("--redaction-mode", choices=["regex", "model", "hybrid"], default=None)
    p_index.add_argument("--redaction-profile", default=None)
    p_index.add_argument("--redaction-instruction", default=None)
    p_index.add_argument(
        "--include-label",
        action="append",
        default=None,
        help="Repeatable. Only index messages matching at least one of these labels (comma-separated accepted).",
    )
    p_index.add_argument(
        "--exclude-label",
        action="append",
        default=None,
        help="Repeatable. Skip indexing messages matching any of these labels (comma-separated accepted).",
    )
    p_index.add_argument(
        "--max-index-chars",
        type=int,
        default=None,
        help="Trim indexed subject/snippet/body text per field to this character limit.",
    )
    p_index.add_argument(
        "--index-level",
        choices=[INDEX_LEVEL_REDACTED, INDEX_LEVEL_FULL],
        default=INDEX_LEVEL_REDACTED,
        help="Which dense index level to build. Redacted is the default safe index; full is operator opt-in.",
    )

    p_search = sub.add_parser("search", help="Semantic search over indexed messages")
    p_search.add_argument("query", help="Search query text")
    p_search.add_argument("--account-email", default=None, help="Optional account email filter")
    p_search.add_argument("--label", default=None, help="Optional label filter, e.g. INBOX")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--clearance", choices=["redacted", "full"], default="redacted")
    p_search.add_argument(
        "--search-level",
        choices=[INDEX_LEVEL_AUTO, INDEX_LEVEL_REDACTED, INDEX_LEVEL_FULL],
        default=INDEX_LEVEL_AUTO,
        help="Control which dense index level ranking uses. 'auto' prefers full only when available and full clearance is requested.",
    )
    p_search.add_argument("--strategy", choices=["dense", "lexical", "hybrid"], default=None)
    p_search.add_argument(
        "--from-date",
        default=None,
        help=(
            "Optional inclusive UTC lower bound. Accepts YYYY-MM-DD or ISO-8601 datetime "
            "(naive datetimes are treated as UTC)."
        ),
    )
    p_search.add_argument(
        "--to-date",
        default=None,
        help=(
            "Optional exclusive UTC upper bound. YYYY-MM-DD is interpreted as next-day 00:00 UTC "
            "(inclusive calendar-day behavior)."
        ),
    )

    p_eval = sub.add_parser(
        "eval-retrieval", help="Run local retrieval eval over mailbox-grounded qrels"
    )
    p_eval.add_argument("--eval-file", required=True, help="Path to JSON eval cases")
    p_eval.add_argument("--top-k", type=int, default=10)
    p_eval.add_argument("--strategy", choices=["dense", "lexical", "hybrid"], default="hybrid")
    p_eval.add_argument("--clearance", choices=["redacted", "full"], default="redacted")

    p_stress = sub.add_parser("stress-run", help="Run isolated end-to-end stress workflow")
    p_stress.add_argument(
        "--isolated-root",
        default=".stress-runs",
        help="Parent directory for isolated run artifacts",
    )
    p_stress.add_argument(
        "--max-messages",
        type=int,
        default=20,
        help="Global backfill cap for stress run (total across all configured accounts)",
    )
    p_stress.add_argument("--enrich-limit", type=int, default=500)
    p_stress.add_argument("--profiles-limit", type=int, default=200)
    p_stress.add_argument("--profiles-use-llm", action="store_true")
    p_stress.add_argument("--redaction-mode", choices=["regex", "model", "hybrid"], default=None)
    p_stress.add_argument("--redaction-profile", default=None)
    p_stress.add_argument("--redaction-instruction", default=None)
    p_stress.add_argument("--search-query", default="project")
    p_stress.add_argument("--search-top-k", type=int, default=5)
    p_stress.add_argument("--search-account-email", default=None)
    p_stress.add_argument("--search-label", default=None)
    p_stress.add_argument("--strategy", choices=["dense", "lexical", "hybrid"], default="hybrid")
    p_stress.add_argument("--eval-file", default=None)
    p_stress.add_argument("--report-file", default=None)
    p_stress.add_argument("--no-copy-tokens", action="store_true")

    p_consolidate = sub.add_parser(
        "consolidate-run",
        help="Run one complete base pipeline on a single isolated target DB",
    )
    p_consolidate.add_argument(
        "--target-root", required=True, help="Isolated target directory for DB/artifacts"
    )
    p_consolidate.add_argument(
        "--max-messages",
        type=int,
        default=20,
        help="Global backfill cap for consolidate run (total across all configured accounts)",
    )
    p_consolidate.add_argument("--enrich-limit", type=int, default=500)
    p_consolidate.add_argument("--profiles-limit", type=int, default=200)
    p_consolidate.add_argument("--profiles-use-llm", action="store_true")
    p_consolidate.add_argument(
        "--redaction-mode", choices=["regex", "model", "hybrid"], default=None
    )
    p_consolidate.add_argument("--redaction-profile", default=None)
    p_consolidate.add_argument("--redaction-instruction", default=None)
    p_consolidate.add_argument("--report-file", default=None)
    p_consolidate.add_argument("--no-copy-tokens", action="store_true")

    p_eval_bootstrap = sub.add_parser(
        "eval-bootstrap",
        help="Bootstrap local eval starter cases from current DB message IDs (safe metadata only)",
    )
    p_eval_bootstrap.add_argument("--output-file", required=True)
    p_eval_bootstrap.add_argument("--limit", type=int, default=20)

    p_status = sub.add_parser("status", help="Show mailbox/vector/profile status summary")
    p_status.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output (default behavior; provided for explicit machine workflows).",
    )

    p_latest = sub.add_parser("latest", help="Show latest messages with safe previews")
    p_latest.add_argument(
        "--limit", type=int, default=10, help="Number of newest messages to return"
    )
    p_latest.add_argument("--account-email", default=None, help="Optional account email filter")
    p_latest.add_argument(
        "--clearance",
        choices=["redacted", "full"],
        default="redacted",
        help="Return regex-redacted previews by default; use full for raw previews.",
    )
    p_latest.add_argument(
        "--from-date",
        default=None,
        help=(
            "Optional inclusive UTC lower bound. Accepts YYYY-MM-DD or ISO-8601 datetime "
            "(naive datetimes are treated as UTC)."
        ),
    )
    p_latest.add_argument(
        "--to-date",
        default=None,
        help=(
            "Optional exclusive UTC upper bound. YYYY-MM-DD is interpreted as next-day 00:00 UTC "
            "(inclusive calendar-day behavior)."
        ),
    )
    p_latest.add_argument(
        "--max-subject-chars",
        type=int,
        default=120,
        help="Max characters for subject in preview output",
    )
    p_latest.add_argument(
        "--max-snippet-chars",
        type=int,
        default=160,
        help="Max characters for snippet in preview output",
    )
    p_latest.add_argument(
        "--max-body-chars",
        type=int,
        default=240,
        help="Max characters for body preview output",
    )
    p_latest.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output (default behavior; provided for explicit machine workflows).",
    )

    p_message = sub.add_parser(
        "message",
        help="Fetch one message by msg_id with safe preview output",
    )
    p_message.add_argument("msg_id", help="Message ID to fetch")
    p_message.add_argument(
        "--clearance",
        choices=["redacted", "full"],
        default="full",
        help="Return raw previews (full) or regex-redacted previews.",
    )
    p_message.add_argument(
        "--max-subject-chars",
        type=int,
        default=120,
        help="Max characters for subject in preview output",
    )
    p_message.add_argument(
        "--max-snippet-chars",
        type=int,
        default=160,
        help="Max characters for snippet in preview output",
    )
    p_message.add_argument(
        "--max-body-chars",
        type=int,
        default=240,
        help="Max characters for body preview output",
    )
    p_message.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output (default behavior; provided for explicit machine workflows).",
    )

    p_profile_search = sub.add_parser("profile-search", help="Search contacts/profiles by keyword")
    p_profile_search.add_argument(
        "keyword", help="Search keyword (matches contact email + profile JSON text)"
    )
    p_profile_search.add_argument("--limit", type=int, default=20, help="Max profiles to return")
    p_profile_search.add_argument(
        "--max-profile-chars",
        type=int,
        default=200,
        help="Max characters from profile_json to include in safe preview",
    )
    p_profile_search.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output (default behavior; provided for explicit machine workflows).",
    )

    sub.add_parser("validate", help="Run lightweight contract checks")
    return parser


def _parse_label_overrides(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    out: list[str] = []
    for raw in values:
        parts = [part.strip() for part in str(raw).split(",")]
        out.extend(part for part in parts if part)
    return out


def _truncate_text(value: str | None, *, max_chars: int) -> str:
    text = "" if value is None else str(value)
    compact = " ".join(text.split())
    if max_chars < 1:
        max_chars = 1
    if len(compact) <= max_chars:
        return compact
    if max_chars <= 3:
        return compact[:max_chars]
    return compact[: max_chars - 3] + "..."


def _parse_date_value(raw: str, *, flag_name: str) -> tuple[datetime, bool]:
    text = str(raw or "").strip()
    if not text:
        raise ValueError(f"{flag_name} cannot be empty")

    is_day_only = len(text) == 10 and text[4] == "-" and text[7] == "-"
    if is_day_only:
        try:
            dt = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise ValueError(f"{flag_name} must be YYYY-MM-DD or ISO-8601 datetime") from exc
        return dt, True

    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{flag_name} must be YYYY-MM-DD or ISO-8601 datetime") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed, False


def _resolve_date_range(
    from_date: str | None,
    to_date: str | None,
) -> tuple[int | None, int | None]:
    from_ts_ms: int | None = None
    to_ts_ms: int | None = None

    if from_date:
        start_dt, _ = _parse_date_value(from_date, flag_name="--from-date")
        from_ts_ms = int(start_dt.timestamp() * 1000)

    if to_date:
        end_dt, day_only = _parse_date_value(to_date, flag_name="--to-date")
        if day_only:
            end_dt = end_dt + timedelta(days=1)
        to_ts_ms = int(end_dt.timestamp() * 1000)

    if from_ts_ms is not None and to_ts_ms is not None and from_ts_ms >= to_ts_ms:
        raise ValueError(
            "Invalid date range: --from-date must be earlier than --to-date "
            "(range is [from-date, to-date) in UTC)."
        )
    return from_ts_ms, to_ts_ms


def _message_status(conn, *, newest: bool) -> dict[str, object] | None:
    order = "DESC" if newest else "ASC"
    row = conn.execute(
        f"""
        SELECT msg_id, account_email, internal_ts, date_iso, history_id
        FROM messages
        ORDER BY COALESCE(internal_ts, 0) {order}, msg_id {order}
        LIMIT 1
        """
    ).fetchone()
    if not row:
        return None

    internal_ts = int(row[2]) if row[2] is not None else None
    observed_iso = row[3]
    if internal_ts is not None and internal_ts > 0:
        ts_seconds = internal_ts / 1000
        observed_iso = datetime.fromtimestamp(ts_seconds, tz=timezone.utc).isoformat()
    else:
        ts_seconds = None

    freshness_seconds = (
        max(0, int(datetime.now(tz=timezone.utc).timestamp() - ts_seconds))
        if ts_seconds is not None
        else None
    )
    freshness_hours = round(freshness_seconds / 3600, 2) if freshness_seconds is not None else None

    return {
        "msg_id": row[0],
        "account_email": row[1],
        "internal_ts": internal_ts,
        "date_iso": row[3],
        "history_id": int(row[4]) if row[4] is not None else None,
        "observed_at": observed_iso,
        "freshness_seconds": freshness_seconds,
        "freshness_hours": freshness_hours,
    }


def _latest_message_status(conn) -> dict[str, object] | None:
    return _message_status(conn, newest=True)


def _first_message_status(conn) -> dict[str, object] | None:
    return _message_status(conn, newest=False)


def _history_sync_status(conn) -> dict[str, object]:
    latest_rows = conn.execute(
        """
        SELECT msg_id, account_email, internal_ts, date_iso, history_id
        FROM messages
        ORDER BY account_email ASC, COALESCE(internal_ts, 0) DESC, msg_id DESC
        """
    ).fetchall()
    latest_by_account: dict[str, dict[str, object]] = {}
    for row in latest_rows:
        account_email = str(row[1])
        if account_email in latest_by_account:
            continue
        internal_ts = int(row[2]) if row[2] is not None else None
        observed_iso = row[3]
        if internal_ts is not None and internal_ts > 0:
            observed_iso = datetime.fromtimestamp(internal_ts / 1000, tz=timezone.utc).isoformat()
        latest_by_account[account_email] = {
            "msg_id": row[0],
            "internal_ts": internal_ts,
            "date_iso": row[3],
            "history_id": int(row[4]) if row[4] is not None else None,
            "observed_at": observed_iso,
        }

    cursor_rows = conn.execute(
        """
        SELECT account_email, scope, history_id, updated_at
        FROM sync_cursors
        ORDER BY account_email ASC, scope ASC
        """
    ).fetchall()
    cursor_by_account: dict[str, dict[str, object]] = {}
    for row in cursor_rows:
        cursor_by_account[str(row[0])] = {
            "scope": row[1],
            "history_id": int(row[2]) if row[2] is not None else None,
            "updated_at": row[3],
        }

    accounts = sorted(set(latest_by_account) | set(cursor_by_account))
    items: list[dict[str, object]] = []
    for account_email in accounts:
        latest = latest_by_account.get(account_email)
        cursor = cursor_by_account.get(account_email)
        latest_history_id = latest.get("history_id") if latest else None
        cursor_history_id = cursor.get("history_id") if cursor else None
        cursor_ahead_by = None
        if latest_history_id is not None and cursor_history_id is not None:
            cursor_ahead_by = max(0, int(cursor_history_id) - int(latest_history_id))
        backfill_repair_advisable = bool(cursor_ahead_by and cursor_ahead_by > 0)
        items.append(
            {
                "account_email": account_email,
                "cursor_scope": cursor.get("scope") if cursor else None,
                "cursor_history_id": cursor_history_id,
                "cursor_updated_at": cursor.get("updated_at") if cursor else None,
                "latest_ingested_message": latest,
                "cursor_ahead_by_history_ids": cursor_ahead_by,
                "backfill_repair_advisable": backfill_repair_advisable,
                "heuristic": (
                    "cursor is ahead of the latest locally ingested history_id; repair --backfill-limit N may help fill recoverable gaps"
                    if backfill_repair_advisable
                    else None
                ),
            }
        )

    return {
        "accounts": items,
        "any_backfill_repair_advisable": any(
            bool(item["backfill_repair_advisable"]) for item in items
        ),
    }


def _redaction_persistence_status(conn) -> dict[str, object]:
    rows = conn.execute(
        """
        SELECT policy_version, COALESCE(status, 'active') AS status, count(*)
        FROM redaction_entries
        GROUP BY policy_version, COALESCE(status, 'active')
        ORDER BY policy_version, status
        """
    ).fetchall()
    active_by_policy: dict[str, int] = {}
    rejected_by_policy: dict[str, int] = {}
    for policy_version, status, count in rows:
        key = str(policy_version or "")
        if status == "rejected":
            rejected_by_policy[key] = int(count)
        else:
            active_by_policy[key] = int(count)

    invalid_active_entries = 0
    invalid_rows = conn.execute(
        """
        SELECT key_name, original_value
        FROM redaction_entries
        WHERE COALESCE(status, 'active') = 'active'
        """
    ).fetchall()
    for key_name, original_value in invalid_rows:
        if not is_persistent_redaction_value_allowed(str(key_name), str(original_value)):
            invalid_active_entries += 1

    legacy_active_entries = sum(
        count for policy, count in active_by_policy.items() if policy != REDACTION_POLICY_VERSION
    )
    return {
        "active_by_policy_version": active_by_policy,
        "rejected_by_policy_version": rejected_by_policy,
        "legacy_active_entries": legacy_active_entries,
        "invalid_active_entries": invalid_active_entries,
    }


def run_status(conn, cfg) -> dict[str, object]:
    level_counts = vector_level_counts(conn)
    available_levels = sorted(level_counts)
    full_available = INDEX_LEVEL_FULL in available_levels
    pending_redacted = count_pending_vector_updates(conn, cfg, index_level=INDEX_LEVEL_REDACTED)
    pending_full = (
        count_pending_vector_updates(conn, cfg, index_level=INDEX_LEVEL_FULL)
        if full_available
        else None
    )
    active_redactions = int(
        conn.execute(
            "SELECT count(*) FROM redaction_entries WHERE COALESCE(status, 'active') = 'active'"
        ).fetchone()[0]
    )
    rejected_redactions = int(
        conn.execute(
            "SELECT count(*) FROM redaction_entries WHERE COALESCE(status, 'active') = 'rejected'"
        ).fetchone()[0]
    )
    attachment_total = int(
        conn.execute("SELECT count(*) FROM message_attachments").fetchone()[0]
    )
    attachment_messages = int(
        conn.execute("SELECT count(DISTINCT msg_id) FROM message_attachments").fetchone()[0]
    )
    inline_attachments = int(
        conn.execute("SELECT count(*) FROM message_attachments WHERE is_inline = 1").fetchone()[0]
    )
    attachment_state_counts = {
        str(state or ""): int(count)
        for state, count in conn.execute(
            """
            SELECT inventory_state, count(*)
            FROM message_attachments
            GROUP BY inventory_state
            ORDER BY inventory_state
            """
        ).fetchall()
    }
    policy_drift = {
        INDEX_LEVEL_REDACTED: int(
            conn.execute(
                """
                SELECT count(*) FROM vector_index_state
                WHERE index_level = ? AND COALESCE(redaction_policy_version, '') != ?
                """,
                (INDEX_LEVEL_REDACTED, REDACTION_POLICY_VERSION),
            ).fetchone()[0]
        )
    }
    if full_available:
        policy_drift[INDEX_LEVEL_FULL] = int(
            conn.execute(
                """
                SELECT count(*) FROM vector_index_state
                WHERE index_level = ? AND COALESCE(redaction_policy_version, '') != ?
                """,
                (INDEX_LEVEL_FULL, REDACTION_POLICY_VERSION),
            ).fetchone()[0]
        )

    counts = {
        "messages": int(conn.execute("SELECT count(*) FROM messages").fetchone()[0]),
        "attachments": attachment_total,
        "message_vectors": int(
            conn.execute("SELECT count(*) FROM message_vectors").fetchone()[0]
        ),
        "message_chunk_vectors": int(
            conn.execute("SELECT count(*) FROM message_chunk_vectors").fetchone()[0]
        ),
        "enrichments": int(conn.execute("SELECT count(*) FROM message_enrichment").fetchone()[0]),
        "profiles": int(conn.execute("SELECT count(*) FROM contact_profiles").fetchone()[0]),
        "active_redaction_entries": active_redactions,
        "rejected_redaction_entries": rejected_redactions,
    }
    pending_enrichments = int(
        conn.execute(
            """
            SELECT count(*)
            FROM messages m
            LEFT JOIN message_enrichment e ON e.msg_id = m.msg_id
            WHERE e.msg_id IS NULL
            """
        ).fetchone()[0]
    )
    heuristic_fallback_enrichments = int(
        conn.execute(
            """
            SELECT count(*)
            FROM message_enrichment
            WHERE COALESCE(model, '') = 'heuristic-fallback'
            """
        ).fetchone()[0]
    )
    inventory_messages = int(
        conn.execute("SELECT count(*) FROM message_attachment_inventory_state").fetchone()[0]
    )
    missing_inventory_messages = int(
        conn.execute(
            """
            SELECT count(*)
            FROM messages m
            LEFT JOIN message_attachment_inventory_state s ON s.msg_id = m.msg_id
            WHERE s.msg_id IS NULL
            """
        ).fetchone()[0]
    )
    llm_endpoint_reachable = _endpoint_reachable(cfg.llm.endpoint) if cfg.llm.enabled else None
    embeddings_endpoint_reachable = _endpoint_reachable(cfg.embeddings.endpoint)
    history_sync = _history_sync_status(conn)
    redaction_persistence = _redaction_persistence_status(conn)
    action_needed = any(int(value or 0) > 0 for value in policy_drift.values()) or bool(
        history_sync["any_backfill_repair_advisable"]
        or redaction_persistence["legacy_active_entries"]
        or redaction_persistence["invalid_active_entries"]
    )
    return {
        "counts": counts,
        "redaction_policy_version": REDACTION_POLICY_VERSION,
        "endpoint_health": {
            "llm": {
                "enabled": bool(cfg.llm.enabled),
                "endpoint": cfg.llm.endpoint,
                "reachable": llm_endpoint_reachable,
            },
            "embeddings": {
                "endpoint": cfg.embeddings.endpoint,
                "reachable": embeddings_endpoint_reachable,
            },
        },
        "enrichment_status": {
            "pending": pending_enrichments,
            "heuristic_fallback": heuristic_fallback_enrichments,
            "repairable": pending_enrichments + heuristic_fallback_enrichments,
            "degraded": heuristic_fallback_enrichments > 0,
        },
        "attachment_inventory": {
            "attachments": attachment_total,
            "messages_with_attachments": attachment_messages,
            "messages_with_inventory": inventory_messages,
            "messages_missing_inventory": missing_inventory_messages,
            "inline_attachments": inline_attachments,
            "non_inline_attachments": max(0, attachment_total - inline_attachments),
            "state_counts": attachment_state_counts,
        },
        "available_index_levels": available_levels,
        "full_search_available": full_available,
        "vector_level_counts": level_counts,
        "pending_vectors": {
            INDEX_LEVEL_REDACTED: pending_redacted,
            INDEX_LEVEL_FULL: pending_full,
        },
        "policy_drift_vectors": policy_drift,
        "redaction_persistence": redaction_persistence,
        "history_sync": history_sync,
        "ingest_triage": {
            "enabled": bool(cfg.ingest_triage.enabled),
            "mode": cfg.ingest_triage.mode,
            "summary": ingest_triage_summary(conn),
        },
        "action_needed": action_needed,
        "first_message": _first_message_status(conn),
        "latest_message": _latest_message_status(conn),
    }


def _format_message_preview(
    row,
    *,
    clearance: str,
    max_subject_chars: int,
    max_snippet_chars: int,
    max_body_chars: int,
) -> dict[str, object]:
    labels = []
    try:
        labels = json.loads(row[7] or "[]")
    except json.JSONDecodeError:
        labels = []

    subject_text = row[8]
    snippet_text = row[9]
    body_text = row[10]
    if clearance == "redacted":
        subject_text = redact_text(subject_text or "", mode="regex")
        snippet_text = redact_text(snippet_text or "", mode="regex")
        body_text = redact_text(body_text or "", mode="regex")

    return {
        "msg_id": row[0],
        "account_email": row[1],
        "thread_id": row[2],
        "internal_ts": int(row[3]) if row[3] is not None else None,
        "date_iso": row[4],
        "from_addr": row[5],
        "to_addr": row[6],
        "labels": labels,
        "subject": _truncate_text(subject_text, max_chars=max_subject_chars),
        "snippet": _truncate_text(snippet_text, max_chars=max_snippet_chars),
        "body_preview": _truncate_text(body_text, max_chars=max_body_chars),
    }


def run_latest(
    conn,
    *,
    limit: int,
    account_email: str | None,
    from_ts_ms: int | None,
    to_ts_ms: int | None,
    clearance: str,
    max_subject_chars: int,
    max_snippet_chars: int,
    max_body_chars: int,
) -> dict[str, object]:
    safe_limit = max(1, int(limit))
    sql = (
        "SELECT msg_id, account_email, thread_id, internal_ts, date_iso, from_addr, to_addr, labels_json, "
        "subject, snippet, body_text FROM messages"
    )
    params: list[str | int] = []
    clauses: list[str] = []
    if account_email:
        clauses.append("account_email = ?")
        params.append(account_email)
    if from_ts_ms is not None:
        clauses.append("COALESCE(internal_ts, 0) >= ?")
        params.append(int(from_ts_ms))
    if to_ts_ms is not None:
        clauses.append("COALESCE(internal_ts, 0) < ?")
        params.append(int(to_ts_ms))
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY COALESCE(internal_ts, 0) DESC LIMIT ?"
    params.append(safe_limit)

    rows = conn.execute(sql, params).fetchall()

    messages = [
        _format_message_preview(
            row,
            clearance=clearance,
            max_subject_chars=max_subject_chars,
            max_snippet_chars=max_snippet_chars,
            max_body_chars=max_body_chars,
        )
        for row in rows
    ]

    return {
        "count": len(messages),
        "limit": safe_limit,
        "messages": messages,
    }


def run_message(
    conn,
    *,
    msg_id: str,
    clearance: str,
    max_subject_chars: int,
    max_snippet_chars: int,
    max_body_chars: int,
) -> dict[str, object]:
    row = conn.execute(
        "SELECT msg_id, account_email, thread_id, internal_ts, date_iso, from_addr, to_addr, labels_json, "
        "subject, snippet, body_text FROM messages WHERE msg_id = ? LIMIT 1",
        (msg_id,),
    ).fetchone()
    if row is None:
        return {"found": False, "msg_id": msg_id, "count": 0, "message": None}

    return {
        "found": True,
        "msg_id": msg_id,
        "count": 1,
        "message": _format_message_preview(
            row,
            clearance=clearance,
            max_subject_chars=max_subject_chars,
            max_snippet_chars=max_snippet_chars,
            max_body_chars=max_body_chars,
        ),
    }


def _search_sender_map(conn, msg_ids: list[str]) -> dict[str, tuple[str | None, str | None]]:
    ordered_ids = [str(msg_id) for msg_id in msg_ids if str(msg_id).strip()]
    if not ordered_ids:
        return {}

    placeholders = ",".join("?" for _ in ordered_ids)
    rows = conn.execute(
        f"SELECT msg_id, from_addr, to_addr FROM messages WHERE msg_id IN ({placeholders})",
        ordered_ids,
    ).fetchall()
    return {str(row[0]): (row[1], row[2]) for row in rows}


def _apply_clearance_to_sender(value: str | None, *, clearance: str) -> str | None:
    if value is None or clearance == "full":
        return value
    return redact_text(value, mode="regex")


def run_profile_search(
    conn,
    *,
    keyword: str,
    limit: int,
    max_profile_chars: int,
) -> dict[str, object]:
    safe_limit = max(1, int(limit))
    term = (keyword or "").strip()
    if not term:
        return {"count": 0, "limit": safe_limit, "results": []}

    like_term = f"%{term.lower()}%"
    rows = conn.execute(
        """
        SELECT
          cs.contact_email,
          cs.display_name,
          cs.message_count,
          cs.first_seen,
          cs.last_seen,
          cp.profile_json,
          cp.model,
          cp.updated_at
        FROM contact_stats cs
        LEFT JOIN contact_profiles cp ON cp.contact_email = cs.contact_email
        WHERE lower(cs.contact_email) LIKE ?
           OR lower(COALESCE(cp.profile_json, '')) LIKE ?
        ORDER BY cs.message_count DESC, cs.contact_email ASC
        LIMIT ?
        """,
        (like_term, like_term, safe_limit),
    ).fetchall()

    results: list[dict[str, object]] = []
    for row in rows:
        profile_json = row[5]
        results.append(
            {
                "contact_email": row[0],
                "display_name": row[1],
                "message_count": int(row[2]) if row[2] is not None else 0,
                "first_seen": row[3],
                "last_seen": row[4],
                "profile_model": row[6],
                "profile_updated_at": row[7],
                "profile_preview": _truncate_text(profile_json, max_chars=max_profile_chars),
                "has_profile": bool(profile_json),
            }
        )

    return {
        "query": term,
        "count": len(results),
        "limit": safe_limit,
        "results": results,
    }


def run_validate(conn) -> dict:
    checks = {
        "messages_table": conn.execute(
            "SELECT name FROM sqlite_master WHERE name='messages'"
        ).fetchone()
        is not None,
        "raw_messages_table": conn.execute(
            "SELECT name FROM sqlite_master WHERE name='raw_messages'"
        ).fetchone()
        is not None,
        "sync_cursors_table": conn.execute(
            "SELECT name FROM sqlite_master WHERE name='sync_cursors'"
        ).fetchone()
        is not None,
        "message_vectors_table": conn.execute(
            "SELECT name FROM sqlite_master WHERE name='message_vectors'"
        ).fetchone()
        is not None,
        "message_chunk_vectors_table": conn.execute(
            "SELECT name FROM sqlite_master WHERE name='message_chunk_vectors'"
        ).fetchone()
        is not None,
        "message_fts_table": conn.execute(
            "SELECT name FROM sqlite_master WHERE name='message_fts'"
        ).fetchone()
        is not None,
        "message_fts_redacted_table": conn.execute(
            "SELECT name FROM sqlite_master WHERE name='message_fts_redacted'"
        ).fetchone()
        is not None,
        "vector_index_state_table": conn.execute(
            "SELECT name FROM sqlite_master WHERE name='vector_index_state'"
        ).fetchone()
        is not None,
        "redaction_entries_table": conn.execute(
            "SELECT name FROM sqlite_master WHERE name='redaction_entries'"
        ).fetchone()
        is not None,
    }
    checks["ok"] = all(checks.values())
    return checks


def _emit_ingest_progress(event: dict) -> None:
    event_name = str(event.get("event") or "")
    stage = str(event.get("stage") or "")

    if event_name == "progress":
        account = str(event.get("account") or "-")
        account_pos = int(event.get("account_index") or 0)
        account_total = int(event.get("accounts_total") or 0)

        if stage == "repair_backfill":
            line = (
                f"[ingest-progress] repair_backfill acct={account} ({account_pos}/{account_total}) "
                f"scanned={int(event.get('backfill_scanned') or 0)} "
                f"candidates={int(event.get('backfill_candidates') or 0)} "
                f"ingested={int(event.get('backfill_ingested') or 0)} "
                f"skipped={int(event.get('backfill_skipped_existing') or 0)} "
                f"failed={int(event.get('backfill_failed') or 0)}"
            )
        elif stage == "attachment_backfill_account":
            line = (
                f"[ingest-progress] attachment_backfill acct={account} ({account_pos}/{account_total}) "
                f"selected={int(event.get('selected_messages') or 0)} "
                f"processed={int(event.get('processed_messages') or 0)} "
                f"refreshed={int(event.get('refreshed_messages') or 0)} "
                f"failed={int(event.get('failed_messages') or 0)} "
                f"attachments={int(event.get('attachments_upserted') or 0)}"
            )
        else:
            new_ids = int(event.get("new_ids") or event.get("new_ids_total") or 0)
            ingested = int(event.get("ingested") or 0)
            failed = int(event.get("failed") or 0)
            line = (
                f"[ingest-progress] {stage} acct={account} ({account_pos}/{account_total}) "
                f"new_ids={new_ids} ingested={ingested} failed={failed}"
            )

        print(line, file=sys.stderr, flush=True)
        return

    if event_name == "stage":
        account = str(event.get("account") or "-")
        if stage in {"update_account_start", "update_account_done"}:
            line = (
                f"[ingest-stage] {stage} acct={account} "
                f"new_ids={int(event.get('new_ids') or event.get('new_ids_total') or 0)} "
                f"ingested={int(event.get('ingested') or 0)} "
                f"failed={int(event.get('failed') or 0)}"
            )
        elif stage in {"repair_account_start", "repair_account_done"}:
            line = (
                f"[ingest-stage] {stage} acct={account} "
                f"scanned={int(event.get('account_scanned') or 0)} "
                f"candidates={int(event.get('account_candidates') or 0)} "
                f"ingested={int(event.get('account_ingested') or 0)} "
                f"failed={int(event.get('account_failed') or 0)}"
            )
        elif stage in {"attachment_backfill_account_start", "attachment_backfill_account_done"}:
            line = (
                f"[ingest-stage] {stage} acct={account} "
                f"selected={int(event.get('selected_messages') or 0)} "
                f"processed={int(event.get('account_processed') or 0)} "
                f"refreshed={int(event.get('account_refreshed') or 0)} "
                f"failed={int(event.get('account_failed') or 0)}"
            )
        elif stage in {
            "update_start",
            "update_done",
            "backfill_start",
            "backfill_done",
            "repair_start",
            "repair_done",
            "attachment_backfill_start",
            "attachment_backfill_done",
        }:
            line = f"[ingest-stage] {stage}"
        else:
            line = f"[ingest-stage] {stage} acct={account}"

        print(line, file=sys.stderr, flush=True)



def _emit_index_progress(event: dict) -> None:
    kind = str(event.get("event") or "")
    pos = int(event.get("position") or 0)
    total = int(event.get("total") or 0)
    msg_id = str(event.get("msg_id") or "")
    acct = str(event.get("account_email") or "")
    chunk_count = int(event.get("chunk_count") or 0)
    indexed = int(event.get("indexed") or 0)
    chunks_indexed = int(event.get("chunks_indexed") or 0)
    failed = int(event.get("failed") or 0)
    elapsed_ms = int(event.get("elapsed_ms") or 0)
    substep = str(event.get("substep") or "")

    prefix = f"[index-progress] msg {pos}/{total} {msg_id} ({acct})"
    if kind == "message_start":
        line = f"{prefix} start | step={substep}"
    elif kind == "message_chunks_ready":
        line = f"{prefix} chunks_discovered={chunk_count} | next={substep}"
    elif kind == "message_substep":
        line = f"{prefix} step={substep}"
    elif kind == "message_unchanged":
        line = f"{prefix} unchanged → skipped | elapsed={elapsed_ms}ms"
    elif kind == "message_skipped_filtered":
        line = f"{prefix} filtered by label rules → skipped | elapsed={elapsed_ms}ms"
    elif kind == "message_failed":
        line = f"{prefix} FAILED at {substep} | elapsed={elapsed_ms}ms | indexed={indexed} chunks={chunks_indexed} failed={failed}"
    elif kind == "message_done":
        mpm = float(event.get("messages_per_min") or 0.0)
        line = (
            f"{prefix} done | chunks={chunk_count} elapsed={elapsed_ms}ms throughput={mpm:.2f}/min "
            f"| in-batch indexed={indexed} chunks={chunks_indexed} failed={failed}"
        )
    else:
        line = f"{prefix} event={kind}"

    print(line, file=sys.stderr, flush=True)


def _run_index_vectors_for_ingest(
    conn,
    cfg,
    *,
    limit: int | None,
    pending_only: bool,
) -> dict[str, int | str]:
    skip_applied_light = bool(cfg.ingest_triage.enabled and cfg.ingest_triage.mode == "enforce")
    pending_before = count_pending_vector_updates(
        conn,
        cfg,
        index_level=INDEX_LEVEL_REDACTED,
        skip_applied_light=skip_applied_light,
    )
    out = index_vectors(
        conn,
        cfg,
        index_level=INDEX_LEVEL_REDACTED,
        limit=limit,
        pending_only=pending_only,
        progress_callback=_emit_index_progress,
        skip_applied_light=skip_applied_light,
    )
    pending_after = count_pending_vector_updates(
        conn,
        cfg,
        index_level=INDEX_LEVEL_REDACTED,
        skip_applied_light=skip_applied_light,
    )
    out["pending_before"] = pending_before
    out["pending_after"] = pending_after
    out["skip_applied_light"] = skip_applied_light
    return out


def _endpoint_reachable(url: str, timeout: float = 2.0) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def _validate_accounts(cfg) -> None:
    """Pre-flight check: verify credential files exist for all configured accounts."""
    import os

    for acct in cfg.accounts:
        if not os.path.isfile(acct.credentials_file):
            raise FileNotFoundError(
                f"Credentials file not found for account '{acct.name}': {acct.credentials_file}\n\n"
                "To set up credentials:\n"
                "  1. Go to https://console.cloud.google.com/apis/credentials\n"
                "  2. Create an OAuth 2.0 Client ID (type: Desktop app)\n"
                "  3. Download the JSON and save it as: " + acct.credentials_file + "\n"
                "  4. Enable the Gmail API at "
                "https://console.cloud.google.com/apis/library/gmail.googleapis.com\n\n"
                "See the README 'Getting started' section for a full walkthrough."
            )


def _check_python_version() -> None:
    if sys.version_info < (3, 11):
        sys.exit(
            f"inbox-vault requires Python 3.11 or later (detected {sys.version_info.major}"
            f".{sys.version_info.minor}.{sys.version_info.micro}).\n"
            "Install a supported version and try again."
        )


def main(argv: list[str] | None = None) -> None:
    _check_python_version()
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    password = resolve_password(cfg.db)

    if args.command == "stress-run":
        print(
            json.dumps(
                run_isolated_stress(
                    cfg,
                    password,
                    isolated_root=args.isolated_root,
                    max_messages=args.max_messages,
                    enrich_limit=args.enrich_limit,
                    profiles_limit=args.profiles_limit,
                    profiles_use_llm=args.profiles_use_llm,
                    redaction_mode=args.redaction_mode,
                    redaction_profile=args.redaction_profile,
                    redaction_instruction=args.redaction_instruction,
                    search_query=args.search_query,
                    search_top_k=args.search_top_k,
                    search_account_email=args.search_account_email,
                    search_label=args.search_label,
                    strategy=args.strategy,
                    eval_file=args.eval_file,
                    copy_tokens=not args.no_copy_tokens,
                    report_path=args.report_file,
                ),
                indent=2,
            )
        )
        return

    if args.command == "consolidate-run":
        print(
            json.dumps(
                run_consolidation(
                    cfg,
                    password,
                    target_root=args.target_root,
                    max_messages=args.max_messages,
                    enrich_limit=args.enrich_limit,
                    profiles_limit=args.profiles_limit,
                    profiles_use_llm=args.profiles_use_llm,
                    redaction_mode=args.redaction_mode,
                    redaction_profile=args.redaction_profile,
                    redaction_instruction=args.redaction_instruction,
                    copy_tokens=not args.no_copy_tokens,
                    report_path=args.report_file,
                ),
                indent=2,
            )
        )
        return

    if args.command in {"backfill", "update", "repair", "backfill-attachments"}:
        _validate_accounts(cfg)

    conn = get_conn(cfg.db.path, password)
    try:
        if args.command == "backfill":
            if args.index_limit is not None and args.index_limit < 1:
                raise ValueError("--index-limit must be >= 1")
            should_index = bool(
                args.index_vectors
                or (cfg.indexing.auto_index_after_ingest and not args.no_index_vectors)
            )
            pending_only = (
                True
                if args.index_pending_only
                else False
                if args.index_all
                else cfg.indexing.auto_index_pending_only
            )
            index_limit = (
                args.index_limit if args.index_limit is not None else cfg.indexing.auto_index_limit
            )

            out = {
                "ingest": backfill(
                    conn,
                    cfg,
                    max_messages=args.max_messages,
                    progress_callback=_emit_ingest_progress,
                )
            }
            if args.enrich:
                enrich_diag: dict[str, int] = {}
                out["enrich"] = {
                    "updated": enrich_pending(conn, cfg, diagnostics=enrich_diag),
                    "diagnostics": enrich_diag,
                }
            if args.build_profiles:
                profile_diag: dict[str, int] = {}
                out["profiles"] = {
                    "updated": build_profiles(
                        conn,
                        cfg,
                        use_llm=args.profiles_use_llm,
                        diagnostics=profile_diag,
                    ),
                    "diagnostics": profile_diag,
                }
            if should_index:
                out["index_vectors"] = _run_index_vectors_for_ingest(
                    conn,
                    cfg,
                    limit=index_limit,
                    pending_only=pending_only,
                )
            print(json.dumps(out, indent=2))
            return

        if args.command == "update":
            if args.index_limit is not None and args.index_limit < 1:
                raise ValueError("--index-limit must be >= 1")
            if args.backfill is not None and args.backfill < 1:
                raise ValueError("--backfill must be >= 1")

            pending_only = (
                True
                if args.index_pending_only
                else False
                if args.index_all
                else True
            )
            index_limit = (
                args.index_limit if args.index_limit is not None else cfg.indexing.auto_index_limit
            )

            if args.backfill is not None:
                out = {
                    "ingest": backfill(
                        conn,
                        cfg,
                        max_messages=args.backfill,
                        progress_callback=_emit_ingest_progress,
                    )
                }
            else:
                out = {
                    "ingest": update(
                        conn,
                        cfg,
                        progress_callback=_emit_ingest_progress,
                    )
                }

            if args.enrich:
                if cfg.llm.enabled and _endpoint_reachable(cfg.llm.endpoint):
                    enrich_diag: dict[str, int] = {}
                    out["enrich"] = {
                        "updated": enrich_pending(conn, cfg, diagnostics=enrich_diag),
                        "diagnostics": enrich_diag,
                    }
                elif cfg.llm.enabled:
                    print(
                        f"[warning] LLM endpoint not reachable ({cfg.llm.endpoint}), "
                        "skipping enrichment. Use --no-enrich to silence.",
                        file=sys.stderr,
                    )
                    out["enrich"] = {"skipped": "endpoint_unreachable"}

            if args.build_profiles:
                profile_diag: dict[str, int] = {}
                out["profiles"] = {
                    "updated": build_profiles(
                        conn,
                        cfg,
                        use_llm=args.profiles_use_llm,
                        diagnostics=profile_diag,
                    ),
                    "diagnostics": profile_diag,
                }

            if args.index_vectors:
                if _endpoint_reachable(cfg.embeddings.endpoint):
                    out["index_vectors"] = _run_index_vectors_for_ingest(
                        conn,
                        cfg,
                        limit=index_limit,
                        pending_only=pending_only,
                    )
                else:
                    print(
                        f"[warning] Embedding endpoint not reachable ({cfg.embeddings.endpoint}), "
                        "skipping indexing. Use --no-index-vectors to silence.",
                        file=sys.stderr,
                    )
                    out["index_vectors"] = {"skipped": "endpoint_unreachable"}

            print(json.dumps(out, indent=2))
            return

        if args.command == "repair":
            if args.backfill_limit < 0:
                raise ValueError("--backfill-limit must be >= 0")
            if args.commit_every < 1:
                raise ValueError("--commit-every must be >= 1")
            if args.index_limit is not None and args.index_limit < 1:
                raise ValueError("--index-limit must be >= 1")

            pending_only = (
                True
                if args.index_pending_only
                else False
                if args.index_all
                else True
            )
            index_limit = (
                args.index_limit if args.index_limit is not None else cfg.indexing.auto_index_limit
            )

            out: dict[str, object] = {
                "ingest": repair(
                    conn,
                    cfg,
                    backfill_limit=args.backfill_limit,
                    commit_every_messages=args.commit_every,
                    progress_callback=_emit_ingest_progress,
                )
            }
            if bool(out["ingest"].get("interrupted")):
                out["warning"] = "Repair interrupted; partial progress committed."
                print(json.dumps(out, indent=2))
                return

            if args.enrich:
                enrich_diag: dict[str, int] = {}
                out["enrich"] = {
                    "updated": enrich_pending(
                        conn,
                        cfg,
                        diagnostics=enrich_diag,
                        include_degraded=True,
                    ),
                    "repair_scope": "pending+heuristic-fallback",
                    "diagnostics": enrich_diag,
                }

            if args.index_vectors:
                out["index_vectors"] = _run_index_vectors_for_ingest(
                    conn,
                    cfg,
                    limit=index_limit,
                    pending_only=pending_only,
                )

            print(json.dumps(out, indent=2))
            return

        if args.command == "backfill-attachments":
            if args.limit is not None and args.limit < 1:
                raise ValueError("--limit must be >= 1")
            if args.commit_every < 1:
                raise ValueError("--commit-every must be >= 1")

            print(
                json.dumps(
                    backfill_attachment_inventory(
                        conn,
                        cfg,
                        limit=args.limit,
                        missing_only=not args.all,
                        account_email=args.account_email,
                        commit_every_messages=args.commit_every,
                        progress_callback=_emit_ingest_progress,
                    ),
                    indent=2,
                )
            )
            return

        if args.command == "enrich":
            diagnostics: dict[str, int] = {}
            print(
                json.dumps(
                    {
                        "updated": enrich_pending(
                            conn, cfg, limit=args.limit, diagnostics=diagnostics
                        ),
                        "diagnostics": diagnostics,
                    },
                    indent=2,
                )
            )
            return

        if args.command == "build-profiles":
            diagnostics: dict[str, int] = {}
            if args.rebuild:
                diagnostics["cleared_before_rebuild"] = clear_contact_profiles(conn)
            print(
                json.dumps(
                    {
                        "updated": build_profiles(
                            conn,
                            cfg,
                            use_llm=args.use_llm,
                            limit=args.limit,
                            diagnostics=diagnostics,
                        ),
                        "diagnostics": diagnostics,
                    },
                    indent=2,
                )
            )
            return

        if args.command == "index-vectors":
            include_labels = _parse_label_overrides(args.include_label)
            exclude_labels = _parse_label_overrides(args.exclude_label)
            pending_before = count_pending_vector_updates(
                conn,
                cfg,
                index_level=args.index_level,
                account_email=args.account_email,
                include_labels=include_labels,
                exclude_labels=exclude_labels,
                max_index_chars=args.max_index_chars,
            )
            out = index_vectors(
                conn,
                cfg,
                index_level=args.index_level,
                account_email=args.account_email,
                limit=args.limit,
                force=args.force,
                pending_only=args.pending_only,
                redaction_mode=args.redaction_mode,
                redaction_profile=args.redaction_profile,
                redaction_instruction=args.redaction_instruction,
                include_labels=include_labels,
                exclude_labels=exclude_labels,
                max_index_chars=args.max_index_chars,
                progress_callback=_emit_index_progress,
            )
            pending_after = count_pending_vector_updates(
                conn,
                cfg,
                index_level=args.index_level,
                account_email=args.account_email,
                include_labels=include_labels,
                exclude_labels=exclude_labels,
                max_index_chars=args.max_index_chars,
            )
            out["pending_before"] = pending_before
            out["pending_after"] = pending_after
            print(json.dumps(out, indent=2))
            return

        if args.command == "search":
            from_ts_ms, to_ts_ms = _resolve_date_range(args.from_date, args.to_date)
            rows, diagnostics = search_vectors(
                conn,
                cfg,
                args.query,
                account_email=args.account_email,
                label=args.label,
                top_k=args.top_k,
                clearance=args.clearance,
                search_level=args.search_level,
                strategy=args.strategy,
                from_ts_ms=from_ts_ms,
                to_ts_ms=to_ts_ms,
                include_diagnostics=True,
            )
            sender_map = _search_sender_map(conn, [item.msg_id for item in rows])
            print(
                json.dumps(
                    {
                        "query": args.query,
                        "count": len(rows),
                        "clearance": args.clearance,
                        "diagnostics": {
                            "search_level_requested": diagnostics.requested_level,
                            "search_level_used": diagnostics.used_level,
                            "search_level_fallback": diagnostics.fallback_from_level,
                            "full_level_available": diagnostics.full_level_available,
                        },
                        "results": [
                            {
                                "score": round(item.score, 6),
                                "msg_id": item.msg_id,
                                "thread_id": item.thread_id,
                                "account_email": item.account_email,
                                "from_addr": _apply_clearance_to_sender(
                                    sender_map.get(item.msg_id, (None, None))[0],
                                    clearance=args.clearance,
                                ),
                                "to_addr": _apply_clearance_to_sender(
                                    sender_map.get(item.msg_id, (None, None))[1],
                                    clearance=args.clearance,
                                ),
                                "labels": item.labels,
                                "content": item.content,
                            }
                            for item in rows
                        ],
                    },
                    indent=2,
                )
            )
            return

        if args.command == "eval-retrieval":
            print(
                json.dumps(
                    run_retrieval_eval(
                        conn,
                        cfg,
                        eval_file=args.eval_file,
                        strategy=args.strategy,
                        top_k=args.top_k,
                        clearance=args.clearance,
                    ),
                    indent=2,
                )
            )
            return

        if args.command == "eval-bootstrap":
            print(
                json.dumps(
                    bootstrap_eval_template(
                        conn,
                        output_file=args.output_file,
                        limit=args.limit,
                    ),
                    indent=2,
                )
            )
            return

        if args.command == "status":
            print(json.dumps(run_status(conn, cfg), indent=2))
            return

        if args.command == "latest":
            from_ts_ms, to_ts_ms = _resolve_date_range(args.from_date, args.to_date)
            print(
                json.dumps(
                    run_latest(
                        conn,
                        limit=args.limit,
                        account_email=args.account_email,
                        from_ts_ms=from_ts_ms,
                        to_ts_ms=to_ts_ms,
                        clearance=args.clearance,
                        max_subject_chars=args.max_subject_chars,
                        max_snippet_chars=args.max_snippet_chars,
                        max_body_chars=args.max_body_chars,
                    ),
                    indent=2,
                )
            )
            return

        if args.command == "message":
            print(
                json.dumps(
                    run_message(
                        conn,
                        msg_id=args.msg_id,
                        clearance=args.clearance,
                        max_subject_chars=args.max_subject_chars,
                        max_snippet_chars=args.max_snippet_chars,
                        max_body_chars=args.max_body_chars,
                    ),
                    indent=2,
                )
            )
            return

        if args.command == "profile-search":
            print(
                json.dumps(
                    run_profile_search(
                        conn,
                        keyword=args.keyword,
                        limit=args.limit,
                        max_profile_chars=args.max_profile_chars,
                    ),
                    indent=2,
                )
            )
            return

        if args.command == "validate":
            print(json.dumps(run_validate(conn), indent=2))
            return
    finally:
        conn.close()


if __name__ == "__main__":
    main()
