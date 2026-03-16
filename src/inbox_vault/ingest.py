from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from googleapiclient.errors import HttpError

from .config import AppConfig
from .db import (
    get_cursor,
    get_oldest_internal_ts,
    message_exists,
    upsert_contact_seen,
    upsert_cursor,
    upsert_message,
    upsert_raw,
)
from .gmail_client import (
    fetch_full_message_payload,
    get_authenticated_email,
    get_profile_history_id,
    get_service,
    list_incremental_added_ids,
    list_message_ids_paged,
    payload_to_record,
)

MAILBOX_SCOPE = "INBOX,SENT"
LOG = logging.getLogger(__name__)

_PLACEHOLDER_EMAILS = {"you@gmail.com", "operator@example.com", ""}


def _resolve_account_email(acct, service) -> None:
    """Validate configured email against the authenticated Gmail account.

    Mutates acct.email in place if the configured value is a placeholder or
    does not match the authenticated account.
    """
    import sys

    api_email = get_authenticated_email(service).lower()
    configured = acct.email.lower()

    if configured in _PLACEHOLDER_EMAILS:
        print(
            f"[auto-detect] Config email '{acct.email}' is a placeholder. "
            f"Using authenticated email: {api_email}",
            file=sys.stderr,
        )
        acct.email = api_email
    elif configured != api_email:
        print(
            f"[warning] Config email '{configured}' does not match authenticated "
            f"Gmail account '{api_email}'. Using '{api_email}'.",
            file=sys.stderr,
        )
        acct.email = api_email

_ProgressCallback = Callable[[dict[str, Any]], None]


def _emit_progress(progress_callback: _ProgressCallback | None, event: dict[str, Any]) -> None:
    if progress_callback is None:
        return
    progress_callback(event)


def _update_contacts_from_record(conn, record: dict):
    if record.get("from_addr"):
        upsert_contact_seen(conn, record["from_addr"])
    if record.get("to_addr"):
        upsert_contact_seen(conn, record["to_addr"])


def _ingest_message_id(conn, service, account_email: str, msg_id: str) -> bool:
    try:
        raw = fetch_full_message_payload(service, msg_id)
        if not raw:
            return False

        rec = payload_to_record(raw, account_email)
        if not rec.get("msg_id"):
            return False

        upsert_message(conn, rec)
        upsert_raw(conn, rec["msg_id"], account_email, raw)
        _update_contacts_from_record(conn, rec)
        return True
    except Exception:
        # Privacy-safe logging: keep identifiers only, no message content.
        LOG.exception("Failed to ingest message_id=%s account=%s", msg_id, account_email)
        return False


def backfill(
    conn,
    cfg: AppConfig,
    max_messages: int | None = None,
    *,
    progress_callback: _ProgressCallback | None = None,
) -> dict:
    stats = {"accounts": 0, "ingested": 0, "skipped_existing": 0, "failed": 0}
    progress_every = max(1, int(cfg.gmail_progress_every))

    _emit_progress(
        progress_callback,
        {
            "event": "stage",
            "stage": "backfill_start",
            "accounts_total": len(cfg.accounts),
            "max_messages": max_messages,
            "progress_every": progress_every,
        },
    )

    if max_messages is None:
        for acct_idx, acct in enumerate(cfg.accounts, start=1):
            service = get_service(
                acct.credentials_file,
                acct.token_file,
                timeout_seconds=cfg.gmail_request_timeout_seconds,
            )
            _resolve_account_email(acct, service)
            stats["accounts"] += 1

            _emit_progress(
                progress_callback,
                {
                    "event": "stage",
                    "stage": "backfill_account_start",
                    "account": acct.email,
                    "account_index": acct_idx,
                    "accounts_total": len(cfg.accounts),
                    "max_messages": None,
                },
            )

            ids = list_message_ids_paged(service, query=cfg.gmail_query, max_messages=None)
            account_processed = 0
            for msg_id in ids:
                account_processed += 1
                if message_exists(conn, msg_id):
                    stats["skipped_existing"] += 1
                elif _ingest_message_id(conn, service, acct.email, msg_id):
                    stats["ingested"] += 1
                else:
                    stats["failed"] += 1

                if account_processed % progress_every == 0:
                    _emit_progress(
                        progress_callback,
                        {
                            "event": "progress",
                            "stage": "backfill_account",
                            "account": acct.email,
                            "account_index": acct_idx,
                            "accounts_total": len(cfg.accounts),
                            "account_processed": account_processed,
                            "ingested": stats["ingested"],
                            "skipped_existing": stats["skipped_existing"],
                            "failed": stats["failed"],
                        },
                    )

            upsert_cursor(conn, acct.email, MAILBOX_SCOPE, get_profile_history_id(service))
            conn.commit()

            _emit_progress(
                progress_callback,
                {
                    "event": "stage",
                    "stage": "backfill_account_done",
                    "account": acct.email,
                    "account_index": acct_idx,
                    "accounts_total": len(cfg.accounts),
                    "account_processed": account_processed,
                    "ingested": stats["ingested"],
                    "skipped_existing": stats["skipped_existing"],
                    "failed": stats["failed"],
                },
            )

        _emit_progress(
            progress_callback,
            {"event": "stage", "stage": "backfill_done", **stats},
        )
        return stats

    account_states: list[dict[str, Any]] = []
    for acct in cfg.accounts:
        service = get_service(
            acct.credentials_file,
            acct.token_file,
            timeout_seconds=cfg.gmail_request_timeout_seconds,
        )
        _resolve_account_email(acct, service)
        stats["accounts"] += 1
        account_states.append(
            {
                "acct": acct,
                "service": service,
                "ids": list_message_ids_paged(service, query=cfg.gmail_query, max_messages=max_messages),
                "index": 0,
            }
        )

    planned: list[tuple[object, str, str]] = []
    while len(planned) < max_messages:
        progressed = False
        for state in account_states:
            idx = state["index"]
            ids = state["ids"]
            if idx >= len(ids):
                continue
            planned.append((state["service"], state["acct"].email, ids[idx]))
            state["index"] += 1
            progressed = True
            if len(planned) >= max_messages:
                break
        if not progressed:
            break

    processed = 0
    for service, account_email, msg_id in planned:
        processed += 1
        if message_exists(conn, msg_id):
            stats["skipped_existing"] += 1
        elif _ingest_message_id(conn, service, account_email, msg_id):
            stats["ingested"] += 1
        else:
            stats["failed"] += 1

        if processed % progress_every == 0:
            _emit_progress(
                progress_callback,
                {
                    "event": "progress",
                    "stage": "backfill_global_cap",
                    "processed": processed,
                    "planned": len(planned),
                    "ingested": stats["ingested"],
                    "skipped_existing": stats["skipped_existing"],
                    "failed": stats["failed"],
                },
            )

    for state in account_states:
        upsert_cursor(conn, state["acct"].email, MAILBOX_SCOPE, get_profile_history_id(state["service"]))

    conn.commit()

    _emit_progress(
        progress_callback,
        {"event": "stage", "stage": "backfill_done", **stats, "processed": processed},
    )

    return stats


_TIME_FILTER_TOKEN_RE = re.compile(
    r"\b(?:newer_than|older_than|newer|older|after|before):\S+", re.IGNORECASE
)


def _sanitize_idle_backfill_query(query: str) -> str:
    sanitized = _TIME_FILTER_TOKEN_RE.sub(" ", query)
    sanitized = " ".join(sanitized.split()).strip()
    return sanitized or "label:inbox OR label:sent"


def _build_idle_backfill_query(
    conn,
    cfg: AppConfig,
    *,
    account_email: str,
) -> tuple[str, int | None]:
    base_query = cfg.gmail_idle_backfill_query or cfg.gmail_query
    sanitized_base = _sanitize_idle_backfill_query(base_query)

    oldest_internal_ts = get_oldest_internal_ts(conn, account_email)
    if oldest_internal_ts is None:
        return sanitized_base, None

    oldest_day = datetime.fromtimestamp(oldest_internal_ts / 1000, tz=timezone.utc).date()
    cutoff_day = oldest_day + timedelta(days=1)
    cutoff_token = cutoff_day.strftime("%Y/%m/%d")
    return f"({sanitized_base}) before:{cutoff_token}", oldest_internal_ts


def update(
    conn,
    cfg: AppConfig,
    *,
    progress_callback: _ProgressCallback | None = None,
) -> dict:
    progress_every = max(1, int(cfg.gmail_progress_every))
    stats = {
        "accounts": 0,
        "new_ids": 0,
        "ingested": 0,
        "cursor_resets": 0,
        "failed": 0,
    }

    _emit_progress(
        progress_callback,
        {
            "event": "stage",
            "stage": "update_start",
            "accounts_total": len(cfg.accounts),
            "progress_every": progress_every,
        },
    )

    for acct_idx, acct in enumerate(cfg.accounts, start=1):
        service = get_service(
            acct.credentials_file,
            acct.token_file,
            timeout_seconds=cfg.gmail_request_timeout_seconds,
        )
        _resolve_account_email(acct, service)
        stats["accounts"] += 1

        _emit_progress(
            progress_callback,
            {
                "event": "stage",
                "stage": "update_account_start",
                "account": acct.email,
                "account_index": acct_idx,
                "accounts_total": len(cfg.accounts),
            },
        )

        start_id = get_cursor(conn, acct.email, MAILBOX_SCOPE)
        if start_id is None:
            start_id = get_profile_history_id(service)
            upsert_cursor(conn, acct.email, MAILBOX_SCOPE, start_id)
            conn.commit()

        try:
            ids, latest = list_incremental_added_ids(service, start_id)
        except HttpError as e:
            if getattr(e.resp, "status", None) == 404:
                latest = get_profile_history_id(service)
                upsert_cursor(conn, acct.email, MAILBOX_SCOPE, latest)
                conn.commit()
                stats["cursor_resets"] += 1
                _emit_progress(
                    progress_callback,
                    {
                        "event": "stage",
                        "stage": "update_account_cursor_reset",
                        "account": acct.email,
                        "account_index": acct_idx,
                        "accounts_total": len(cfg.accounts),
                        "cursor_resets": stats["cursor_resets"],
                    },
                )
                continue
            raise

        stats["new_ids"] += len(ids)
        _emit_progress(
            progress_callback,
            {
                "event": "stage",
                "stage": "update_incremental_window",
                "account": acct.email,
                "account_index": acct_idx,
                "accounts_total": len(cfg.accounts),
                "new_ids": len(ids),
                "new_ids_total": stats["new_ids"],
            },
        )

        incremental_processed = 0
        for msg_id in ids:
            incremental_processed += 1
            if _ingest_message_id(conn, service, acct.email, msg_id):
                stats["ingested"] += 1
            else:
                stats["failed"] += 1

            if incremental_processed % progress_every == 0:
                _emit_progress(
                    progress_callback,
                    {
                        "event": "progress",
                        "stage": "update_incremental_ingest",
                        "account": acct.email,
                        "account_index": acct_idx,
                        "accounts_total": len(cfg.accounts),
                        "new_ids": len(ids),
                        "new_ids_total": stats["new_ids"],
                        "ingested": stats["ingested"],
                        "failed": stats["failed"],
                    },
                )

        upsert_cursor(conn, acct.email, MAILBOX_SCOPE, latest)
        conn.commit()

        _emit_progress(
            progress_callback,
            {
                "event": "stage",
                "stage": "update_account_done",
                "account": acct.email,
                "account_index": acct_idx,
                "accounts_total": len(cfg.accounts),
                "new_ids_total": stats["new_ids"],
                "ingested": stats["ingested"],
                "failed": stats["failed"],
            },
        )

    _emit_progress(
        progress_callback,
        {
            "event": "stage",
            "stage": "update_done",
            **stats,
        },
    )

    return stats


def repair(
    conn,
    cfg: AppConfig,
    *,
    backfill_limit: int | None = 0,
    commit_every_messages: int = 100,
    progress_callback: _ProgressCallback | None = None,
) -> dict:
    resolved_backfill_limit = max(0, int(backfill_limit or 0))
    safe_commit_every = max(1, int(commit_every_messages))
    progress_every = max(1, int(cfg.gmail_progress_every))

    stats: dict[str, Any] = {
        "accounts": 0,
        "backfill_limit": resolved_backfill_limit,
        "backfill_attempted_accounts": 0,
        "backfill_scanned": 0,
        "backfill_candidates": 0,
        "backfill_ingested": 0,
        "backfill_skipped_existing": 0,
        "backfill_failed": 0,
        "backfill_queries": {},
        "interrupted": False,
    }

    _emit_progress(
        progress_callback,
        {
            "event": "stage",
            "stage": "repair_start",
            "accounts_total": len(cfg.accounts),
            "backfill_limit": resolved_backfill_limit,
            "commit_every_messages": safe_commit_every,
            "progress_every": progress_every,
        },
    )

    account_states: list[dict[str, Any]] = []

    for acct_idx, acct in enumerate(cfg.accounts, start=1):
        service = get_service(
            acct.credentials_file,
            acct.token_file,
            timeout_seconds=cfg.gmail_request_timeout_seconds,
        )
        _resolve_account_email(acct, service)
        stats["accounts"] += 1

        current_cursor = get_cursor(conn, acct.email, MAILBOX_SCOPE)
        if current_cursor is None:
            upsert_cursor(conn, acct.email, MAILBOX_SCOPE, get_profile_history_id(service))
            conn.commit()

        query = None
        ids: list[str] = []
        if resolved_backfill_limit > 0:
            query, _oldest = _build_idle_backfill_query(conn, cfg, account_email=acct.email)
            scan_cap = max(resolved_backfill_limit, min(resolved_backfill_limit * 10, 2000))
            ids = list_message_ids_paged(service, query=query, max_messages=scan_cap)
            stats["backfill_attempted_accounts"] += 1
            stats["backfill_queries"][acct.email] = query

        state = {
            "acct": acct,
            "service": service,
            "account_index": acct_idx,
            "ids": ids,
            "index": 0,
            "account_scanned": 0,
            "account_candidates": 0,
            "account_ingested": 0,
            "account_skipped_existing": 0,
            "account_failed": 0,
        }
        account_states.append(state)

        _emit_progress(
            progress_callback,
            {
                "event": "stage",
                "stage": "repair_account_start",
                "account": acct.email,
                "account_index": acct_idx,
                "accounts_total": len(cfg.accounts),
                "backfill_limit": resolved_backfill_limit,
                "planned_scan": len(ids),
            },
        )

    processed_since_commit = 0

    if resolved_backfill_limit > 0:
        try:
            while stats["backfill_candidates"] < resolved_backfill_limit:
                progressed = False
                for state in account_states:
                    idx = state["index"]
                    ids = state["ids"]
                    if idx >= len(ids):
                        continue

                    progressed = True
                    msg_id = ids[idx]
                    state["index"] += 1

                    stats["backfill_scanned"] += 1
                    state["account_scanned"] += 1

                    if message_exists(conn, msg_id):
                        stats["backfill_skipped_existing"] += 1
                        state["account_skipped_existing"] += 1
                    else:
                        stats["backfill_candidates"] += 1
                        state["account_candidates"] += 1
                        if _ingest_message_id(conn, state["service"], state["acct"].email, msg_id):
                            stats["backfill_ingested"] += 1
                            state["account_ingested"] += 1
                        else:
                            stats["backfill_failed"] += 1
                            state["account_failed"] += 1

                    processed_since_commit += 1
                    if processed_since_commit >= safe_commit_every:
                        conn.commit()
                        processed_since_commit = 0

                    if stats["backfill_scanned"] % progress_every == 0:
                        _emit_progress(
                            progress_callback,
                            {
                                "event": "progress",
                                "stage": "repair_backfill",
                                "account": state["acct"].email,
                                "account_index": state["account_index"],
                                "accounts_total": len(cfg.accounts),
                                "backfill_limit": resolved_backfill_limit,
                                "backfill_scanned": stats["backfill_scanned"],
                                "backfill_candidates": stats["backfill_candidates"],
                                "backfill_ingested": stats["backfill_ingested"],
                                "backfill_skipped_existing": stats["backfill_skipped_existing"],
                                "backfill_failed": stats["backfill_failed"],
                            },
                        )

                    if stats["backfill_candidates"] >= resolved_backfill_limit:
                        break

                if not progressed:
                    break
        except KeyboardInterrupt:
            stats["interrupted"] = True
            conn.commit()

    for state in account_states:
        upsert_cursor(conn, state["acct"].email, MAILBOX_SCOPE, get_profile_history_id(state["service"]))
        conn.commit()
        _emit_progress(
            progress_callback,
            {
                "event": "stage",
                "stage": "repair_account_done",
                "account": state["acct"].email,
                "account_index": state["account_index"],
                "accounts_total": len(cfg.accounts),
                "account_scanned": state["account_scanned"],
                "account_candidates": state["account_candidates"],
                "account_ingested": state["account_ingested"],
                "account_skipped_existing": state["account_skipped_existing"],
                "account_failed": state["account_failed"],
            },
        )

    stats["ingested"] = stats["backfill_ingested"]
    stats["failed"] = stats["backfill_failed"]

    _emit_progress(
        progress_callback,
        {
            "event": "stage",
            "stage": "repair_done",
            **stats,
        },
    )

    return stats
