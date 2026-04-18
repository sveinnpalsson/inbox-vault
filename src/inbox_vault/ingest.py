from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from googleapiclient.errors import HttpError

from .config import AppConfig
from .db import (
    attachment_key,
    get_cursor,
    get_stream_observation_count,
    get_oldest_internal_ts,
    message_exists,
    update_attachment_materialization,
    upsert_contact_seen,
    upsert_cursor,
    upsert_message_attachments,
    upsert_message_ingest_triage,
    upsert_message,
    upsert_raw,
)
from .gmail_client import (
    extract_inline_attachment_bytes,
    fetch_attachment_bytes,
    fetch_full_message_payload,
    get_authenticated_email,
    get_profile_history_id,
    get_service,
    list_incremental_added_ids,
    list_message_ids_paged,
    payload_to_record,
)
from .ingest_triage import derive_ingest_triage

MAILBOX_SCOPE = "INBOX,SENT"
LOG = logging.getLogger(__name__)

_PLACEHOLDER_EMAILS = {"you@gmail.com", "operator@example.com", ""}
_SAFE_ATTACHMENT_EXT_RE = re.compile(r"^\.[a-z0-9]{1,10}$")


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


def _persist_observe_only_triage(
    conn,
    cfg: AppConfig,
    *,
    rec: dict[str, Any],
    raw_payload: dict[str, Any],
) -> None:
    if not cfg.ingest_triage.enabled:
        return
    provisional = derive_ingest_triage(rec, raw_payload=raw_payload)
    proposal = derive_ingest_triage(
        rec,
        raw_payload=raw_payload,
        prior_observation_count=get_stream_observation_count(conn, provisional.stream_id),
    )
    applied_tier = "full"
    if cfg.ingest_triage.mode == "enforce" and proposal.triage_tier != "full":
        applied_tier = "light"
    upsert_message_ingest_triage(
        conn,
        {
            "msg_id": proposal.msg_id,
            "account_email": proposal.account_email,
            "stream_id": proposal.stream_id,
            "stream_kind": proposal.stream_kind,
            "subject_family": proposal.subject_family,
            "sender_domain": proposal.sender_domain,
            "triage_tier": proposal.triage_tier,
            "proposed_tier": proposal.triage_tier,
            "applied_tier": applied_tier,
            "enforcement_mode": cfg.ingest_triage.mode,
            "decision_source": proposal.decision_source,
            "bulk_score": proposal.bulk_score,
            "importance_score": proposal.importance_score,
            "novelty_score": proposal.novelty_score,
            "signals_json": proposal.signals_json,
        },
    )


def _ingest_message_id(conn, cfg: AppConfig, service, account_email: str, msg_id: str) -> bool:
    ingested, _materialization = _ingest_message_id_with_attachment_materialization(
        conn,
        cfg,
        service,
        account_email,
        msg_id,
        materialize_attachments=False,
    )
    return ingested


def _ingest_message_id_with_attachment_materialization(
    conn,
    cfg: AppConfig,
    service,
    account_email: str,
    msg_id: str,
    *,
    materialize_attachments: bool,
) -> tuple[bool, dict[str, int]]:
    materialization = {
        "selected_attachments": 0,
        "materialized_attachments": 0,
        "failed_attachments": 0,
        "bytes_written": 0,
        "inline_sourced": 0,
        "gmail_fetched": 0,
    }
    try:
        raw = fetch_full_message_payload(service, msg_id)
        if not raw:
            return False, materialization

        rec = payload_to_record(raw, account_email)
        if not rec.get("msg_id"):
            return False, materialization

        upsert_message(conn, rec)
        upsert_raw(conn, rec["msg_id"], account_email, raw)
        attachments = list(rec.get("attachments", []))
        materialization["selected_attachments"] = len(attachments)
        upsert_message_attachments(
            conn,
            rec["msg_id"],
            account_email,
            attachments,
        )
        _update_contacts_from_record(conn, rec)
        _persist_observe_only_triage(conn, cfg, rec=rec, raw_payload=raw)
        if materialize_attachments and attachments:
            materialization = _materialize_ingested_message_attachments(
                conn,
                account_email=account_email,
                service=service,
                msg_id=rec["msg_id"],
                raw_payload=raw,
                attachments=attachments,
            )
        return True, materialization
    except Exception:
        # Privacy-safe logging: keep identifiers only, no message content.
        LOG.exception("Failed to ingest message_id=%s account=%s", msg_id, account_email)
        return False, materialization


def _materialize_ingested_message_attachments(
    conn,
    *,
    account_email: str,
    service,
    msg_id: str,
    raw_payload: dict[str, Any],
    attachments: list[dict[str, Any]],
) -> dict[str, int]:
    db_dir = Path(conn.execute("PRAGMA database_list").fetchone()[2]).resolve().parent
    stats = {
        "selected_attachments": len(attachments),
        "materialized_attachments": 0,
        "failed_attachments": 0,
        "bytes_written": 0,
        "inline_sourced": 0,
        "gmail_fetched": 0,
    }

    for attachment in attachments:
        part_id = str(attachment.get("part_id") or "").strip()
        if not part_id:
            continue
        attachment_key_value = attachment_key(account_email, msg_id, part_id)
        try:
            payload_bytes = extract_inline_attachment_bytes(raw_payload, part_id)
            source = "inline"
            if payload_bytes is None:
                gmail_attachment_id = str(attachment.get("gmail_attachment_id") or "").strip()
                if not gmail_attachment_id:
                    raise RuntimeError(
                        "attachment bytes unavailable: missing inline data and gmail_attachment_id"
                    )
                payload_bytes = fetch_attachment_bytes(service, msg_id, gmail_attachment_id)
                source = "gmail_attachment_get"

            relpath = _attachment_materialization_relpath(
                account_email=account_email,
                attachment_key_value=attachment_key_value,
                mime_type=str(attachment.get("mime_type") or ""),
            )
            abs_path = db_dir / relpath
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = abs_path.with_name(abs_path.name + ".tmp")
            tmp_path.write_bytes(payload_bytes)
            tmp_path.replace(abs_path)

            checksum = hashlib.sha256(payload_bytes).hexdigest()
            materialized_at = datetime.now(timezone.utc).isoformat()
            update_attachment_materialization(
                conn,
                msg_id=msg_id,
                part_id=part_id,
                attachment_key_value=attachment_key_value,
                storage_kind="file",
                storage_path=relpath,
                content_sha256=checksum,
                content_size_bytes=len(payload_bytes),
                materialized_at=materialized_at,
            )
            stats["materialized_attachments"] += 1
            stats["bytes_written"] += len(payload_bytes)
            if source == "inline":
                stats["inline_sourced"] += 1
            else:
                stats["gmail_fetched"] += 1
        except Exception:
            LOG.exception(
                "Failed to materialize attachment bytes for message_id=%s part_id=%s account=%s",
                msg_id,
                part_id,
                account_email,
            )
            stats["failed_attachments"] += 1

    return stats


def _attachment_backfill_candidates(
    conn,
    *,
    account_email: str,
    missing_only: bool,
    limit: int | None,
) -> list[str]:
    sql = """
        SELECT m.msg_id
        FROM messages m
        LEFT JOIN message_attachment_inventory_state s ON s.msg_id = m.msg_id
        WHERE m.account_email = ?
    """
    params: list[Any] = [account_email]
    if missing_only:
        sql += " AND s.msg_id IS NULL"
    sql += " ORDER BY COALESCE(m.internal_ts, 0) DESC, m.msg_id DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    rows = conn.execute(sql, params).fetchall()
    return [str(row[0]) for row in rows if row and row[0] is not None]


def backfill_attachment_inventory(
    conn,
    cfg: AppConfig,
    *,
    limit: int | None = None,
    missing_only: bool = True,
    account_email: str | None = None,
    commit_every_messages: int = 100,
    progress_callback: _ProgressCallback | None = None,
) -> dict[str, Any]:
    safe_limit = None if limit is None else max(1, int(limit))
    safe_commit_every = max(1, int(commit_every_messages))
    progress_every = max(1, int(cfg.gmail_progress_every))

    stats: dict[str, Any] = {
        "accounts": 0,
        "limit": safe_limit,
        "missing_only": bool(missing_only),
        "selected_messages": 0,
        "processed_messages": 0,
        "refreshed_messages": 0,
        "failed_messages": 0,
        "attachments_upserted": 0,
        "messages_with_attachments": 0,
        "messages_without_attachments": 0,
    }

    _emit_progress(
        progress_callback,
        {
            "event": "stage",
            "stage": "attachment_backfill_start",
            "accounts_total": len(cfg.accounts),
            "limit": safe_limit,
            "missing_only": bool(missing_only),
            "account_filter": account_email,
            "commit_every_messages": safe_commit_every,
            "progress_every": progress_every,
        },
    )

    processed_since_commit = 0
    remaining = safe_limit

    for acct_idx, acct in enumerate(cfg.accounts, start=1):
        service = get_service(
            acct.credentials_file,
            acct.token_file,
            timeout_seconds=cfg.gmail_request_timeout_seconds,
        )
        _resolve_account_email(acct, service)

        if account_email and acct.email != account_email:
            continue

        stats["accounts"] += 1
        candidate_ids = _attachment_backfill_candidates(
            conn,
            account_email=acct.email,
            missing_only=missing_only,
            limit=remaining,
        )
        stats["selected_messages"] += len(candidate_ids)

        _emit_progress(
            progress_callback,
            {
                "event": "stage",
                "stage": "attachment_backfill_account_start",
                "account": acct.email,
                "account_index": acct_idx,
                "accounts_total": len(cfg.accounts),
                "selected_messages": len(candidate_ids),
            },
        )

        account_processed = 0
        account_refreshed = 0
        account_failed = 0

        for msg_id in candidate_ids:
            account_processed += 1
            stats["processed_messages"] += 1
            try:
                raw = fetch_full_message_payload(service, msg_id)
                if not raw:
                    stats["failed_messages"] += 1
                    account_failed += 1
                    continue

                rec = payload_to_record(raw, acct.email)
                attachments = list(rec.get("attachments", []))
                upsert_message_attachments(conn, msg_id, acct.email, attachments)
                stats["refreshed_messages"] += 1
                stats["attachments_upserted"] += len(attachments)
                if attachments:
                    stats["messages_with_attachments"] += 1
                else:
                    stats["messages_without_attachments"] += 1
                account_refreshed += 1
            except Exception:
                LOG.exception(
                    "Failed to backfill attachment inventory for message_id=%s account=%s",
                    msg_id,
                    acct.email,
                )
                stats["failed_messages"] += 1
                account_failed += 1

            processed_since_commit += 1
            if processed_since_commit >= safe_commit_every:
                conn.commit()
                processed_since_commit = 0

            if account_processed % progress_every == 0:
                _emit_progress(
                    progress_callback,
                    {
                        "event": "progress",
                        "stage": "attachment_backfill_account",
                        "account": acct.email,
                        "account_index": acct_idx,
                        "accounts_total": len(cfg.accounts),
                        "selected_messages": len(candidate_ids),
                        "processed_messages": stats["processed_messages"],
                        "refreshed_messages": stats["refreshed_messages"],
                        "failed_messages": stats["failed_messages"],
                        "attachments_upserted": stats["attachments_upserted"],
                    },
                )

        conn.commit()
        processed_since_commit = 0

        _emit_progress(
            progress_callback,
            {
                "event": "stage",
                "stage": "attachment_backfill_account_done",
                "account": acct.email,
                "account_index": acct_idx,
                "accounts_total": len(cfg.accounts),
                "selected_messages": len(candidate_ids),
                "account_processed": account_processed,
                "account_refreshed": account_refreshed,
                "account_failed": account_failed,
            },
        )

        if remaining is not None:
            remaining = max(0, remaining - len(candidate_ids))
            if remaining == 0:
                break

    _emit_progress(
        progress_callback,
        {
            "event": "stage",
            "stage": "attachment_backfill_done",
            **stats,
        },
    )

    return stats


def _safe_attachment_extension(mime_type: str) -> str:
    normalized = str(mime_type or "").split(";", 1)[0].strip().lower()
    ext = mimetypes.guess_extension(normalized) or ".bin"
    if ext == ".jpe":
        ext = ".jpg"
    if not _SAFE_ATTACHMENT_EXT_RE.match(ext):
        return ".bin"
    return ext


def _attachment_materialization_relpath(
    *,
    account_email: str,
    attachment_key_value: str,
    mime_type: str,
) -> str:
    account_key = hashlib.sha256(str(account_email or "").encode("utf-8")).hexdigest()[:16]
    filename = f"{attachment_key_value}{_safe_attachment_extension(mime_type)}"
    return str(Path("attachments") / account_key / attachment_key_value[:2] / filename)


def _attachment_materialization_candidates(conn, *, account_email: str) -> list[dict[str, str]]:
    rows = conn.execute(
        """
        SELECT msg_id, account_email, part_id, attachment_key, gmail_attachment_id, mime_type,
               storage_kind, storage_path
        FROM message_attachments
        WHERE account_email = ?
        ORDER BY COALESCE(last_seen_at, '') DESC, msg_id DESC, part_id DESC
        """,
        (account_email,),
    ).fetchall()
    return [
        {
            "msg_id": str(row[0] or ""),
            "account_email": str(row[1] or ""),
            "part_id": str(row[2] or ""),
            "attachment_key": str(row[3] or ""),
            "gmail_attachment_id": str(row[4] or ""),
            "mime_type": str(row[5] or ""),
            "storage_kind": str(row[6] or ""),
            "storage_path": str(row[7] or ""),
        }
        for row in rows
        if row and row[0] is not None and row[2] is not None
    ]


def _merge_attachment_materialization_stats(
    stats: dict[str, Any],
    materialization: dict[str, int],
) -> None:
    stats["selected_attachments"] = int(stats.get("selected_attachments") or 0) + int(
        materialization.get("selected_attachments") or 0
    )
    stats["materialized_attachments"] = int(stats.get("materialized_attachments") or 0) + int(
        materialization.get("materialized_attachments") or 0
    )
    stats["attachment_materialization_failed"] = int(
        stats.get("attachment_materialization_failed") or 0
    ) + int(materialization.get("failed_attachments") or 0)
    stats["attachment_bytes_written"] = int(stats.get("attachment_bytes_written") or 0) + int(
        materialization.get("bytes_written") or 0
    )
    stats["inline_attachment_sources"] = int(
        stats.get("inline_attachment_sources") or 0
    ) + int(materialization.get("inline_sourced") or 0)
    stats["gmail_attachment_fetches"] = int(
        stats.get("gmail_attachment_fetches") or 0
    ) + int(materialization.get("gmail_fetched") or 0)


def materialize_attachment_bytes(
    conn,
    cfg: AppConfig,
    *,
    limit: int | None = None,
    missing_only: bool = True,
    account_email: str | None = None,
    commit_every_attachments: int = 100,
    progress_callback: _ProgressCallback | None = None,
) -> dict[str, Any]:
    safe_limit = None if limit is None else max(1, int(limit))
    safe_commit_every = max(1, int(commit_every_attachments))
    progress_every = max(1, int(cfg.gmail_progress_every))
    db_dir = Path(cfg.db.path).resolve().parent

    stats: dict[str, Any] = {
        "accounts": 0,
        "limit": safe_limit,
        "missing_only": bool(missing_only),
        "selected_attachments": 0,
        "processed_attachments": 0,
        "materialized_attachments": 0,
        "failed_attachments": 0,
        "bytes_written": 0,
        "inline_sourced": 0,
        "gmail_fetched": 0,
    }

    _emit_progress(
        progress_callback,
        {
            "event": "stage",
            "stage": "attachment_materialize_start",
            "accounts_total": len(cfg.accounts),
            "limit": safe_limit,
            "missing_only": bool(missing_only),
            "account_filter": account_email,
            "commit_every_attachments": safe_commit_every,
            "progress_every": progress_every,
        },
    )

    processed_since_commit = 0
    remaining = safe_limit

    for acct_idx, acct in enumerate(cfg.accounts, start=1):
        service = get_service(
            acct.credentials_file,
            acct.token_file,
            timeout_seconds=cfg.gmail_request_timeout_seconds,
        )
        _resolve_account_email(acct, service)

        if account_email and acct.email != account_email:
            continue

        stats["accounts"] += 1
        candidates = []
        for candidate in _attachment_materialization_candidates(conn, account_email=acct.email):
            attachment_key_value = (
                str(candidate.get("attachment_key") or "").strip()
                or attachment_key(acct.email, candidate["msg_id"], candidate["part_id"])
            )
            storage_path = str(candidate.get("storage_path") or "").strip()
            abs_path = db_dir / storage_path if storage_path else None
            already_materialized = bool(
                str(candidate.get("storage_kind") or "").strip() == "file"
                and storage_path
                and abs_path is not None
                and abs_path.is_file()
            )
            if missing_only and already_materialized:
                continue
            candidate["attachment_key"] = attachment_key_value
            candidates.append(candidate)
            if remaining is not None and len(candidates) >= remaining:
                break

        stats["selected_attachments"] += len(candidates)
        _emit_progress(
            progress_callback,
            {
                "event": "stage",
                "stage": "attachment_materialize_account_start",
                "account": acct.email,
                "account_index": acct_idx,
                "accounts_total": len(cfg.accounts),
                "selected_attachments": len(candidates),
            },
        )

        account_processed = 0
        account_materialized = 0
        account_failed = 0

        for candidate in candidates:
            account_processed += 1
            stats["processed_attachments"] += 1
            msg_id = candidate["msg_id"]
            part_id = candidate["part_id"]
            attachment_key_value = candidate["attachment_key"]
            try:
                raw_row = conn.execute(
                    "SELECT raw_json FROM raw_messages WHERE msg_id = ?",
                    (msg_id,),
                ).fetchone()
                raw_payload = None
                if raw_row and raw_row[0]:
                    raw_payload = json.loads(str(raw_row[0]))

                payload_bytes = None
                source = ""
                if raw_payload is not None:
                    payload_bytes = extract_inline_attachment_bytes(raw_payload, part_id)
                    if payload_bytes is not None:
                        source = "inline"

                if payload_bytes is None:
                    gmail_attachment_id = str(candidate.get("gmail_attachment_id") or "").strip()
                    if not gmail_attachment_id:
                        raise RuntimeError("attachment bytes unavailable: missing inline data and gmail_attachment_id")
                    payload_bytes = fetch_attachment_bytes(service, msg_id, gmail_attachment_id)
                    source = "gmail_attachment_get"

                relpath = _attachment_materialization_relpath(
                    account_email=acct.email,
                    attachment_key_value=attachment_key_value,
                    mime_type=str(candidate.get("mime_type") or ""),
                )
                abs_path = db_dir / relpath
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = abs_path.with_name(abs_path.name + ".tmp")
                tmp_path.write_bytes(payload_bytes)
                tmp_path.replace(abs_path)

                checksum = hashlib.sha256(payload_bytes).hexdigest()
                materialized_at = datetime.now(timezone.utc).isoformat()
                update_attachment_materialization(
                    conn,
                    msg_id=msg_id,
                    part_id=part_id,
                    attachment_key_value=attachment_key_value,
                    storage_kind="file",
                    storage_path=relpath,
                    content_sha256=checksum,
                    content_size_bytes=len(payload_bytes),
                    materialized_at=materialized_at,
                )
                stats["materialized_attachments"] += 1
                stats["bytes_written"] += len(payload_bytes)
                if source == "inline":
                    stats["inline_sourced"] += 1
                else:
                    stats["gmail_fetched"] += 1
                account_materialized += 1
            except Exception:
                LOG.exception(
                    "Failed to materialize attachment bytes for message_id=%s part_id=%s account=%s",
                    msg_id,
                    part_id,
                    acct.email,
                )
                stats["failed_attachments"] += 1
                account_failed += 1

            processed_since_commit += 1
            if processed_since_commit >= safe_commit_every:
                conn.commit()
                processed_since_commit = 0

            if account_processed % progress_every == 0:
                _emit_progress(
                    progress_callback,
                    {
                        "event": "progress",
                        "stage": "attachment_materialize_account",
                        "account": acct.email,
                        "account_index": acct_idx,
                        "accounts_total": len(cfg.accounts),
                        "selected_attachments": len(candidates),
                        "processed_attachments": stats["processed_attachments"],
                        "materialized_attachments": stats["materialized_attachments"],
                        "failed_attachments": stats["failed_attachments"],
                        "bytes_written": stats["bytes_written"],
                    },
                )

        conn.commit()
        processed_since_commit = 0

        _emit_progress(
            progress_callback,
            {
                "event": "stage",
                "stage": "attachment_materialize_account_done",
                "account": acct.email,
                "account_index": acct_idx,
                "accounts_total": len(cfg.accounts),
                "selected_attachments": len(candidates),
                "account_processed": account_processed,
                "account_materialized": account_materialized,
                "account_failed": account_failed,
            },
        )

        if remaining is not None:
            remaining = max(0, remaining - len(candidates))
            if remaining == 0:
                break

    _emit_progress(
        progress_callback,
        {
            "event": "stage",
            "stage": "attachment_materialize_done",
            **stats,
        },
    )

    return stats


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
                elif _ingest_message_id(conn, cfg, service, acct.email, msg_id):
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
        elif _ingest_message_id(conn, cfg, service, account_email, msg_id):
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
        "selected_attachments": 0,
        "materialized_attachments": 0,
        "attachment_materialization_failed": 0,
        "attachment_bytes_written": 0,
        "inline_attachment_sources": 0,
        "gmail_attachment_fetches": 0,
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
            ingested, materialization = _ingest_message_id_with_attachment_materialization(
                conn,
                cfg,
                service,
                acct.email,
                msg_id,
                materialize_attachments=cfg.gmail_materialize_attachments_on_update,
            )
            if ingested:
                stats["ingested"] += 1
                _merge_attachment_materialization_stats(stats, materialization)
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
        "selected_attachments": 0,
        "materialized_attachments": 0,
        "attachment_materialization_failed": 0,
        "attachment_bytes_written": 0,
        "inline_attachment_sources": 0,
        "gmail_attachment_fetches": 0,
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
                        ingested, materialization = _ingest_message_id_with_attachment_materialization(
                            conn,
                            cfg,
                            state["service"],
                            state["acct"].email,
                            msg_id,
                            materialize_attachments=cfg.gmail_materialize_attachments_on_repair,
                        )
                        if ingested:
                            stats["backfill_ingested"] += 1
                            state["account_ingested"] += 1
                            _merge_attachment_materialization_stats(stats, materialization)
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
