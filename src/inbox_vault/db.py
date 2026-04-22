from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlcipher3 import dbapi2 as sqlite

from .redaction import (
    REDACTION_POLICY_VERSION,
    is_persistent_redaction_value_allowed,
)


class DBLockRetryExhausted(RuntimeError):
    """Raised when sqlite lock retries are exhausted for a write operation."""

def _is_lock_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "locked" in message or "busy" in message


def _run_with_lock_retry(
    operation,
    *,
    op_name: str,
    max_retries: int,
    backoff_base_seconds: float,
):
    attempts = max(1, int(max_retries) + 1)
    delay = max(0.0, float(backoff_base_seconds))
    for attempt in range(1, attempts + 1):
        try:
            return operation(), max(0, attempt - 1)
        except sqlite.OperationalError as exc:
            if not _is_lock_error(exc):
                raise
            if attempt >= attempts:
                raise DBLockRetryExhausted(
                    f"SQLite lock retries exhausted for {op_name} "
                    f"after {attempts} attempts (last_error={exc})"
                ) from exc
            if delay > 0:
                time.sleep(delay)
                delay = min(delay * 2.0, 2.0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sqlcipher_quote(value: str) -> str:
    return value.replace("'", "''")


def _compose_source_text(subject: str | None, snippet: str | None, body: str | None) -> str:
    return "\n".join(
        [
            f"Subject: {(subject or '').strip()}",
            f"Snippet: {(snippet or '').strip()}",
            f"Body: {(body or '').strip()}",
        ]
    ).strip()


def attachment_key(account_email: str, msg_id: str, part_id: str) -> str:
    payload = "\0".join([str(account_email or ""), str(msg_id or ""), str(part_id or "")])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_conn(db_path: str, password: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite.connect(db_path, timeout=30.0, check_same_thread=False)

    escaped_password = _sqlcipher_quote(password)
    conn.execute(f"PRAGMA key='{escaped_password}';")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS messages (
          msg_id TEXT PRIMARY KEY,
          account_email TEXT NOT NULL,
          thread_id TEXT,
          date_iso TEXT,
          internal_ts INTEGER,
          from_addr TEXT,
          to_addr TEXT,
          subject TEXT,
          snippet TEXT,
          body_text TEXT,
          labels_json TEXT,
          history_id INTEGER,
          last_seen_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS raw_messages (
          msg_id TEXT PRIMARY KEY,
          account_email TEXT NOT NULL,
          raw_json TEXT NOT NULL,
          fetched_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS message_attachments (
          msg_id TEXT NOT NULL REFERENCES messages(msg_id) ON DELETE CASCADE,
          account_email TEXT NOT NULL,
          part_id TEXT NOT NULL,
          attachment_key TEXT NOT NULL DEFAULT '',
          gmail_attachment_id TEXT NOT NULL DEFAULT '',
          mime_type TEXT NOT NULL DEFAULT '',
          filename TEXT NOT NULL DEFAULT '',
          size_bytes INTEGER,
          content_disposition TEXT NOT NULL DEFAULT '',
          content_id TEXT NOT NULL DEFAULT '',
          is_inline INTEGER NOT NULL DEFAULT 0,
          inventory_state TEXT NOT NULL DEFAULT 'metadata_only',
          storage_kind TEXT NOT NULL DEFAULT '',
          storage_path TEXT NOT NULL DEFAULT '',
          content_sha256 TEXT NOT NULL DEFAULT '',
          content_size_bytes INTEGER NOT NULL DEFAULT 0,
          materialized_at TEXT NOT NULL DEFAULT '',
          last_seen_at TEXT NOT NULL,
          PRIMARY KEY (msg_id, part_id)
        );

        CREATE TABLE IF NOT EXISTS message_attachment_inventory_state (
          msg_id TEXT PRIMARY KEY REFERENCES messages(msg_id) ON DELETE CASCADE,
          account_email TEXT NOT NULL,
          inventory_state TEXT NOT NULL DEFAULT 'metadata_only',
          attachment_count INTEGER NOT NULL DEFAULT 0,
          inventoried_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS message_enrichment (
          msg_id TEXT PRIMARY KEY REFERENCES messages(msg_id) ON DELETE CASCADE,
          category TEXT,
          importance INTEGER,
          action TEXT,
          summary TEXT,
          model TEXT,
          enriched_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS contact_stats (
          contact_email TEXT PRIMARY KEY,
          display_name TEXT,
          first_seen TEXT,
          last_seen TEXT,
          message_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS contact_profiles (
          contact_email TEXT PRIMARY KEY,
          profile_json TEXT NOT NULL,
          model TEXT,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sync_cursors (
          account_email TEXT NOT NULL,
          scope TEXT NOT NULL,
          history_id INTEGER NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (account_email, scope)
        );

        CREATE TABLE IF NOT EXISTS message_ingest_triage (
          msg_id TEXT PRIMARY KEY REFERENCES messages(msg_id) ON DELETE CASCADE,
          account_email TEXT NOT NULL,
          stream_id TEXT NOT NULL,
          stream_kind TEXT NOT NULL DEFAULT '',
          subject_family TEXT NOT NULL DEFAULT '',
          sender_domain TEXT NOT NULL DEFAULT '',
          triage_tier TEXT NOT NULL,
          proposed_tier TEXT NOT NULL DEFAULT '',
          applied_tier TEXT NOT NULL DEFAULT '',
          enforcement_mode TEXT NOT NULL DEFAULT 'observe',
          decision_source TEXT NOT NULL DEFAULT 'rules',
          bulk_score INTEGER NOT NULL DEFAULT 0,
          importance_score INTEGER NOT NULL DEFAULT 0,
          novelty_score REAL NOT NULL DEFAULT 0,
          signals_json TEXT NOT NULL,
          evaluated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS message_stream_reputation (
          stream_id TEXT PRIMARY KEY,
          account_email TEXT NOT NULL,
          stream_kind TEXT NOT NULL DEFAULT '',
          sender_domain TEXT NOT NULL DEFAULT '',
          subject_family TEXT NOT NULL DEFAULT '',
          first_seen_at TEXT NOT NULL,
          last_seen_at TEXT NOT NULL,
          observation_count INTEGER NOT NULL DEFAULT 0,
          full_count INTEGER NOT NULL DEFAULT 0,
          light_count INTEGER NOT NULL DEFAULT 0,
          minimal_count INTEGER NOT NULL DEFAULT 0,
          suppressed_count INTEGER NOT NULL DEFAULT 0,
          reputation_score REAL NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS redaction_entries (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          scope_type TEXT NOT NULL,
          scope_id TEXT NOT NULL,
          key_name TEXT NOT NULL,
          placeholder TEXT NOT NULL,
          value_norm TEXT NOT NULL,
          original_value TEXT NOT NULL,
          source_mode TEXT NOT NULL,
          policy_version TEXT NOT NULL DEFAULT '',
          status TEXT NOT NULL DEFAULT 'active',
          validator_name TEXT NOT NULL DEFAULT '',
          detector_sources TEXT NOT NULL DEFAULT '',
          modality TEXT NOT NULL DEFAULT '',
          source_field TEXT NOT NULL DEFAULT '',
          first_seen_at TEXT NOT NULL,
          last_seen_at TEXT NOT NULL,
          hit_count INTEGER NOT NULL DEFAULT 1,
          UNIQUE(scope_type, scope_id, key_name, value_norm),
          UNIQUE(scope_type, scope_id, placeholder)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(
          msg_id UNINDEXED,
          account_email UNINDEXED,
          thread_id UNINDEXED,
          labels_text UNINDEXED,
          content,
          tokenize='unicode61 remove_diacritics 2'
        );

        CREATE TABLE IF NOT EXISTS message_vectors (
          msg_id TEXT NOT NULL REFERENCES messages(msg_id) ON DELETE CASCADE,
          index_level TEXT NOT NULL,
          account_email TEXT NOT NULL,
          thread_id TEXT,
          labels_json TEXT NOT NULL,
          source_text TEXT NOT NULL,
          source_text_redacted TEXT NOT NULL,
          embedding_json TEXT NOT NULL,
          embedding_dim INTEGER NOT NULL,
          embedding_model TEXT NOT NULL,
          content_hash TEXT NOT NULL,
          redaction_policy_version TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (msg_id, index_level)
        );

        CREATE TABLE IF NOT EXISTS message_chunk_vectors (
          chunk_id TEXT NOT NULL,
          index_level TEXT NOT NULL,
          msg_id TEXT NOT NULL REFERENCES messages(msg_id) ON DELETE CASCADE,
          account_email TEXT NOT NULL,
          thread_id TEXT,
          labels_json TEXT NOT NULL,
          chunk_index INTEGER NOT NULL,
          chunk_type TEXT NOT NULL,
          chunk_start INTEGER NOT NULL,
          chunk_end INTEGER NOT NULL,
          chunk_text TEXT NOT NULL,
          chunk_text_redacted TEXT NOT NULL,
          embedding_json TEXT NOT NULL,
          embedding_dim INTEGER NOT NULL,
          embedding_model TEXT NOT NULL,
          content_hash TEXT NOT NULL,
          redaction_policy_version TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (chunk_id, index_level)
        );

        CREATE TABLE IF NOT EXISTS vector_index_state (
          msg_id TEXT NOT NULL REFERENCES messages(msg_id) ON DELETE CASCADE,
          index_level TEXT NOT NULL,
          content_hash TEXT NOT NULL,
          redaction_policy_version TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (msg_id, index_level)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS message_fts_redacted USING fts5(
          msg_id UNINDEXED,
          account_email UNINDEXED,
          thread_id UNINDEXED,
          labels_text UNINDEXED,
          content,
          tokenize='unicode61 remove_diacritics 2'
        );

        CREATE INDEX IF NOT EXISTS idx_messages_account ON messages(account_email);
        CREATE INDEX IF NOT EXISTS idx_message_attachments_account
          ON message_attachments(account_email, inventory_state);
        CREATE INDEX IF NOT EXISTS idx_attachment_inventory_state_account
          ON message_attachment_inventory_state(account_email, inventoried_at);
        CREATE INDEX IF NOT EXISTS idx_ingest_triage_stream ON message_ingest_triage(stream_id);
        CREATE INDEX IF NOT EXISTS idx_ingest_triage_tier ON message_ingest_triage(triage_tier);
        CREATE INDEX IF NOT EXISTS idx_stream_reputation_account ON message_stream_reputation(account_email);
        CREATE INDEX IF NOT EXISTS idx_redaction_scope ON redaction_entries(scope_type, scope_id);
        CREATE INDEX IF NOT EXISTS idx_redaction_placeholder ON redaction_entries(scope_type, scope_id, placeholder);
        CREATE INDEX IF NOT EXISTS idx_vectors_account ON message_vectors(account_email, index_level);
        CREATE INDEX IF NOT EXISTS idx_chunk_vectors_msg ON message_chunk_vectors(msg_id, index_level);
        CREATE INDEX IF NOT EXISTS idx_chunk_vectors_account ON message_chunk_vectors(account_email, index_level);
        CREATE INDEX IF NOT EXISTS idx_vector_state_level ON vector_index_state(index_level);
        """
    )

    redaction_cols = {row[1] for row in conn.execute("PRAGMA table_info(redaction_entries)").fetchall()}
    for column, column_type, default in [
        ("policy_version", "TEXT", "''"),
        ("status", "TEXT", "'active'"),
        ("validator_name", "TEXT", "''"),
        ("detector_sources", "TEXT", "''"),
        ("modality", "TEXT", "''"),
        ("source_field", "TEXT", "''"),
    ]:
        if redaction_cols and column not in redaction_cols:
            conn.execute(
                f"ALTER TABLE redaction_entries ADD COLUMN {column} {column_type} NOT NULL DEFAULT {default}"
            )

    triage_cols = {row[1] for row in conn.execute("PRAGMA table_info(message_ingest_triage)").fetchall()}
    for column, column_type, default in [
        ("proposed_tier", "TEXT", "''"),
        ("applied_tier", "TEXT", "''"),
        ("enforcement_mode", "TEXT", "'observe'"),
    ]:
        if triage_cols and column not in triage_cols:
            conn.execute(
                f"ALTER TABLE message_ingest_triage ADD COLUMN {column} {column_type} NOT NULL DEFAULT {default}"
            )

    attachment_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(message_attachments)").fetchall()
    }
    for column, column_type, default in [
        ("attachment_key", "TEXT", "''"),
        ("storage_kind", "TEXT", "''"),
        ("storage_path", "TEXT", "''"),
        ("content_sha256", "TEXT", "''"),
        ("content_size_bytes", "INTEGER", "0"),
        ("materialized_at", "TEXT", "''"),
    ]:
        if attachment_cols and column not in attachment_cols:
            conn.execute(
                f"ALTER TABLE message_attachments ADD COLUMN {column} {column_type} NOT NULL DEFAULT {default}"
            )

    try:
        conn.execute("SELECT count(*) FROM sqlite_master;").fetchone()
    except sqlite.DatabaseError as exc:  # pragma: no cover
        conn.close()
        raise RuntimeError("Unable to open encrypted database; check password and DB path") from exc

    return conn


def upsert_message_fts(
    conn,
    *,
    msg_id: str,
    account_email: str,
    thread_id: str | None,
    labels: list[str],
    subject: str | None,
    snippet: str | None,
    body_text: str | None,
):
    labels_text = " ".join((label or "").strip().upper() for label in labels if str(label).strip())
    content = _compose_source_text(subject, snippet, body_text)

    conn.execute("DELETE FROM message_fts WHERE msg_id = ?", (msg_id,))
    conn.execute(
        """
        INSERT INTO message_fts (msg_id, account_email, thread_id, labels_text, content)
        VALUES (?, ?, ?, ?, ?)
        """,
        (msg_id, account_email, thread_id, labels_text, content),
    )


def upsert_message_fts_redacted(
    conn,
    *,
    msg_id: str,
    account_email: str,
    thread_id: str | None,
    labels: list[str],
    redacted_content: str,
):
    labels_text = " ".join((label or "").strip().upper() for label in labels if str(label).strip())
    conn.execute("DELETE FROM message_fts_redacted WHERE msg_id = ?", (msg_id,))
    conn.execute(
        """
        INSERT INTO message_fts_redacted (msg_id, account_email, thread_id, labels_text, content)
        VALUES (?, ?, ?, ?, ?)
        """,
        (msg_id, account_email, thread_id, labels_text, redacted_content),
    )


def upsert_message(conn, rec: dict[str, Any]):
    conn.execute(
        """
        INSERT INTO messages (
          msg_id, account_email, thread_id, date_iso, internal_ts, from_addr, to_addr,
          subject, snippet, body_text, labels_json, history_id, last_seen_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(msg_id) DO UPDATE SET
          account_email=excluded.account_email,
          thread_id=excluded.thread_id,
          date_iso=excluded.date_iso,
          internal_ts=excluded.internal_ts,
          from_addr=excluded.from_addr,
          to_addr=excluded.to_addr,
          subject=excluded.subject,
          snippet=excluded.snippet,
          body_text=excluded.body_text,
          labels_json=excluded.labels_json,
          history_id=excluded.history_id,
          last_seen_at=excluded.last_seen_at
        """,
        (
            rec["msg_id"],
            rec["account_email"],
            rec.get("thread_id"),
            rec.get("date_iso"),
            rec.get("internal_ts"),
            rec.get("from_addr"),
            rec.get("to_addr"),
            rec.get("subject"),
            rec.get("snippet"),
            rec.get("body_text"),
            json.dumps(rec.get("labels", [])),
            rec.get("history_id"),
            utc_now(),
        ),
    )
    upsert_message_fts(
        conn,
        msg_id=rec["msg_id"],
        account_email=rec["account_email"],
        thread_id=rec.get("thread_id"),
        labels=list(rec.get("labels", [])),
        subject=rec.get("subject"),
        snippet=rec.get("snippet"),
        body_text=rec.get("body_text"),
    )


def upsert_raw(conn, msg_id: str, account_email: str, raw_payload: dict):
    conn.execute(
        """
        INSERT INTO raw_messages (msg_id, account_email, raw_json, fetched_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(msg_id) DO UPDATE SET
          account_email=excluded.account_email,
          raw_json=excluded.raw_json,
          fetched_at=excluded.fetched_at
        """,
        (msg_id, account_email, json.dumps(raw_payload), utc_now()),
    )


def upsert_message_attachments(
    conn,
    msg_id: str,
    account_email: str,
    attachments: list[dict[str, Any]],
):
    existing_rows = conn.execute(
        """
        SELECT part_id, attachment_key, storage_kind, storage_path, content_sha256,
               content_size_bytes, materialized_at
        FROM message_attachments
        WHERE msg_id = ?
        """,
        (msg_id,),
    ).fetchall()
    existing_materialization = {
        str(row[0] or ""): {
            "attachment_key": str(row[1] or ""),
            "storage_kind": str(row[2] or ""),
            "storage_path": str(row[3] or ""),
            "content_sha256": str(row[4] or ""),
            "content_size_bytes": int(row[5] or 0),
            "materialized_at": str(row[6] or ""),
        }
        for row in existing_rows
        if row and row[0] is not None
    }
    conn.execute("DELETE FROM message_attachments WHERE msg_id = ?", (msg_id,))

    seen_at = utc_now()
    rows: list[tuple[Any, ...]] = []
    for attachment in attachments:
        part_id = str(attachment.get("part_id") or "").strip()
        if not part_id:
            continue
        existing = existing_materialization.get(part_id, {})
        stable_attachment_key = str(existing.get("attachment_key") or "").strip() or attachment_key(
            account_email,
            msg_id,
            part_id,
        )
        rows.append(
            (
                msg_id,
                account_email,
                part_id,
                stable_attachment_key,
                str(attachment.get("gmail_attachment_id") or "").strip(),
                str(attachment.get("mime_type") or "").strip().lower(),
                str(attachment.get("filename") or "").strip(),
                attachment.get("size_bytes"),
                str(attachment.get("content_disposition") or "").strip(),
                str(attachment.get("content_id") or "").strip(),
                1 if attachment.get("is_inline") else 0,
                str(attachment.get("inventory_state") or "metadata_only").strip(),
                str(existing.get("storage_kind") or "").strip(),
                str(existing.get("storage_path") or "").strip(),
                str(existing.get("content_sha256") or "").strip(),
                int(existing.get("content_size_bytes") or 0),
                str(existing.get("materialized_at") or "").strip(),
                seen_at,
            )
        )

    if rows:
        conn.executemany(
            """
            INSERT INTO message_attachments (
              msg_id, account_email, part_id, attachment_key, gmail_attachment_id, mime_type,
              filename, size_bytes, content_disposition, content_id, is_inline, inventory_state,
              storage_kind, storage_path, content_sha256, content_size_bytes, materialized_at,
              last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    conn.execute(
        """
        INSERT INTO message_attachment_inventory_state (
          msg_id, account_email, inventory_state, attachment_count, inventoried_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(msg_id) DO UPDATE SET
          account_email=excluded.account_email,
          inventory_state=excluded.inventory_state,
          attachment_count=excluded.attachment_count,
          inventoried_at=excluded.inventoried_at
        """,
        (msg_id, account_email, "metadata_only", len(rows), seen_at),
    )


def update_attachment_materialization(
    conn,
    *,
    msg_id: str,
    part_id: str,
    attachment_key_value: str,
    storage_kind: str,
    storage_path: str,
    content_sha256: str,
    content_size_bytes: int,
    materialized_at: str,
):
    conn.execute(
        """
        UPDATE message_attachments
        SET attachment_key = ?,
            storage_kind = ?,
            storage_path = ?,
            content_sha256 = ?,
            content_size_bytes = ?,
            materialized_at = ?
        WHERE msg_id = ? AND part_id = ?
        """,
        (
            attachment_key_value,
            storage_kind,
            storage_path,
            content_sha256,
            int(content_size_bytes),
            materialized_at,
            msg_id,
            part_id,
        ),
    )


def message_exists(conn, msg_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM messages WHERE msg_id = ?", (msg_id,)).fetchone()
    return bool(row)


def upsert_cursor(conn, account_email: str, scope: str, history_id: int):
    conn.execute(
        """
        INSERT INTO sync_cursors (account_email, scope, history_id, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(account_email, scope) DO UPDATE SET
          history_id=excluded.history_id,
          updated_at=excluded.updated_at
        """,
        (account_email, scope, history_id, utc_now()),
    )


def get_cursor(conn, account_email: str, scope: str) -> int | None:
    row = conn.execute(
        "SELECT history_id FROM sync_cursors WHERE account_email = ? AND scope = ?",
        (account_email, scope),
    ).fetchone()
    return int(row[0]) if row else None


def get_oldest_internal_ts(conn, account_email: str) -> int | None:
    row = conn.execute(
        """
        SELECT MIN(internal_ts)
        FROM messages
        WHERE account_email = ?
          AND internal_ts IS NOT NULL
          AND internal_ts > 0
        """,
        (account_email,),
    ).fetchone()
    if not row or row[0] is None:
        return None
    return int(row[0])


def get_stream_observation_count(conn, stream_id: str) -> int:
    row = conn.execute(
        "SELECT observation_count FROM message_stream_reputation WHERE stream_id = ?",
        (stream_id,),
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def fetch_applied_light_message_ids(conn) -> set[str]:
    rows = conn.execute(
        "SELECT msg_id FROM message_ingest_triage WHERE COALESCE(applied_tier, '') = 'light'"
    ).fetchall()
    return {str(row[0]) for row in rows if row and row[0] is not None}


def upsert_contact_seen(conn, email: str, display_name: str | None = None):
    now = utc_now()
    row = conn.execute(
        "SELECT message_count, first_seen FROM contact_stats WHERE contact_email = ?",
        (email,),
    ).fetchone()
    if row:
        conn.execute(
            """
            UPDATE contact_stats
            SET message_count = message_count + 1,
                last_seen = ?,
                display_name = COALESCE(display_name, ?)
            WHERE contact_email = ?
            """,
            (now, display_name, email),
        )
    else:
        conn.execute(
            """
            INSERT INTO contact_stats (contact_email, display_name, first_seen, last_seen, message_count)
            VALUES (?, ?, ?, ?, 1)
            """,
            (email, display_name, now, now),
        )


def upsert_message_ingest_triage(conn, triage: dict[str, Any]) -> str | None:
    previous = conn.execute(
        "SELECT stream_id FROM message_ingest_triage WHERE msg_id = ?",
        (triage["msg_id"],),
    ).fetchone()
    previous_stream_id = str(previous[0]) if previous and previous[0] is not None else None

    conn.execute(
        """
        INSERT INTO message_ingest_triage (
          msg_id, account_email, stream_id, stream_kind, subject_family, sender_domain,
          triage_tier, proposed_tier, applied_tier, enforcement_mode,
          decision_source, bulk_score, importance_score, novelty_score,
          signals_json, evaluated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(msg_id) DO UPDATE SET
          account_email=excluded.account_email,
          stream_id=excluded.stream_id,
          stream_kind=excluded.stream_kind,
          subject_family=excluded.subject_family,
          sender_domain=excluded.sender_domain,
          triage_tier=excluded.triage_tier,
          proposed_tier=excluded.proposed_tier,
          applied_tier=excluded.applied_tier,
          enforcement_mode=excluded.enforcement_mode,
          decision_source=excluded.decision_source,
          bulk_score=excluded.bulk_score,
          importance_score=excluded.importance_score,
          novelty_score=excluded.novelty_score,
          signals_json=excluded.signals_json,
          evaluated_at=excluded.evaluated_at
        """,
        (
            triage["msg_id"],
            triage["account_email"],
            triage["stream_id"],
            triage.get("stream_kind", ""),
            triage.get("subject_family", ""),
            triage.get("sender_domain", ""),
            triage.get("triage_tier", triage.get("proposed_tier", "full")),
            triage.get("proposed_tier", triage.get("triage_tier", "full")),
            triage.get("applied_tier", "full"),
            triage.get("enforcement_mode", "observe"),
            triage.get("decision_source", "rules"),
            int(triage.get("bulk_score", 0)),
            int(triage.get("importance_score", 0)),
            float(triage.get("novelty_score", 0.0)),
            triage.get("signals_json", "{}"),
            utc_now(),
        ),
    )
    _refresh_message_stream_reputation(conn, triage["stream_id"])
    if previous_stream_id and previous_stream_id != triage["stream_id"]:
        _refresh_message_stream_reputation(conn, previous_stream_id)
    return previous_stream_id


def _refresh_message_stream_reputation(conn, stream_id: str) -> None:
    row = conn.execute(
        """
        SELECT
          account_email,
          stream_kind,
          sender_domain,
          subject_family,
          count(*) AS observation_count,
          SUM(CASE WHEN COALESCE(applied_tier, triage_tier) = 'full' THEN 1 ELSE 0 END) AS full_count,
          SUM(CASE WHEN COALESCE(applied_tier, triage_tier) = 'light' THEN 1 ELSE 0 END) AS light_count,
          SUM(CASE WHEN COALESCE(applied_tier, triage_tier) = 'minimal' THEN 1 ELSE 0 END) AS minimal_count,
          SUM(CASE WHEN COALESCE(applied_tier, triage_tier) = 'suppressed' THEN 1 ELSE 0 END) AS suppressed_count,
          MIN(evaluated_at) AS first_seen_at,
          MAX(evaluated_at) AS last_seen_at
        FROM message_ingest_triage
        WHERE stream_id = ?
        GROUP BY account_email, stream_kind, sender_domain, subject_family
        ORDER BY observation_count DESC, last_seen_at DESC
        LIMIT 1
        """,
        (stream_id,),
    ).fetchone()
    if not row:
        conn.execute("DELETE FROM message_stream_reputation WHERE stream_id = ?", (stream_id,))
        return

    observation_count = int(row[4] or 0)
    full_count = int(row[5] or 0)
    light_count = int(row[6] or 0)
    minimal_count = int(row[7] or 0)
    suppressed_count = int(row[8] or 0)
    reputation_score = float(full_count - minimal_count - (suppressed_count * 2))

    conn.execute(
        """
        INSERT INTO message_stream_reputation (
          stream_id, account_email, stream_kind, sender_domain, subject_family,
          first_seen_at, last_seen_at, observation_count,
          full_count, light_count, minimal_count, suppressed_count, reputation_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(stream_id) DO UPDATE SET
          account_email=excluded.account_email,
          stream_kind=excluded.stream_kind,
          sender_domain=excluded.sender_domain,
          subject_family=excluded.subject_family,
          first_seen_at=excluded.first_seen_at,
          last_seen_at=excluded.last_seen_at,
          observation_count=excluded.observation_count,
          full_count=excluded.full_count,
          light_count=excluded.light_count,
          minimal_count=excluded.minimal_count,
          suppressed_count=excluded.suppressed_count,
          reputation_score=excluded.reputation_score
        """,
        (
            stream_id,
            row[0],
            row[1],
            row[2],
            row[3],
            row[9],
            row[10],
            observation_count,
            full_count,
            light_count,
            minimal_count,
            suppressed_count,
            reputation_score,
        ),
    )


def ingest_triage_summary(conn) -> dict[str, Any]:
    proposed_tiers = {
        str(row[0]): int(row[1])
        for row in conn.execute(
            """
            SELECT COALESCE(proposed_tier, triage_tier), count(*)
            FROM message_ingest_triage
            GROUP BY COALESCE(proposed_tier, triage_tier)
            """
        ).fetchall()
    }
    applied_tiers = {
        str(row[0]): int(row[1])
        for row in conn.execute(
            """
            SELECT COALESCE(applied_tier, triage_tier), count(*)
            FROM message_ingest_triage
            GROUP BY COALESCE(applied_tier, triage_tier)
            """
        ).fetchall()
    }
    return {
        "messages": int(conn.execute("SELECT count(*) FROM message_ingest_triage").fetchone()[0]),
        "streams": int(conn.execute("SELECT count(*) FROM message_stream_reputation").fetchone()[0]),
        "proposed_tiers": proposed_tiers,
        "applied_tiers": applied_tiers,
        "applied_light_messages": int(
            conn.execute(
                """
                SELECT count(*)
                FROM message_ingest_triage
                WHERE COALESCE(applied_tier, triage_tier) = 'light'
                """
            ).fetchone()[0]
        ),
    }


def upsert_enrichment(conn, msg_id: str, data: dict[str, Any], model: str):
    conn.execute(
        """
        INSERT INTO message_enrichment (msg_id, category, importance, action, summary, model, enriched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(msg_id) DO UPDATE SET
          category=excluded.category,
          importance=excluded.importance,
          action=excluded.action,
          summary=excluded.summary,
          model=excluded.model,
          enriched_at=excluded.enriched_at
        """,
        (
            msg_id,
            data.get("category"),
            int(data.get("importance", 0)) if data.get("importance") is not None else None,
            data.get("action"),
            data.get("summary"),
            model,
            utc_now(),
        ),
    )


def enrichment_repair_candidates(
    conn,
    *,
    limit: int | None = 1000,
    include_degraded: bool = False,
):
    limit_sql = ""
    params: tuple[object, ...] = ()
    if limit is not None:
        safe_limit = max(1, int(limit))
        limit_sql = " LIMIT ?"
        params = (safe_limit,)
    if include_degraded:
        return conn.execute(
            f"""
            SELECT m.msg_id, m.subject, m.snippet, m.body_text, m.from_addr, m.to_addr, m.date_iso
            FROM messages m
            LEFT JOIN message_enrichment e ON e.msg_id = m.msg_id
            WHERE e.msg_id IS NULL OR COALESCE(e.model, '') = 'heuristic-fallback'
            ORDER BY COALESCE(m.internal_ts, 0) DESC
            {limit_sql}
            """,
            params,
        ).fetchall()
    return conn.execute(
        f"""
        SELECT m.msg_id, m.subject, m.snippet, m.body_text, m.from_addr, m.to_addr, m.date_iso
        FROM messages m
        LEFT JOIN message_enrichment e ON e.msg_id = m.msg_id
        WHERE e.msg_id IS NULL
        ORDER BY COALESCE(m.internal_ts, 0) DESC
        {limit_sql}
        """,
        params,
    ).fetchall()


def unenriched_messages(conn, limit: int = 1000):
    return enrichment_repair_candidates(conn, limit=limit, include_degraded=False)


def profile_candidates(conn):
    return conn.execute(
        """
        SELECT
          cs.contact_email,
          cs.display_name,
          cs.message_count,
          cs.first_seen,
          cs.last_seen,
          cp.profile_json,
          cp.model
        FROM contact_stats cs
        LEFT JOIN contact_profiles cp ON cp.contact_email = cs.contact_email
        ORDER BY cs.message_count DESC
        """
    ).fetchall()


def contact_directional_counts(conn, email: str, user_emails: list[str]) -> tuple[int, int]:
    if not user_emails:
        return 0, 0

    normalized_users = sorted(
        {str(item).strip().lower() for item in user_emails if str(item).strip()}
    )
    if not normalized_users:
        return 0, 0

    placeholders = ",".join("?" for _ in normalized_users)

    inbound = int(
        conn.execute(
            f"""
            SELECT count(*)
            FROM messages
            WHERE lower(from_addr) = lower(?)
              AND lower(to_addr) IN ({placeholders})
            """,
            [email, *normalized_users],
        ).fetchone()[0]
    )
    outbound = int(
        conn.execute(
            f"""
            SELECT count(*)
            FROM messages
            WHERE lower(to_addr) = lower(?)
              AND lower(from_addr) IN ({placeholders})
            """,
            [email, *normalized_users],
        ).fetchone()[0]
    )
    return inbound, outbound


def messages_for_contact(conn, email: str, limit: int = 25):
    safe_limit = max(1, int(limit))
    return conn.execute(
        """
        SELECT subject, snippet, body_text, from_addr, to_addr, date_iso
        FROM messages
        WHERE from_addr = ? OR to_addr = ?
        ORDER BY COALESCE(internal_ts, 0) DESC
        LIMIT ?
        """,
        (email, email, safe_limit),
    ).fetchall()


def upsert_contact_profile(conn, email: str, profile: dict[str, Any], model: str | None):
    conn.execute(
        """
        INSERT INTO contact_profiles (contact_email, profile_json, model, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(contact_email) DO UPDATE SET
          profile_json=excluded.profile_json,
          model=excluded.model,
          updated_at=excluded.updated_at
        """,
        (email, json.dumps(profile), model, utc_now()),
    )


def clear_contact_profiles(conn) -> int:
    row = conn.execute("SELECT count(*) FROM contact_profiles").fetchone()
    cleared = int(row[0]) if row and row[0] is not None else 0
    conn.execute("DELETE FROM contact_profiles")
    return cleared


def vector_index_source_rows(conn, *, account_email: str | None = None, limit: int | None = None):
    sql = (
        "SELECT msg_id, account_email, thread_id, subject, snippet, body_text, labels_json "
        "FROM messages"
    )
    params: list[Any] = []
    if account_email:
        sql += " WHERE account_email = ?"
        params.append(account_email)
    sql += " ORDER BY COALESCE(internal_ts, 0) DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(max(1, int(limit)))
    return conn.execute(sql, params).fetchall()


def get_vector_state(conn, msg_id: str, *, index_level: str):
    return conn.execute(
        """
        SELECT content_hash, redaction_policy_version
        FROM vector_index_state
        WHERE msg_id = ? AND index_level = ?
        """,
        (msg_id, index_level),
    ).fetchone()


def upsert_message_vector(
    conn,
    *,
    msg_id: str,
    index_level: str,
    account_email: str,
    thread_id: str | None,
    labels: list[str],
    source_text: str,
    source_text_redacted: str,
    embedding: list[float],
    embedding_model: str,
    content_hash: str,
    redaction_policy_version: str = REDACTION_POLICY_VERSION,
    lock_max_retries: int = 0,
    lock_backoff_base_seconds: float = 0.05,
) -> int:
    _, retries_used = _run_with_lock_retry(
        lambda: conn.execute(
            """
            INSERT INTO message_vectors (
              msg_id, index_level, account_email, thread_id, labels_json, source_text, source_text_redacted,
              embedding_json, embedding_dim, embedding_model, content_hash, redaction_policy_version, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(msg_id, index_level) DO UPDATE SET
              account_email=excluded.account_email,
              thread_id=excluded.thread_id,
              labels_json=excluded.labels_json,
              source_text=excluded.source_text,
              source_text_redacted=excluded.source_text_redacted,
              embedding_json=excluded.embedding_json,
              embedding_dim=excluded.embedding_dim,
              embedding_model=excluded.embedding_model,
              content_hash=excluded.content_hash,
              redaction_policy_version=excluded.redaction_policy_version,
              updated_at=excluded.updated_at
            """,
            (
                msg_id,
                index_level,
                account_email,
                thread_id,
                json.dumps(labels),
                source_text,
                source_text_redacted,
                json.dumps(embedding),
                len(embedding),
                embedding_model,
                content_hash,
                redaction_policy_version,
                utc_now(),
            ),
        ),
        op_name="upsert_message_vector",
        max_retries=lock_max_retries,
        backoff_base_seconds=lock_backoff_base_seconds,
    )
    return retries_used


def delete_message_chunk_vectors(
    conn,
    *,
    msg_id: str,
    index_level: str,
    lock_max_retries: int = 0,
    lock_backoff_base_seconds: float = 0.05,
) -> int:
    _, retries_used = _run_with_lock_retry(
        lambda: conn.execute(
            "DELETE FROM message_chunk_vectors WHERE msg_id = ? AND index_level = ?",
            (msg_id, index_level),
        ),
        op_name="delete_message_chunk_vectors",
        max_retries=lock_max_retries,
        backoff_base_seconds=lock_backoff_base_seconds,
    )
    return retries_used


def upsert_message_chunk_vector(
    conn,
    *,
    chunk_id: str,
    index_level: str,
    msg_id: str,
    account_email: str,
    thread_id: str | None,
    labels: list[str],
    chunk_index: int,
    chunk_type: str,
    chunk_start: int,
    chunk_end: int,
    chunk_text: str,
    chunk_text_redacted: str,
    embedding: list[float],
    embedding_model: str,
    content_hash: str,
    redaction_policy_version: str = REDACTION_POLICY_VERSION,
    lock_max_retries: int = 0,
    lock_backoff_base_seconds: float = 0.05,
) -> int:
    _, retries_used = _run_with_lock_retry(
        lambda: conn.execute(
            """
            INSERT INTO message_chunk_vectors (
              chunk_id, index_level, msg_id, account_email, thread_id, labels_json,
              chunk_index, chunk_type, chunk_start, chunk_end,
              chunk_text, chunk_text_redacted,
              embedding_json, embedding_dim, embedding_model, content_hash, redaction_policy_version, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id, index_level) DO UPDATE SET
              msg_id=excluded.msg_id,
              account_email=excluded.account_email,
              thread_id=excluded.thread_id,
              labels_json=excluded.labels_json,
              chunk_index=excluded.chunk_index,
              chunk_type=excluded.chunk_type,
              chunk_start=excluded.chunk_start,
              chunk_end=excluded.chunk_end,
              chunk_text=excluded.chunk_text,
              chunk_text_redacted=excluded.chunk_text_redacted,
              embedding_json=excluded.embedding_json,
              embedding_dim=excluded.embedding_dim,
              embedding_model=excluded.embedding_model,
              content_hash=excluded.content_hash,
              redaction_policy_version=excluded.redaction_policy_version,
              updated_at=excluded.updated_at
            """,
            (
                chunk_id,
                index_level,
                msg_id,
                account_email,
                thread_id,
                json.dumps(labels),
                chunk_index,
                chunk_type,
                chunk_start,
                chunk_end,
                chunk_text,
                chunk_text_redacted,
                json.dumps(embedding),
                len(embedding),
                embedding_model,
                content_hash,
                redaction_policy_version,
                utc_now(),
            ),
        ),
        op_name="upsert_message_chunk_vector",
        max_retries=lock_max_retries,
        backoff_base_seconds=lock_backoff_base_seconds,
    )
    return retries_used


def upsert_vector_state(
    conn,
    *,
    msg_id: str,
    index_level: str,
    content_hash: str,
    redaction_policy_version: str = REDACTION_POLICY_VERSION,
    lock_max_retries: int = 0,
    lock_backoff_base_seconds: float = 0.05,
) -> int:
    _, retries_used = _run_with_lock_retry(
        lambda: conn.execute(
            """
            INSERT INTO vector_index_state (
              msg_id, index_level, content_hash, redaction_policy_version, updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(msg_id, index_level) DO UPDATE SET
              content_hash=excluded.content_hash,
              redaction_policy_version=excluded.redaction_policy_version,
              updated_at=excluded.updated_at
            """,
            (msg_id, index_level, content_hash, redaction_policy_version, utc_now()),
        ),
        op_name="upsert_vector_state",
        max_retries=lock_max_retries,
        backoff_base_seconds=lock_backoff_base_seconds,
    )
    return retries_used


def fetch_redaction_entries(
    conn,
    *,
    scope_type: str,
    scope_id: str,
) -> list[tuple[str, str, str, str]]:
    rows = conn.execute(
        """
        SELECT key_name, placeholder, value_norm, original_value
        FROM redaction_entries
        WHERE scope_type = ? AND scope_id = ? AND status = 'active'
        ORDER BY key_name, placeholder
        """,
        (scope_type, scope_id),
    ).fetchall()
    out: list[tuple[str, str, str, str]] = []
    for row in rows:
        key_name = str(row[0])
        placeholder = str(row[1])
        value_norm = str(row[2])
        original_value = str(row[3])
        if not is_persistent_redaction_value_allowed(key_name, original_value):
            continue
        out.append((key_name, placeholder, value_norm, original_value))
    return out


def prune_invalid_redaction_entries(
    conn,
    *,
    scope_type: str,
    scope_id: str,
    lock_max_retries: int = 0,
    lock_backoff_base_seconds: float = 0.05,
) -> int:
    rows = conn.execute(
        """
        SELECT id, key_name, original_value
        FROM redaction_entries
        WHERE scope_type = ? AND scope_id = ? AND status = 'active'
        """,
        (scope_type, scope_id),
    ).fetchall()
    invalid_ids = [
        int(row[0])
        for row in rows
        if not is_persistent_redaction_value_allowed(str(row[1]), str(row[2]))
    ]
    if not invalid_ids:
        return 0

    now = utc_now()

    def _write():
        conn.executemany(
            "UPDATE redaction_entries SET status = 'rejected', last_seen_at = ? WHERE id = ?",
            [(now, row_id) for row_id in invalid_ids],
        )

    _run_with_lock_retry(
        _write,
        op_name="prune_invalid_redaction_entries",
        max_retries=lock_max_retries,
        backoff_base_seconds=lock_backoff_base_seconds,
    )
    return len(invalid_ids)


def upsert_redaction_entries(
    conn,
    *,
    scope_type: str,
    scope_id: str,
    entries: list[dict[str, str]],
    lock_max_retries: int = 0,
    lock_backoff_base_seconds: float = 0.05,
) -> int:
    if not entries:
        return 0
    sanitized_entries = [
        entry
        for entry in entries
        if is_persistent_redaction_value_allowed(
            str(entry.get("key_name") or ""),
            str(entry.get("original_value") or ""),
        )
    ]
    if not sanitized_entries:
        return 0

    now = utc_now()

    def _entry_params(entry: dict[str, str]) -> tuple[Any, ...]:
        return (
            scope_type,
            scope_id,
            entry["key_name"],
            entry["placeholder"],
            entry["value_norm"],
            entry["original_value"],
            entry.get("source_mode", "unknown"),
            entry.get("policy_version", REDACTION_POLICY_VERSION),
            entry.get("status", "active"),
            entry.get("validator_name", "typed_v1"),
            entry.get("detector_sources", entry.get("source_mode", "unknown")),
            entry.get("modality", ""),
            entry.get("source_field", ""),
            now,
            now,
        )

    def _upsert_by_identity(entry: dict[str, str]) -> None:
        conn.execute(
            """
            INSERT INTO redaction_entries (
              scope_type, scope_id, key_name, placeholder, value_norm,
              original_value, source_mode, policy_version, status, validator_name, detector_sources,
              modality, source_field, first_seen_at, last_seen_at, hit_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(scope_type, scope_id, key_name, value_norm) DO UPDATE SET
              placeholder=excluded.placeholder,
              original_value=excluded.original_value,
              source_mode=excluded.source_mode,
              policy_version=excluded.policy_version,
              status=excluded.status,
              validator_name=excluded.validator_name,
              detector_sources=excluded.detector_sources,
              modality=excluded.modality,
              source_field=excluded.source_field,
              last_seen_at=excluded.last_seen_at,
              hit_count=redaction_entries.hit_count + 1
            """,
            _entry_params(entry),
        )

    def _write():
        for entry in sanitized_entries:
            try:
                _upsert_by_identity(entry)
            except sqlite.IntegrityError as exc:
                if "redaction_entries.scope_type, redaction_entries.scope_id, redaction_entries.placeholder" not in str(exc):
                    raise
                conn.execute(
                    """
                    UPDATE redaction_entries
                    SET
                      key_name = ?,
                      value_norm = ?,
                      original_value = ?,
                      source_mode = ?,
                      policy_version = ?,
                      status = ?,
                      validator_name = ?,
                      detector_sources = ?,
                      modality = ?,
                      source_field = ?,
                      last_seen_at = ?,
                      hit_count = hit_count + 1
                    WHERE scope_type = ? AND scope_id = ? AND placeholder = ?
                    """,
                    (
                        entry["key_name"],
                        entry["value_norm"],
                        entry["original_value"],
                        entry.get("source_mode", "unknown"),
                        entry.get("policy_version", REDACTION_POLICY_VERSION),
                        entry.get("status", "active"),
                        entry.get("validator_name", "typed_v1"),
                        entry.get("detector_sources", entry.get("source_mode", "unknown")),
                        entry.get("modality", ""),
                        entry.get("source_field", ""),
                        now,
                        scope_type,
                        scope_id,
                        entry["placeholder"],
                    ),
                )

    _, retries_used = _run_with_lock_retry(
        _write,
        op_name="upsert_redaction_entries",
        max_retries=lock_max_retries,
        backoff_base_seconds=lock_backoff_base_seconds,
    )
    return retries_used


def unredact_with_scope(
    conn,
    *,
    scope_type: str,
    scope_id: str,
    text: str,
) -> str:
    if not text:
        return ""
    rows = conn.execute(
        """
        SELECT key_name, placeholder, original_value
        FROM redaction_entries
        WHERE scope_type = ? AND scope_id = ? AND status = 'active'
        ORDER BY length(placeholder) DESC
        """,
        (scope_type, scope_id),
    ).fetchall()
    out = text
    for key_name, placeholder, original_value in rows:
        if not is_persistent_redaction_value_allowed(str(key_name), str(original_value)):
            continue
        out = out.replace(str(placeholder), str(original_value))
    return out


def fetch_chunk_vectors_for_search(
    conn,
    *,
    index_level: str,
    account_email: str | None = None,
    label: str | None = None,
    from_ts_ms: int | None = None,
    to_ts_ms: int | None = None,
):
    sql = (
        "SELECT c.chunk_id, c.msg_id, c.account_email, c.thread_id, c.labels_json, c.chunk_index, "
        "c.chunk_type, c.chunk_start, c.chunk_end, c.chunk_text, c.chunk_text_redacted, c.embedding_json, "
        "c.embedding_model FROM message_chunk_vectors c "
        "JOIN messages m ON m.msg_id = c.msg_id"
    )
    params: list[Any] = []
    clauses: list[str] = ["c.index_level = ?"]
    params.append(index_level)
    if account_email:
        clauses.append("c.account_email = ?")
        params.append(account_email)
    if from_ts_ms is not None:
        clauses.append("COALESCE(m.internal_ts, 0) >= ?")
        params.append(int(from_ts_ms))
    if to_ts_ms is not None:
        clauses.append("COALESCE(m.internal_ts, 0) < ?")
        params.append(int(to_ts_ms))
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    rows = conn.execute(sql, params).fetchall()
    if not label:
        return rows
    filtered = []
    wanted = label.strip().upper()
    for row in rows:
        labels = [v.upper() for v in json.loads(row[4] or "[]")]
        if wanted in labels:
            filtered.append(row)
    return filtered


def fetch_vectors_for_search(
    conn,
    *,
    index_level: str,
    account_email: str | None = None,
    label: str | None = None,
    from_ts_ms: int | None = None,
    to_ts_ms: int | None = None,
):
    sql = (
        "SELECT v.msg_id, v.account_email, v.thread_id, v.labels_json, v.source_text, v.source_text_redacted, "
        "v.embedding_json, v.embedding_model FROM message_vectors v "
        "JOIN messages m ON m.msg_id = v.msg_id"
    )
    params: list[Any] = []
    clauses: list[str] = ["v.index_level = ?"]
    params.append(index_level)
    if account_email:
        clauses.append("v.account_email = ?")
        params.append(account_email)
    if from_ts_ms is not None:
        clauses.append("COALESCE(m.internal_ts, 0) >= ?")
        params.append(int(from_ts_ms))
    if to_ts_ms is not None:
        clauses.append("COALESCE(m.internal_ts, 0) < ?")
        params.append(int(to_ts_ms))
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    rows = conn.execute(sql, params).fetchall()
    if not label:
        return rows
    filtered = []
    wanted = label.strip().upper()
    for row in rows:
        labels = [v.upper() for v in json.loads(row[3] or "[]")]
        if wanted in labels:
            filtered.append(row)
    return filtered


def fetch_messages_by_ids(conn, msg_ids: list[str], *, index_level: str):
    if not msg_ids:
        return {}
    placeholders = ",".join("?" for _ in msg_ids)
    rows = conn.execute(
        f"""
        SELECT
          m.msg_id,
          m.account_email,
          m.thread_id,
          m.labels_json,
          m.internal_ts,
          m.subject,
          m.snippet,
          m.body_text,
          v.source_text,
          v.source_text_redacted
        FROM messages m
        LEFT JOIN message_vectors v ON v.msg_id = m.msg_id AND v.index_level = ?
        WHERE m.msg_id IN ({placeholders})
        """,
        [index_level, *msg_ids],
    ).fetchall()
    return {row[0]: row for row in rows}


def vector_level_counts(conn) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for level, n in conn.execute(
        "SELECT index_level, COUNT(*) FROM message_vectors GROUP BY index_level ORDER BY index_level"
    ).fetchall():
        counts.setdefault(str(level), {})["messages"] = int(n)
    for level, n in conn.execute(
        "SELECT index_level, COUNT(*) FROM message_chunk_vectors GROUP BY index_level ORDER BY index_level"
    ).fetchall():
        counts.setdefault(str(level), {})["chunks"] = int(n)
    return counts


def lexical_search_rows(
    conn,
    *,
    query: str,
    account_email: str | None = None,
    label: str | None = None,
    from_ts_ms: int | None = None,
    to_ts_ms: int | None = None,
    limit: int = 50,
):
    safe_limit = max(1, int(limit))

    sql = """
        SELECT
          f.msg_id,
          m.account_email,
          m.thread_id,
          m.labels_json,
          m.subject,
          m.snippet,
          m.body_text,
          bm25(message_fts) AS bm25_score
        FROM message_fts f
        JOIN messages m ON m.msg_id = f.msg_id
        WHERE message_fts MATCH ?
        """
    params: list[Any] = [query]
    if account_email:
        sql += " AND m.account_email = ?"
        params.append(account_email)
    if from_ts_ms is not None:
        sql += " AND COALESCE(m.internal_ts, 0) >= ?"
        params.append(int(from_ts_ms))
    if to_ts_ms is not None:
        sql += " AND COALESCE(m.internal_ts, 0) < ?"
        params.append(int(to_ts_ms))
    sql += " ORDER BY bm25_score LIMIT ?"
    params.append(safe_limit)

    rows = conn.execute(sql, params).fetchall()
    if not label:
        return rows

    wanted = label.strip().upper()
    filtered = []
    for row in rows:
        labels = [v.upper() for v in json.loads(row[3] or "[]")]
        if wanted in labels:
            filtered.append(row)
    return filtered


def lexical_search_rows_redacted(
    conn,
    *,
    query: str,
    account_email: str | None = None,
    label: str | None = None,
    from_ts_ms: int | None = None,
    to_ts_ms: int | None = None,
    limit: int = 50,
):
    safe_limit = max(1, int(limit))
    sql = """
        SELECT
          f.msg_id,
          m.account_email,
          m.thread_id,
          m.labels_json,
          m.subject,
          m.snippet,
          m.body_text,
          bm25(message_fts_redacted) AS bm25_score
        FROM message_fts_redacted f
        JOIN messages m ON m.msg_id = f.msg_id
        WHERE message_fts_redacted MATCH ?
        """
    params: list[Any] = [query]
    if account_email:
        sql += " AND m.account_email = ?"
        params.append(account_email)
    if from_ts_ms is not None:
        sql += " AND COALESCE(m.internal_ts, 0) >= ?"
        params.append(int(from_ts_ms))
    if to_ts_ms is not None:
        sql += " AND COALESCE(m.internal_ts, 0) < ?"
        params.append(int(to_ts_ms))
    sql += " ORDER BY bm25_score LIMIT ?"
    params.append(safe_limit)
    rows = conn.execute(sql, params).fetchall()
    if not label:
        return rows
    wanted = label.strip().upper()
    filtered = []
    for row in rows:
        labels = [v.upper() for v in json.loads(row[3] or "[]")]
        if wanted in labels:
            filtered.append(row)
    return filtered
