from __future__ import annotations

import json

from sqlcipher3 import dbapi2 as sqlite

from inbox_vault.db import (
    DBLockRetryExhausted,
    _run_with_lock_retry,
    clear_contact_profiles,
    get_cursor,
    message_exists,
    messages_for_contact,
    unenriched_messages,
    upsert_contact_profile,
    upsert_contact_seen,
    upsert_cursor,
    upsert_enrichment,
    upsert_message,
    upsert_raw,
)


def test_schema_and_crud_paths(conn):
    expected_tables = {
        "messages",
        "raw_messages",
        "message_enrichment",
        "contact_stats",
        "contact_profiles",
        "sync_cursors",
        "message_vectors_v2",
        "message_chunk_vectors_v2",
        "vector_index_state_v2",
        "message_fts",
        "message_fts_redacted",
    }
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {r[0] for r in rows}
    assert expected_tables.issubset(names)

    rec = {
        "msg_id": "m1",
        "account_email": "acct@example.com",
        "thread_id": "t1",
        "date_iso": "2023-12-04",
        "internal_ts": 1701684900000,
        "from_addr": "alice@example.com",
        "to_addr": "bob@example.com",
        "subject": "Project update",
        "snippet": "Initial snippet",
        "body_text": "Body 1",
        "labels": ["INBOX"],
        "history_id": 10,
    }

    upsert_message(conn, rec)
    assert message_exists(conn, "m1") is True
    fts_count = conn.execute("SELECT count(*) FROM message_fts WHERE msg_id='m1'").fetchone()[0]
    assert fts_count == 1

    rec["snippet"] = "Updated snippet"
    rec["body_text"] = "Body 2"
    upsert_message(conn, rec)
    stored = conn.execute("SELECT snippet, body_text FROM messages WHERE msg_id='m1'").fetchone()
    assert tuple(stored) == ("Updated snippet", "Body 2")

    upsert_raw(conn, "m1", "acct@example.com", {"raw": "value"})
    raw = conn.execute("SELECT raw_json FROM raw_messages WHERE msg_id='m1'").fetchone()[0]
    assert json.loads(raw)["raw"] == "value"

    assert get_cursor(conn, "acct@example.com", "INBOX,SENT") is None
    upsert_cursor(conn, "acct@example.com", "INBOX,SENT", 55)
    assert get_cursor(conn, "acct@example.com", "INBOX,SENT") == 55

    upsert_contact_seen(conn, "alice@example.com", "Alice")
    upsert_contact_seen(conn, "alice@example.com", None)
    stats = conn.execute(
        "SELECT display_name, message_count FROM contact_stats WHERE contact_email='alice@example.com'"
    ).fetchone()
    assert tuple(stats) == ("Alice", 2)

    pending_before = unenriched_messages(conn)
    assert [r[0] for r in pending_before] == ["m1"]

    upsert_enrichment(conn, "m1", {"category": "work", "importance": 8, "summary": "ok"}, model="m")
    pending_after = unenriched_messages(conn)
    assert pending_after == []

    upsert_contact_profile(conn, "alice@example.com", {"role": "partner"}, model="test-model")
    profile_row = conn.execute(
        "SELECT profile_json, model FROM contact_profiles WHERE contact_email='alice@example.com'"
    ).fetchone()
    assert json.loads(profile_row[0])["role"] == "partner"
    assert profile_row[1] == "test-model"

    samples = messages_for_contact(conn, "alice@example.com", limit=5)
    assert len(samples) == 1


def test_clear_contact_profiles_returns_deleted_count(conn):
    upsert_contact_profile(conn, "alice@example.com", {"role": "partner"}, model="test-model")
    upsert_contact_profile(conn, "bob@example.com", {"role": "vendor"}, model=None)

    cleared = clear_contact_profiles(conn)

    assert cleared == 2
    remaining = conn.execute("SELECT count(*) FROM contact_profiles").fetchone()[0]
    assert remaining == 0


def test_run_with_lock_retry_eventual_success():
    calls = {"n": 0}

    def op():
        calls["n"] += 1
        if calls["n"] < 3:
            raise sqlite.OperationalError("database is locked")
        return "ok"

    result, retries = _run_with_lock_retry(
        op,
        op_name="unit-test-op",
        max_retries=4,
        backoff_base_seconds=0.0,
    )
    assert result == "ok"
    assert retries == 2


def test_run_with_lock_retry_exhausted():
    def op():
        raise sqlite.OperationalError("database is locked")

    try:
        _run_with_lock_retry(
            op,
            op_name="unit-test-op",
            max_retries=1,
            backoff_base_seconds=0.0,
        )
        assert False, "expected DBLockRetryExhausted"
    except DBLockRetryExhausted as exc:
        assert "retries exhausted" in str(exc)
