from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import hashlib
import json

import pytest
from googleapiclient.errors import HttpError

from inbox_vault import ingest
from inbox_vault.config import AccountConfig, IngestTriageConfig
from inbox_vault.db import (
    get_cursor,
    message_exists,
    upsert_message,
    upsert_message_attachments,
    upsert_raw,
)
from inbox_vault.vectors import count_pending_vector_updates
from tests.factories import gmail_message_payload


class _FakeResp:
    status = 404
    reason = "Not Found"


@pytest.fixture(autouse=True)
def _skip_email_resolve(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ingest, "_resolve_account_email", lambda *_args, **_kw: None)


def test_backfill_ingests_and_skips_existing(conn, app_cfg, monkeypatch: pytest.MonkeyPatch):
    service = object()

    monkeypatch.setattr(ingest, "get_service", lambda *_args, **_kwargs: service)
    monkeypatch.setattr(ingest, "list_message_ids_paged", lambda *_args, **_kwargs: ["m1", "m2"])
    monkeypatch.setattr(ingest, "get_profile_history_id", lambda *_: 999)

    payloads = {
        "m1": gmail_message_payload("m1", history_id=10),
        "m2": gmail_message_payload("m2", history_id=11),
    }
    monkeypatch.setattr(ingest, "fetch_full_message_payload", lambda _svc, mid: payloads[mid])

    first = ingest.backfill(conn, app_cfg)
    second = ingest.backfill(conn, app_cfg)

    assert first == {"accounts": 1, "ingested": 2, "skipped_existing": 0, "failed": 0}
    assert second == {"accounts": 1, "ingested": 0, "skipped_existing": 2, "failed": 0}
    assert get_cursor(conn, app_cfg.accounts[0].email, ingest.MAILBOX_SCOPE) == 999


def test_update_uses_cursor_and_ingests_incremental(conn, app_cfg, monkeypatch: pytest.MonkeyPatch):
    service = object()
    monkeypatch.setattr(ingest, "get_service", lambda *_args, **_kwargs: service)
    monkeypatch.setattr(ingest, "get_profile_history_id", lambda *_: 400)

    seen_start_ids: list[int] = []

    def _list_incremental(_svc, start_id):
        seen_start_ids.append(start_id)
        return ({"m3", "m4"}, 450)

    monkeypatch.setattr(ingest, "list_incremental_added_ids", _list_incremental)
    monkeypatch.setattr(
        ingest,
        "fetch_full_message_payload",
        lambda _svc, mid: gmail_message_payload(mid, history_id=450, labels=["INBOX"]),
    )

    out = ingest.update(conn, app_cfg)

    assert out == {
        "accounts": 1,
        "new_ids": 2,
        "ingested": 2,
        "cursor_resets": 0,
        "failed": 0,
    }
    assert seen_start_ids == [400]
    assert get_cursor(conn, app_cfg.accounts[0].email, ingest.MAILBOX_SCOPE) == 450


def test_update_resets_cursor_on_404_history_gap(conn, app_cfg, monkeypatch: pytest.MonkeyPatch):
    service = object()
    monkeypatch.setattr(ingest, "get_service", lambda *_args, **_kwargs: service)
    monkeypatch.setattr(ingest, "get_profile_history_id", lambda *_: 777)

    def _raise_404(_svc, _start_id):
        raise HttpError(resp=_FakeResp(), content=b"{}")

    monkeypatch.setattr(ingest, "list_incremental_added_ids", _raise_404)

    out = ingest.update(conn, app_cfg)

    assert out == {
        "accounts": 1,
        "new_ids": 0,
        "ingested": 0,
        "cursor_resets": 1,
        "failed": 0,
    }
    assert get_cursor(conn, app_cfg.accounts[0].email, ingest.MAILBOX_SCOPE) == 777


def test_update_counts_failed_ingest_and_continues(conn, app_cfg, monkeypatch: pytest.MonkeyPatch):
    service = object()
    monkeypatch.setattr(ingest, "get_service", lambda *_args, **_kwargs: service)
    monkeypatch.setattr(ingest, "get_profile_history_id", lambda *_: 400)
    monkeypatch.setattr(
        ingest, "list_incremental_added_ids", lambda *_args, **_kwargs: ({"m-ok", "m-bad"}, 450)
    )

    def _fetch(_svc, msg_id):
        if msg_id == "m-bad":
            raise RuntimeError("synthetic fetch error")
        return gmail_message_payload(msg_id, history_id=450, labels=["INBOX"])

    monkeypatch.setattr(ingest, "fetch_full_message_payload", _fetch)

    out = ingest.update(conn, app_cfg)
    assert out == {
        "accounts": 1,
        "new_ids": 2,
        "ingested": 1,
        "cursor_resets": 0,
        "failed": 1,
    }


def test_update_persists_attachment_inventory_metadata_only(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    service = object()
    monkeypatch.setattr(ingest, "get_service", lambda *_args, **_kwargs: service)
    monkeypatch.setattr(ingest, "get_profile_history_id", lambda *_: 400)
    monkeypatch.setattr(
        ingest, "list_incremental_added_ids", lambda *_args, **_kwargs: ({"m-attach"}, 450)
    )

    payload = gmail_message_payload(
        "m-attach",
        history_id=450,
        labels=["INBOX"],
        attachments=[
            {
                "part_id": "2",
                "attachment_id": "att-pdf",
                "mime_type": "application/pdf",
                "filename": "invoice.pdf",
                "size_bytes": 2048,
                "content_disposition": "attachment",
            },
            {
                "part_id": "3",
                "attachment_id": "att-inline",
                "mime_type": "image/png",
                "filename": "logo.png",
                "size_bytes": 512,
                "content_disposition": "inline",
                "content_id": "<logo-1>",
            },
        ],
    )
    monkeypatch.setattr(ingest, "fetch_full_message_payload", lambda *_args, **_kwargs: payload)

    out = ingest.update(conn, app_cfg)

    assert out == {
        "accounts": 1,
        "new_ids": 1,
        "ingested": 1,
        "cursor_resets": 0,
        "failed": 0,
    }
    rows = conn.execute(
        """
        SELECT part_id, gmail_attachment_id, mime_type, filename, size_bytes,
               content_disposition, content_id, is_inline, inventory_state
        FROM message_attachments
        WHERE msg_id = ?
        ORDER BY part_id
        """,
        ("m-attach",),
    ).fetchall()
    assert rows == [
        ("2", "att-pdf", "application/pdf", "invoice.pdf", 2048, "attachment", "", 0, "metadata_only"),
        ("3", "att-inline", "image/png", "logo.png", 512, "inline", "<logo-1>", 1, "metadata_only"),
    ]
    state_row = conn.execute(
        """
        SELECT attachment_count, inventory_state
        FROM message_attachment_inventory_state
        WHERE msg_id = ?
        """,
        ("m-attach",),
    ).fetchone()
    assert state_row == (2, "metadata_only")


def test_backfill_attachment_inventory_missing_only_refreshes_only_uninventoried_messages(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    service = object()
    monkeypatch.setattr(ingest, "get_service", lambda *_args, **_kwargs: service)

    upsert_message(
        conn,
        {
            "msg_id": "m-missing",
            "account_email": app_cfg.accounts[0].email,
            "thread_id": "t-missing",
            "date_iso": "2026-04-18",
            "internal_ts": 1713398400000,
            "from_addr": "sender@example.com",
            "to_addr": "acct@example.com",
            "subject": "missing",
            "snippet": "missing",
            "body_text": "missing",
            "labels": ["INBOX"],
            "history_id": 1,
        },
    )
    upsert_message(
        conn,
        {
            "msg_id": "m-existing",
            "account_email": app_cfg.accounts[0].email,
            "thread_id": "t-existing",
            "date_iso": "2026-04-17",
            "internal_ts": 1713312000000,
            "from_addr": "sender@example.com",
            "to_addr": "acct@example.com",
            "subject": "existing",
            "snippet": "existing",
            "body_text": "existing",
            "labels": ["INBOX"],
            "history_id": 2,
        },
    )
    conn.commit()
    upsert_message_attachments(
        conn,
        "m-existing",
        app_cfg.accounts[0].email,
        [
            {
                "part_id": "2",
                "gmail_attachment_id": "att-existing",
                "mime_type": "application/pdf",
                "filename": "existing.pdf",
                "size_bytes": 128,
                "content_disposition": "attachment",
                "content_id": "",
                "is_inline": False,
                "inventory_state": "metadata_only",
            }
        ],
    )
    conn.commit()

    fetched_ids: list[str] = []

    def _fetch(_svc, msg_id):
        fetched_ids.append(msg_id)
        return gmail_message_payload(
            msg_id,
            history_id=500,
            labels=["INBOX"],
            attachments=[
                {
                    "part_id": "2",
                    "attachment_id": f"att-{msg_id}",
                    "mime_type": "application/pdf",
                    "filename": f"{msg_id}.pdf",
                    "size_bytes": 2048,
                    "content_disposition": "attachment",
                }
            ],
        )

    monkeypatch.setattr(ingest, "fetch_full_message_payload", _fetch)

    out = ingest.backfill_attachment_inventory(conn, app_cfg)

    assert out == {
        "accounts": 1,
        "limit": None,
        "missing_only": True,
        "selected_messages": 1,
        "processed_messages": 1,
        "refreshed_messages": 1,
        "failed_messages": 0,
        "attachments_upserted": 1,
        "messages_with_attachments": 1,
        "messages_without_attachments": 0,
    }
    assert fetched_ids == ["m-missing"]
    state_rows = conn.execute(
        """
        SELECT msg_id, attachment_count
        FROM message_attachment_inventory_state
        ORDER BY msg_id
        """
    ).fetchall()
    assert state_rows == [("m-existing", 1), ("m-missing", 1)]


def test_backfill_attachment_inventory_all_records_zero_attachment_messages(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    service = object()
    monkeypatch.setattr(ingest, "get_service", lambda *_args, **_kwargs: service)

    upsert_message(
        conn,
        {
            "msg_id": "m-no-attach",
            "account_email": app_cfg.accounts[0].email,
            "thread_id": "t-none",
            "date_iso": "2026-04-18",
            "internal_ts": 1713398400000,
            "from_addr": "sender@example.com",
            "to_addr": "acct@example.com",
            "subject": "none",
            "snippet": "none",
            "body_text": "none",
            "labels": ["INBOX"],
            "history_id": 1,
        },
    )
    conn.commit()

    monkeypatch.setattr(
        ingest,
        "fetch_full_message_payload",
        lambda *_args, **_kwargs: gmail_message_payload(
            "m-no-attach",
            history_id=600,
            labels=["INBOX"],
            attachments=[],
        ),
    )

    first = ingest.backfill_attachment_inventory(conn, app_cfg)
    second = ingest.backfill_attachment_inventory(conn, app_cfg)

    assert first["messages_without_attachments"] == 1
    assert second["selected_messages"] == 0
    state_row = conn.execute(
        """
        SELECT attachment_count, inventory_state
        FROM message_attachment_inventory_state
        WHERE msg_id = ?
        """,
        ("m-no-attach",),
    ).fetchone()
    assert state_row == (0, "metadata_only")


def test_materialize_attachment_bytes_uses_inline_payload_data(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    service = object()
    monkeypatch.setattr(ingest, "get_service", lambda *_args, **_kwargs: service)
    monkeypatch.setattr(
        ingest,
        "fetch_attachment_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("inline attachment should not call Gmail attachment fetch")
        ),
    )

    msg_id = "m-inline-cache"
    raw_payload = gmail_message_payload(
        msg_id,
        history_id=700,
        labels=["INBOX"],
        attachments=[
            {
                "part_id": "2",
                "attachment_id": "att-inline-cache",
                "mime_type": "application/pdf",
                "filename": "report.pdf",
                "size_bytes": 11,
                "content_disposition": "attachment",
                "inline_data_text": "inline-bytes",
            }
        ],
    )
    upsert_message(
        conn,
        {
            "msg_id": msg_id,
            "account_email": app_cfg.accounts[0].email,
            "thread_id": "t-inline-cache",
            "date_iso": "2026-04-18",
            "internal_ts": 1713398400000,
            "from_addr": "sender@example.com",
            "to_addr": "acct@example.com",
            "subject": "inline cache",
            "snippet": "inline cache",
            "body_text": "inline cache",
            "labels": ["INBOX"],
            "history_id": 1,
        },
    )
    upsert_raw(conn, msg_id, app_cfg.accounts[0].email, raw_payload)
    upsert_message_attachments(
        conn,
        msg_id,
        app_cfg.accounts[0].email,
        [
            {
                "part_id": "2",
                "gmail_attachment_id": "att-inline-cache",
                "mime_type": "application/pdf",
                "filename": "report.pdf",
                "size_bytes": 11,
                "content_disposition": "attachment",
                "content_id": "",
                "is_inline": False,
                "inventory_state": "metadata_only",
            }
        ],
    )
    conn.commit()

    out = ingest.materialize_attachment_bytes(conn, app_cfg)

    assert out["selected_attachments"] == 1
    assert out["materialized_attachments"] == 1
    assert out["inline_sourced"] == 1
    assert out["gmail_fetched"] == 0

    row = conn.execute(
        """
        SELECT storage_kind, storage_path, content_sha256, content_size_bytes
        FROM message_attachments
        WHERE msg_id = ? AND part_id = ?
        """,
        (msg_id, "2"),
    ).fetchone()
    assert row is not None
    assert row[0] == "file"
    cache_path = Path(app_cfg.db.path).resolve().parent / str(row[1])
    assert cache_path.is_file()
    assert cache_path.read_bytes() == b"inline-bytes"
    assert row[2] == hashlib.sha256(b"inline-bytes").hexdigest()
    assert row[3] == len(b"inline-bytes")


def test_materialize_attachment_bytes_fetches_via_gmail_and_recovers_missing_cache(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    service = object()
    monkeypatch.setattr(ingest, "get_service", lambda *_args, **_kwargs: service)

    msg_id = "m-gmail-cache"
    raw_payload = gmail_message_payload(
        msg_id,
        history_id=701,
        labels=["INBOX"],
        attachments=[
            {
                "part_id": "2",
                "attachment_id": "att-gmail-cache",
                "mime_type": "image/jpeg",
                "filename": "photo.jpg",
                "size_bytes": 9,
                "content_disposition": "attachment",
            }
        ],
    )
    upsert_message(
        conn,
        {
            "msg_id": msg_id,
            "account_email": app_cfg.accounts[0].email,
            "thread_id": "t-gmail-cache",
            "date_iso": "2026-04-18",
            "internal_ts": 1713398400000,
            "from_addr": "sender@example.com",
            "to_addr": "acct@example.com",
            "subject": "gmail cache",
            "snippet": "gmail cache",
            "body_text": "gmail cache",
            "labels": ["INBOX"],
            "history_id": 1,
        },
    )
    upsert_raw(conn, msg_id, app_cfg.accounts[0].email, raw_payload)
    upsert_message_attachments(
        conn,
        msg_id,
        app_cfg.accounts[0].email,
        [
            {
                "part_id": "2",
                "gmail_attachment_id": "att-gmail-cache",
                "mime_type": "image/jpeg",
                "filename": "photo.jpg",
                "size_bytes": 9,
                "content_disposition": "attachment",
                "content_id": "",
                "is_inline": False,
                "inventory_state": "metadata_only",
            }
        ],
    )
    conn.commit()

    fetch_calls: list[tuple[str, str]] = []

    def _fetch_attachment_bytes(_svc, fetched_msg_id, attachment_id):
        fetch_calls.append((fetched_msg_id, attachment_id))
        return b"jpeg-bytes"

    monkeypatch.setattr(ingest, "fetch_attachment_bytes", _fetch_attachment_bytes)

    first = ingest.materialize_attachment_bytes(conn, app_cfg)
    assert first["gmail_fetched"] == 1
    row = conn.execute(
        """
        SELECT storage_path
        FROM message_attachments
        WHERE msg_id = ? AND part_id = ?
        """,
        (msg_id, "2"),
    ).fetchone()
    assert row is not None
    cache_path = Path(app_cfg.db.path).resolve().parent / str(row[0])
    assert cache_path.is_file()
    cache_path.unlink()

    second = ingest.materialize_attachment_bytes(conn, app_cfg)
    assert second["selected_attachments"] == 1
    assert second["materialized_attachments"] == 1
    assert second["gmail_fetched"] == 1
    assert fetch_calls == [
        (msg_id, "att-gmail-cache"),
        (msg_id, "att-gmail-cache"),
    ]


def test_update_persists_observe_only_ingest_triage(conn, app_cfg, monkeypatch: pytest.MonkeyPatch):
    triage_cfg = replace(app_cfg, ingest_triage=IngestTriageConfig(enabled=True, mode="observe"))
    service = object()
    monkeypatch.setattr(ingest, "get_service", lambda *_args, **_kwargs: service)
    monkeypatch.setattr(ingest, "get_profile_history_id", lambda *_: 400)
    monkeypatch.setattr(
        ingest, "list_incremental_added_ids", lambda *_args, **_kwargs: ({"m-triage"}, 450)
    )

    payload = gmail_message_payload(
        "m-triage",
        history_id=450,
        from_addr="no-reply@example.com",
        to_addr="acct@example.com",
        subject="Weekly Digest 2026-04-17",
        body_text="Top stories for you",
        labels=["INBOX", "CATEGORY_PROMOTIONS"],
    )
    payload["payload"]["headers"].extend(
        [
            {"name": "List-Id", "value": "digest.example.com"},
            {"name": "List-Unsubscribe", "value": "<https://example.com/unsub>"},
            {"name": "Precedence", "value": "bulk"},
        ]
    )
    monkeypatch.setattr(ingest, "fetch_full_message_payload", lambda *_args, **_kwargs: payload)

    out = ingest.update(conn, triage_cfg)

    assert out == {
        "accounts": 1,
        "new_ids": 1,
        "ingested": 1,
        "cursor_resets": 0,
        "failed": 0,
    }

    triage_row = conn.execute(
        """
        SELECT stream_kind, subject_family, sender_domain, proposed_tier, applied_tier, novelty_score, signals_json
        FROM message_ingest_triage
        WHERE msg_id = ?
        """,
        ("m-triage",),
    ).fetchone()
    assert triage_row is not None
    assert triage_row[0] == "bulk"
    assert triage_row[1] == "weekly digest"
    assert triage_row[2] == "example.com"
    assert triage_row[3] in {"light", "minimal"}
    assert triage_row[4] == "full"
    assert float(triage_row[5]) == 1.0
    assert json.loads(str(triage_row[6]))["list_id"] is True

    reputation_row = conn.execute(
        """
        SELECT observation_count, full_count, light_count, minimal_count
        FROM message_stream_reputation
        """
    ).fetchone()
    assert reputation_row is not None
    assert int(reputation_row[0]) == 1
    assert int(reputation_row[1]) + int(reputation_row[2]) + int(reputation_row[3]) == 1


def test_update_enforce_mode_applies_light_and_defers_auto_index_candidates(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    triage_cfg = replace(app_cfg, ingest_triage=IngestTriageConfig(enabled=True, mode="enforce"))
    service = object()
    monkeypatch.setattr(ingest, "get_service", lambda *_args, **_kwargs: service)
    monkeypatch.setattr(ingest, "get_profile_history_id", lambda *_: 400)
    monkeypatch.setattr(
        ingest, "list_incremental_added_ids", lambda *_args, **_kwargs: ({"m-light"}, 450)
    )

    payload = gmail_message_payload(
        "m-light",
        history_id=450,
        from_addr="no-reply@example.com",
        to_addr="acct@example.com",
        subject="Weekly Digest 2026-04-17",
        body_text="Top stories for you",
        labels=["INBOX"],
    )
    payload["payload"]["headers"].extend(
        [
            {"name": "List-Id", "value": "digest.example.com"},
            {"name": "List-Unsubscribe", "value": "<https://example.com/unsub>"},
            {"name": "Precedence", "value": "bulk"},
        ]
    )
    monkeypatch.setattr(ingest, "fetch_full_message_payload", lambda *_args, **_kwargs: payload)

    out = ingest.update(conn, triage_cfg)
    assert out["ingested"] == 1
    assert message_exists(conn, "m-light") is True

    triage_row = conn.execute(
        """
        SELECT proposed_tier, applied_tier, enforcement_mode
        FROM message_ingest_triage
        WHERE msg_id = ?
        """,
        ("m-light",),
    ).fetchone()
    assert triage_row is not None
    assert triage_row[0] in {"light", "minimal"}
    assert triage_row[1] == "light"
    assert triage_row[2] == "enforce"

    assert count_pending_vector_updates(conn, triage_cfg, index_level="redacted") == 1
    assert (
        count_pending_vector_updates(
            conn,
            triage_cfg,
            index_level="redacted",
            skip_applied_light=True,
        )
        == 0
    )


def test_update_does_not_trigger_idle_backfill(conn, app_cfg, monkeypatch: pytest.MonkeyPatch):
    service = object()
    monkeypatch.setattr(ingest, "get_service", lambda *_args, **_kwargs: service)
    monkeypatch.setattr(ingest, "get_profile_history_id", lambda *_: 222)
    monkeypatch.setattr(
        ingest,
        "list_incremental_added_ids",
        lambda *_args, **_kwargs: (set(), 225),
    )

    def _should_not_be_called(*_args, **_kwargs):
        raise AssertionError("idle/historical message listing must not run in update")

    monkeypatch.setattr(ingest, "list_message_ids_paged", _should_not_be_called)

    out = ingest.update(conn, app_cfg)
    assert out["new_ids"] == 0
    assert out["ingested"] == 0
    assert out["failed"] == 0


def test_backfill_max_messages_is_global_and_round_robin_fair(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    multi_cfg = replace(
        app_cfg,
        accounts=[
            AccountConfig(
                name="a",
                email="a@example.com",
                credentials_file="creds-a.json",
                token_file="token-a.json",
            ),
            AccountConfig(
                name="b",
                email="b@example.com",
                credentials_file="creds-b.json",
                token_file="token-b.json",
            ),
        ],
    )

    services = {
        "token-a.json": "svc-a",
        "token-b.json": "svc-b",
    }

    monkeypatch.setattr(ingest, "get_service", lambda _creds, token, **_kwargs: services[token])

    listed_calls: list[tuple[str, int | None]] = []

    def _list_message_ids(service, query, max_messages=None):
        listed_calls.append((service, max_messages))
        if service == "svc-a":
            return ["a1", "a2", "a3"]
        return ["b1", "b2", "b3"]

    monkeypatch.setattr(ingest, "list_message_ids_paged", _list_message_ids)

    fetched_order: list[tuple[str, str]] = []

    def _fetch(service, msg_id):
        fetched_order.append((service, msg_id))
        return gmail_message_payload(msg_id, history_id=500)

    monkeypatch.setattr(ingest, "fetch_full_message_payload", _fetch)
    monkeypatch.setattr(
        ingest, "get_profile_history_id", lambda service: 901 if service == "svc-a" else 902
    )

    out = ingest.backfill(conn, multi_cfg, max_messages=4)

    assert out == {"accounts": 2, "ingested": 4, "skipped_existing": 0, "failed": 0}
    assert listed_calls == [("svc-a", 4), ("svc-b", 4)]
    assert fetched_order == [
        ("svc-a", "a1"),
        ("svc-b", "b1"),
        ("svc-a", "a2"),
        ("svc-b", "b2"),
    ]
    assert get_cursor(conn, "a@example.com", ingest.MAILBOX_SCOPE) == 901
    assert get_cursor(conn, "b@example.com", ingest.MAILBOX_SCOPE) == 902


def test_backfill_global_cap_still_skips_duplicate_msg_ids(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    multi_cfg = replace(
        app_cfg,
        accounts=[
            AccountConfig(
                name="a",
                email="a@example.com",
                credentials_file="creds-a.json",
                token_file="token-a.json",
            ),
            AccountConfig(
                name="b",
                email="b@example.com",
                credentials_file="creds-b.json",
                token_file="token-b.json",
            ),
        ],
    )

    services = {
        "token-a.json": "svc-a",
        "token-b.json": "svc-b",
    }

    monkeypatch.setattr(ingest, "get_service", lambda _creds, token, **_kwargs: services[token])

    def _list_message_ids(service, query, max_messages=None):
        assert max_messages == 4
        if service == "svc-a":
            return ["shared", "a-only"]
        return ["shared", "b-only"]

    monkeypatch.setattr(ingest, "list_message_ids_paged", _list_message_ids)
    monkeypatch.setattr(
        ingest,
        "fetch_full_message_payload",
        lambda _svc, mid: gmail_message_payload(mid, history_id=600),
    )
    monkeypatch.setattr(ingest, "get_profile_history_id", lambda *_: 999)

    out = ingest.backfill(conn, multi_cfg, max_messages=4)

    assert out == {"accounts": 2, "ingested": 3, "skipped_existing": 1, "failed": 0}


def test_repair_default_skips_historical_gmail_listing(conn, app_cfg, monkeypatch: pytest.MonkeyPatch):
    service = object()
    monkeypatch.setattr(ingest, "get_service", lambda *_args, **_kwargs: service)
    monkeypatch.setattr(ingest, "get_profile_history_id", lambda *_: 333)

    def _should_not_be_called(*_args, **_kwargs):
        raise AssertionError("historical listing must not run when repair backfill limit is 0")

    monkeypatch.setattr(ingest, "list_message_ids_paged", _should_not_be_called)

    out = ingest.repair(conn, app_cfg)

    assert out["backfill_limit"] == 0
    assert out["backfill_attempted_accounts"] == 0
    assert out["backfill_scanned"] == 0
    assert out["backfill_ingested"] == 0


def test_repair_backfill_ingests_bounded_missing_history(conn, app_cfg, monkeypatch: pytest.MonkeyPatch):
    cfg = replace(
        app_cfg,
        gmail_query="newer_than:30d (label:inbox OR label:sent)",
        gmail_idle_backfill_query=None,
    )
    service = object()

    monkeypatch.setattr(ingest, "get_service", lambda *_args, **_kwargs: service)
    monkeypatch.setattr(ingest, "get_profile_history_id", lambda *_: 333)

    upsert_message(
        conn,
        {
            "msg_id": "existing-id",
            "account_email": cfg.accounts[0].email,
            "thread_id": "t-existing",
            "date_iso": "2026-01-15",
            "internal_ts": 1736899200000,
            "from_addr": "from@example.com",
            "to_addr": "to@example.com",
            "subject": "existing",
            "snippet": "existing",
            "body_text": "existing",
            "labels": ["INBOX"],
            "history_id": 1,
        },
    )
    conn.commit()

    listed_queries: list[str] = []

    def _list_ids(_service, query, max_messages=None):
        listed_queries.append(query)
        assert max_messages == 20
        return ["existing-id", "older-1", "older-2", "older-3"]

    monkeypatch.setattr(ingest, "list_message_ids_paged", _list_ids)
    monkeypatch.setattr(
        ingest,
        "fetch_full_message_payload",
        lambda _svc, mid: gmail_message_payload(mid, history_id=333, labels=["INBOX"]),
    )

    out = ingest.repair(conn, cfg, backfill_limit=2)

    assert "newer_than:" not in listed_queries[0]
    assert "before:2025/01/16" in listed_queries[0]
    assert out["backfill_limit"] == 2
    assert out["backfill_scanned"] == 3
    assert out["backfill_candidates"] == 2
    assert out["backfill_ingested"] == 2
    assert out["backfill_skipped_existing"] == 1
    assert out["backfill_failed"] == 0
    assert out["interrupted"] is False


def test_repair_commits_partial_progress_on_keyboard_interrupt(
    conn,
    app_cfg,
    monkeypatch: pytest.MonkeyPatch,
):
    service = object()
    monkeypatch.setattr(ingest, "get_service", lambda *_args, **_kwargs: service)
    monkeypatch.setattr(ingest, "get_profile_history_id", lambda *_: 444)
    monkeypatch.setattr(
        ingest,
        "list_message_ids_paged",
        lambda *_args, **_kwargs: ["older-1", "older-2"],
    )

    def _fetch(_svc, msg_id):
        if msg_id == "older-2":
            raise KeyboardInterrupt("simulated Ctrl+C")
        return gmail_message_payload(msg_id, history_id=444, labels=["INBOX"])

    monkeypatch.setattr(ingest, "fetch_full_message_payload", _fetch)

    out = ingest.repair(conn, app_cfg, backfill_limit=2, commit_every_messages=1)

    assert out["interrupted"] is True
    assert out["backfill_ingested"] == 1
    assert message_exists(conn, "older-1") is True
