from __future__ import annotations

from dataclasses import replace

import pytest
from googleapiclient.errors import HttpError

from inbox_vault import ingest
from inbox_vault.config import AccountConfig
from inbox_vault.db import get_cursor, message_exists, upsert_message
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
