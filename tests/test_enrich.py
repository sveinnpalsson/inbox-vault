from __future__ import annotations

import pytest
import requests

from inbox_vault.db import upsert_message
from inbox_vault.enrich import enrich_pending
from inbox_vault.json_contracts import validate_enrich_contract


def _seed_message(
    conn,
    msg_id: str,
    *,
    subject: str = "Synthetic subject",
    snippet: str = "Synthetic snippet",
    body_text: str = "Synthetic body text",
    from_addr: str = "sender@example.com",
    to_addr: str = "recipient@example.com",
):
    upsert_message(
        conn,
        {
            "msg_id": msg_id,
            "account_email": "acct@example.com",
            "thread_id": f"t-{msg_id}",
            "date_iso": "2023-12-04",
            "internal_ts": 1701684900000,
            "from_addr": from_addr,
            "to_addr": to_addr,
            "subject": subject,
            "snippet": snippet,
            "body_text": body_text,
            "labels": ["INBOX"],
            "history_id": 10,
        },
    )


def test_enrich_pending_honors_llm_disable(conn, app_cfg):
    _seed_message(conn, "m1")
    app_cfg.llm.enabled = False

    assert enrich_pending(conn, app_cfg) == 0


def test_enrich_pending_falls_back_to_heuristic_when_llm_missing_json(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    _seed_message(
        conn,
        "m1",
        subject="Rent reminder",
        snippet="Lease renewal due soon",
        body_text="Please reply to your landlord before due date",
    )

    monkeypatch.setattr("inbox_vault.enrich.chat_json", lambda *_args, **_kwargs: None)

    diagnostics: dict[str, int] = {}
    updated = enrich_pending(conn, app_cfg, diagnostics=diagnostics)

    assert updated == 1
    row = conn.execute(
        "SELECT category, action, importance, model FROM message_enrichment WHERE msg_id='m1'"
    ).fetchone()
    assert row == ("housing", "reply", 9, "heuristic-fallback")
    assert diagnostics == {
        "attempted": 1,
        "succeeded": 1,
        "http_failed": 0,
        "parse_failed": 2,
        "contract_failed": 0,
        "repair_attempted": 0,
        "repair_succeeded": 0,
        "fallback_used": 1,
    }


def test_enrich_pending_uses_prompt_contract_in_messages(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    _seed_message(conn, "m1")
    captured: dict[str, object] = {}

    def _chat(_cfg, messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return {"category": "work", "importance": 6, "action": "review", "summary": "ok"}

    monkeypatch.setattr("inbox_vault.enrich.chat_json", _chat)

    updated = enrich_pending(conn, app_cfg)
    assert updated == 1
    user_prompt = captured["messages"][1]["content"]
    assert "JSON schema contract:" in user_prompt
    assert "Output rules:" in user_prompt
    assert captured["kwargs"]["temperature"] == 0.0


def test_enrich_pending_populates_diagnostics_with_compact_retry(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    _seed_message(conn, "m1", subject="Normal work update")
    _seed_message(conn, "m2", subject="Lease paperwork", snippet="Tenant notice")
    _seed_message(conn, "m3", subject="Billing notice")

    state = {
        "m1": ["ok"],
        "m2": ["none", "ok"],
        "m3": ["http", "none"],
    }

    def _chat(_cfg, messages, **_kwargs):
        prompt = messages[1]["content"]
        compact = "compact mode" in prompt
        if "Normal work update" in prompt:
            key = "m1"
        elif "Lease paperwork" in prompt:
            key = "m2"
        else:
            key = "m3"

        outcome = state[key].pop(0)
        if outcome == "http":
            raise requests.HTTPError("status=503")
        if outcome == "none":
            return None

        if compact:
            return {
                "category": "housing",
                "importance": 8,
                "action": "reply",
                "summary": "compact retry success",
            }
        return {
            "category": "work",
            "importance": 5,
            "action": "review",
            "summary": "primary success",
        }

    monkeypatch.setattr("inbox_vault.enrich.chat_json", _chat)

    diagnostics: dict[str, int] = {}
    updated = enrich_pending(conn, app_cfg, diagnostics=diagnostics)

    assert updated == 3
    assert diagnostics == {
        "attempted": 3,
        "succeeded": 3,
        "http_failed": 1,
        "parse_failed": 2,
        "contract_failed": 0,
        "repair_attempted": 0,
        "repair_succeeded": 0,
        "fallback_used": 2,
    }

    rows = conn.execute("SELECT msg_id, model FROM message_enrichment ORDER BY msg_id").fetchall()
    assert rows == [
        ("m1", app_cfg.llm.model),
        ("m2", app_cfg.llm.model),
        ("m3", "heuristic-fallback"),
    ]


def test_enrich_pending_repairs_empty_object(conn, app_cfg, monkeypatch: pytest.MonkeyPatch):
    _seed_message(conn, "repair-empty", subject="Project kickoff", snippet="Please review agenda")

    calls = {"n": 0}

    def _chat(_cfg, messages, **_kwargs):
        calls["n"] += 1
        if "Original output:" in messages[1]["content"]:
            return {
                "category": "work",
                "importance": 6,
                "action": "review",
                "summary": "Repaired payload",
            }
        return {}

    monkeypatch.setattr("inbox_vault.enrich.chat_json", _chat)

    diagnostics: dict[str, int] = {}
    updated = enrich_pending(conn, app_cfg, diagnostics=diagnostics)

    assert updated == 1
    row = conn.execute(
        "SELECT category, importance, action, summary, model FROM message_enrichment WHERE msg_id='repair-empty'"
    ).fetchone()
    assert row == ("work", 6, "review", "Repaired payload", app_cfg.llm.model)
    assert diagnostics["contract_failed"] >= 1
    assert diagnostics["repair_attempted"] == 1
    assert diagnostics["repair_succeeded"] == 1
    assert diagnostics["fallback_used"] == 1


def test_enrich_pending_repairs_missing_keys(conn, app_cfg, monkeypatch: pytest.MonkeyPatch):
    _seed_message(conn, "repair-missing", subject="Vendor invoice", snippet="Payment due")

    def _chat(_cfg, messages, **_kwargs):
        if "Original output:" in messages[1]["content"]:
            return {
                "category": "billing",
                "importance": 7,
                "action": "review",
                "summary": "Invoice follow-up",
            }
        return {"category": "billing", "importance": 7}

    monkeypatch.setattr("inbox_vault.enrich.chat_json", _chat)

    updated = enrich_pending(conn, app_cfg)
    assert updated == 1

    payload = conn.execute(
        "SELECT category, importance, action, summary FROM message_enrichment WHERE msg_id='repair-missing'"
    ).fetchone()
    out = {
        "category": payload[0],
        "importance": payload[1],
        "action": payload[2],
        "summary": payload[3],
    }
    ok, _ = validate_enrich_contract(out)
    assert ok is True


def test_enrich_pending_uses_fallback_fill_when_repair_fails(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    _seed_message(conn, "repair-fail", subject="Lease status", snippet="Landlord follow-up")

    def _chat(_cfg, messages, **_kwargs):
        if "Original output:" in messages[1]["content"]:
            return {"category": "", "importance": "high"}
        return {"category": "", "importance": "high"}

    monkeypatch.setattr("inbox_vault.enrich.chat_json", _chat)

    diagnostics: dict[str, int] = {}
    updated = enrich_pending(conn, app_cfg, diagnostics=diagnostics)
    assert updated == 1

    payload = conn.execute(
        "SELECT category, importance, action, summary FROM message_enrichment WHERE msg_id='repair-fail'"
    ).fetchone()
    out = {
        "category": payload[0],
        "importance": payload[1],
        "action": payload[2],
        "summary": payload[3],
    }
    ok, _ = validate_enrich_contract(out)
    assert ok is True
    assert diagnostics["repair_attempted"] == 2
    assert diagnostics["repair_succeeded"] == 0
    assert diagnostics["fallback_used"] == 1


def test_enrich_pending_include_degraded_retries_heuristic_fallback(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    _seed_message(
        conn,
        "fallback-repair",
        subject="Lease renewal",
        snippet="Please sign and return",
    )
    conn.execute(
        """
        INSERT INTO message_enrichment (msg_id, category, importance, action, summary, model, enriched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "fallback-repair",
            "housing",
            9,
            "reply",
            "heuristic summary",
            "heuristic-fallback",
            "2026-04-06T20:00:00+00:00",
        ),
    )
    conn.commit()

    monkeypatch.setattr(
        "inbox_vault.enrich.chat_json",
        lambda *_args, **_kwargs: {
            "category": "housing",
            "importance": 10,
            "action": "reply",
            "summary": "LLM repair succeeded",
        },
    )

    diagnostics: dict[str, int] = {}
    updated = enrich_pending(conn, app_cfg, diagnostics=diagnostics, include_degraded=True)

    assert updated == 1
    row = conn.execute(
        """
        SELECT importance, summary, model
        FROM message_enrichment
        WHERE msg_id='fallback-repair'
        """
    ).fetchone()
    assert row == (10, "LLM repair succeeded", app_cfg.llm.model)
    assert diagnostics["attempted"] == 1
    assert diagnostics["fallback_used"] == 0
