from __future__ import annotations

from inbox_vault.config import LLMConfig
from inbox_vault.db import (
    fetch_redaction_entries,
    unredact_with_scope,
    upsert_redaction_entries,
)
from inbox_vault.redaction import PersistentRedactionMap, redact_with_persistent_map


def test_hybrid_mode_runs_fresh_detection_for_document_and_each_chunk(monkeypatch):
    cfg = LLMConfig(enabled=True, endpoint="http://localhost:8080", model="local", timeout_seconds=1.0)
    table = PersistentRedactionMap()
    calls: list[str] = []

    def fake_detect(_text: str, **kwargs):
        calls.append(kwargs["source"])
        return []

    monkeypatch.setattr("inbox_vault.redaction._model_detect_candidates", fake_detect)

    redact_with_persistent_map(
        "Body: one two three four",
        chunks=["Body: one two", "Body: three four"],
        mode="hybrid",
        llm_cfg=cfg,
        profile="standard",
        instruction="",
        table=table,
    )

    assert calls == ["llm_document", "llm_chunk", "llm_chunk"]


def test_learned_rule_reuse_keeps_placeholder_stable():
    table = PersistentRedactionMap.from_rows(
        [("EMAIL", "<REDACTED_EMAIL_A>", "bob@example.com", "bob@example.com")]
    )

    result = redact_with_persistent_map(
        "Reach bob@example.com now",
        chunks=["Reach bob@example.com now"],
        mode="regex",
        llm_cfg=None,
        profile="standard",
        instruction="",
        table=table,
    )

    assert "<REDACTED_EMAIL_A>" in result.source_text_redacted
    assert result.inserted_entries == []


def test_boundary_spanning_value_replaced_with_full_placeholder():
    table = PersistentRedactionMap.from_rows(
        [("EMAIL", "<REDACTED_EMAIL_A>", "alice@example.com", "alice@example.com")]
    )

    result = redact_with_persistent_map(
        "Contact alice@example.com now",
        chunks=["Contact alice@exam", "ple.com now"],
        mode="regex",
        llm_cfg=None,
        profile="standard",
        instruction="",
        table=table,
    )

    assert "<REDACTED_EMAIL_A>" in result.chunk_text_redacted[0]
    assert "<REDACTED_EMAIL_A>" in result.chunk_text_redacted[1]


def test_unredaction_round_trip_with_db(conn):
    entries = [
        {
            "key_name": "EMAIL",
            "placeholder": "<REDACTED_EMAIL_A>",
            "value_norm": "amy@example.com",
            "original_value": "amy@example.com",
            "source_mode": "llm_chunk",
        }
    ]
    upsert_redaction_entries(conn, scope_type="account", scope_id="acct@example.com", entries=entries)
    conn.commit()

    rows = fetch_redaction_entries(conn, scope_type="account", scope_id="acct@example.com")
    assert rows[0][1] == "<REDACTED_EMAIL_A>"

    restored = unredact_with_scope(
        conn,
        scope_type="account",
        scope_id="acct@example.com",
        text="Please email <REDACTED_EMAIL_A>",
    )
    assert restored == "Please email amy@example.com"
