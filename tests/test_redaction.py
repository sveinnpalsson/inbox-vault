from __future__ import annotations

from inbox_vault.config import LLMConfig
from inbox_vault.redaction import (
    is_redaction_value_allowed,
    model_redact_text,
    redact_text,
    regex_redact_text,
)


def test_regex_redaction_masks_common_tokens():
    text = "Email bob@example.com call 212-555-1234 url https://example.com account acct-99887766"
    out = regex_redact_text(text)
    assert "[REDACTED_EMAIL]" in out
    assert "[REDACTED_PHONE]" in out
    assert "[REDACTED_URL]" in out
    assert "[REDACTED_ACCOUNT]" in out


def test_model_mode_falls_back_to_regex_on_error(monkeypatch):
    cfg = LLMConfig(
        enabled=True, endpoint="http://localhost:8080", model="local", timeout_seconds=1.0
    )

    def fail_chunk(*_args, **_kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr("inbox_vault.redaction._model_redact_chunk", fail_chunk)

    out = redact_text("Reach me at bob@example.com", mode="model", llm_cfg=cfg)
    assert "[REDACTED_EMAIL]" in out


def test_hybrid_mode_applies_regex_after_model(monkeypatch):
    cfg = LLMConfig(
        enabled=True, endpoint="http://localhost:8080", model="local", timeout_seconds=1.0
    )

    monkeypatch.setattr(
        "inbox_vault.redaction._model_redact_chunk",
        lambda chunk, **_kwargs: f"MODEL::{chunk}",
    )

    out = redact_text("URL https://internal.local", mode="hybrid", llm_cfg=cfg)
    assert "MODEL::" in out
    assert "[REDACTED_URL]" in out


def test_model_redaction_uses_chunking(monkeypatch):
    cfg = LLMConfig(
        enabled=True, endpoint="http://localhost:8080", model="local", timeout_seconds=1.0
    )
    calls: list[tuple[str, int, int]] = []

    def fake_chunk(chunk: str, **kwargs):
        calls.append((chunk, kwargs["chunk_index"], kwargs["chunk_total"]))
        return f"[{len(chunk)}]"

    monkeypatch.setattr("inbox_vault.redaction._model_redact_chunk", fake_chunk)

    out = model_redact_text("abcdefghij", llm_cfg=cfg, profile="std", instruction="", chunk_chars=4)
    assert calls == [("abcd", 1, 3), ("efgh", 2, 3), ("ij", 3, 3)]
    assert out == "[4][4][2]"


def test_redaction_value_validator_rejects_common_false_positives():
    assert is_redaction_value_allowed("ACCOUNT", "24") is False
    assert is_redaction_value_allowed("ADDRESS", "CA") is False
    assert is_redaction_value_allowed("PERSON", "LAST NAME") is False
    assert is_redaction_value_allowed("PERSON", "name") is False


def test_redaction_value_validator_rejects_custom_and_accepts_valid_entities():
    assert is_redaction_value_allowed("CUSTOM", "Project Delta") is False
    assert is_redaction_value_allowed("CUSTOM", "amy_doe") is True
    assert is_redaction_value_allowed("CUSTOM", "neo-43CU") is True
    assert is_redaction_value_allowed("CUSTOM", "Agent1") is False
    assert is_redaction_value_allowed("EMAIL", "alice@example.com") is True
    assert is_redaction_value_allowed("PHONE", "+1 (617) 555-1212") is True
    assert is_redaction_value_allowed("PERSON", "Alice Johnson") is True
    assert is_redaction_value_allowed("PERSON", "Tempobono", source_text='Last Name: "Tempobono"') is True
    assert is_redaction_value_allowed("ADDRESS", "123 Main Street") is True
