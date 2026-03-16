from __future__ import annotations

from inbox_vault.redaction import deterministic_map_redact_text
from inbox_vault.redaction_map import (
    DeterministicRedactionMap,
    apply_deterministic_redaction,
)


def test_deterministic_map_redacts_supported_entities():
    text = (
        "Email bob@example.com and alice@example.com. "
        "Call +1 (212) 555-1234. "
        "Visit https://example.com/path. "
        "Account acct-99887766 and 123456789012."
    )

    out, redaction_map = apply_deterministic_redaction(text)

    assert "<REDACTED_EMAIL_A>" in out
    assert "<REDACTED_EMAIL_B>" in out
    assert "<REDACTED_PHONE_A>" in out
    assert "<REDACTED_URL_A>" in out
    assert "<REDACTED_ACCOUNT_A>" in out
    assert "<REDACTED_ACCOUNT_B>" in out
    assert "bob@example.com" not in out
    assert redaction_map.placeholder_to_entity["<REDACTED_EMAIL_A>"] == "bob@example.com"


def test_deterministic_map_preserves_coreference_across_chunks():
    mapping = DeterministicRedactionMap(scope_id="thread-1")

    chunk_1, mapping = deterministic_map_redact_text(
        "Ping bob@example.com or +1 212-555-1234", redaction_map=mapping
    )
    chunk_2, mapping = deterministic_map_redact_text(
        "Forward to BOB@example.com and call 2125551234", redaction_map=mapping
    )

    assert "<REDACTED_EMAIL_A>" in chunk_1
    assert "<REDACTED_EMAIL_A>" in chunk_2
    assert "<REDACTED_PHONE_A>" in chunk_1
    assert "<REDACTED_PHONE_A>" in chunk_2


def test_deterministic_map_round_trip_serialization_and_unredaction():
    original = "Open https://example.com, email bob@example.com"
    redacted, mapping = apply_deterministic_redaction(original, scope_id="account-main")

    payload = mapping.to_dict()
    restored_map = DeterministicRedactionMap.from_dict(payload)

    assert restored_map.scope_id == "account-main"
    assert restored_map.unredact(redacted) == original
