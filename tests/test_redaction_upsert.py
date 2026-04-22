from __future__ import annotations

from inbox_vault import db


def test_upsert_redaction_entries_updates_existing_placeholder_row(tmp_path):
    conn = db.get_conn(tmp_path / "test.db", "pw")

    first = {
        "key_name": "email",
        "placeholder": "[EMAIL_1]",
        "value_norm": "alice@example.com",
        "original_value": "alice@example.com",
        "source_mode": "llm",
        "policy_version": "v1",
        "status": "active",
        "validator_name": "typed_v1",
        "detector_sources": "llm",
        "modality": "text",
        "source_field": "body",
    }
    second = {
        "key_name": "email",
        "placeholder": "[EMAIL_1]",
        "value_norm": "alice+alias@example.com",
        "original_value": "alice+alias@example.com",
        "source_mode": "llm",
        "policy_version": "v1",
        "status": "active",
        "validator_name": "typed_v1",
        "detector_sources": "llm",
        "modality": "text",
        "source_field": "body",
    }

    retries = db.upsert_redaction_entries(
        conn,
        scope_type="account",
        scope_id="acct@example.com",
        entries=[first],
    )
    assert retries == 0

    retries = db.upsert_redaction_entries(
        conn,
        scope_type="account",
        scope_id="acct@example.com",
        entries=[second],
    )
    assert retries == 0

    rows = conn.execute(
        "SELECT key_name, placeholder, value_norm, original_value, hit_count FROM redaction_entries WHERE scope_type = ? AND scope_id = ?",
        ("account", "acct@example.com"),
    ).fetchall()
    assert rows == [
        ("email", "[EMAIL_1]", "alice+alias@example.com", "alice+alias@example.com", 2)
    ]
