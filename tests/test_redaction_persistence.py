from __future__ import annotations

from inbox_vault.config import LLMConfig
from inbox_vault.db import (
    fetch_redaction_entries,
    prune_invalid_redaction_entries,
    unredact_with_scope,
    upsert_redaction_entries,
)
from inbox_vault.redaction import (
    PersistentRedactionMap,
    is_persistent_redaction_value_allowed,
    is_redaction_value_allowed,
    redact_with_persistent_map,
)


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
    assert result.persisted_entries == []


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


def test_invalid_redaction_entries_are_not_persisted(conn):
    upsert_redaction_entries(
        conn,
        scope_type="account",
        scope_id="acct@example.com",
        entries=[
            {
                "key_name": "PERSON",
                "placeholder": "<REDACTED_PERSON_A>",
                "value_norm": "name",
                "original_value": "name",
                "source_mode": "llm_chunk",
            },
            {
                "key_name": "EMAIL",
                "placeholder": "<REDACTED_EMAIL_A>",
                "value_norm": "amy@example.com",
                "original_value": "amy@example.com",
                "source_mode": "llm_chunk",
            },
        ],
    )
    conn.commit()

    rows = fetch_redaction_entries(conn, scope_type="account", scope_id="acct@example.com")
    assert rows == [("EMAIL", "<REDACTED_EMAIL_A>", "amy@example.com", "amy@example.com")]


def test_redaction_value_filters_reject_common_false_positives():
    assert not is_redaction_value_allowed("ACCOUNT", "24")
    assert not is_redaction_value_allowed("ACCOUNT", "20")
    assert not is_redaction_value_allowed("ADDRESS", "CA")
    assert not is_redaction_value_allowed("PERSON", "LAST NAME")
    assert not is_redaction_value_allowed("PERSON", "name")
    assert not is_redaction_value_allowed("PERSON", "Two individuals")
    assert not is_redaction_value_allowed("PERSON", "Student E")
    assert not is_redaction_value_allowed("CUSTOM", "employee id")
    assert not is_redaction_value_allowed("CUSTOM", "username")
    assert not is_redaction_value_allowed("CUSTOM", "abcdef")
    assert not is_redaction_value_allowed("CUSTOM", "Agent1")

    assert is_redaction_value_allowed("EMAIL", "amy@example.com")
    assert is_redaction_value_allowed("PHONE", "617-555-1212")
    assert is_redaction_value_allowed("PERSON", "Amy Doe")
    assert is_redaction_value_allowed(
        "PERSON",
        "Tempobono",
        source_text='Last Name: "Tempobono"',
    )
    assert is_redaction_value_allowed(
        "PERSON",
        "Cesnulis",
        source_text='expert veterinarians ([Cesnulis], [Kakiashvili], [Ponomarov], [Sobhan])',
    )
    assert is_redaction_value_allowed("ADDRESS", "123 Main St")
    assert is_redaction_value_allowed("ADDRESS", "ENG", source_text='State: "ENG"')
    assert is_redaction_value_allowed("ADDRESS", "58", source_text='Building Number: "58"')
    assert is_redaction_value_allowed("CUSTOM", "@amy.doe-77")
    assert is_redaction_value_allowed("CUSTOM", "amy_doe")
    assert is_redaction_value_allowed("CUSTOM", "amy.doe-77")
    assert is_redaction_value_allowed("CUSTOM", "neo-43CU")
    assert is_redaction_value_allowed("CUSTOM", "cust8472")
    assert is_redaction_value_allowed("CUSTOM", "43CU")


def test_persistent_redaction_filters_stay_stricter_than_runtime():
    assert is_redaction_value_allowed("ADDRESS", "Boston", source_text='City: "Boston"')
    assert is_redaction_value_allowed("ADDRESS", "58", source_text='Building Number: "58"')
    assert is_redaction_value_allowed("ADDRESS", "ENG", source_text='State: "ENG"')
    assert is_redaction_value_allowed("PERSON", "Tempobono", source_text='Last Name: "Tempobono"')
    assert not is_persistent_redaction_value_allowed("ADDRESS", "Boston")
    assert not is_persistent_redaction_value_allowed("ADDRESS", "58")
    assert not is_persistent_redaction_value_allowed("ADDRESS", "ENG")
    assert is_persistent_redaction_value_allowed("PERSON", "Tempobono")
    assert is_persistent_redaction_value_allowed("ADDRESS", "123 Main St")
    assert is_persistent_redaction_value_allowed("ADDRESS", "58 Kings Lane, Norwich")


def test_redaction_map_ignores_invalid_persisted_entries():
    table = PersistentRedactionMap.from_rows(
        [
            ("ADDRESS", "<REDACTED_ADDRESS_A>", "ca", "CA"),
            ("PERSON", "<REDACTED_PERSON_A>", "last name", "LAST NAME"),
            ("EMAIL", "<REDACTED_EMAIL_A>", "amy@example.com", "amy@example.com"),
        ]
    )

    assert "<REDACTED_EMAIL_A>" in table.placeholder_to_value
    assert "<REDACTED_ADDRESS_A>" not in table.placeholder_to_value
    assert "<REDACTED_PERSON_A>" not in table.placeholder_to_value


def test_redaction_map_keeps_valid_single_token_person_entries():
    table = PersistentRedactionMap.from_rows(
        [("PERSON", "<REDACTED_PERSON_A>", "tempobono", "Tempobono")]
    )

    assert table.placeholder_to_value["<REDACTED_PERSON_A>"] == "Tempobono"


def test_hybrid_redaction_expands_composed_address_model_candidates(monkeypatch):
    from inbox_vault.redaction import RedactionCandidate

    def fake_model_detect_candidates(text: str, **kwargs):
        return [
            RedactionCandidate(
                key_name="ADDRESS",
                value="58 Kings Lane, Norwich, ENG, NR1 3PS",
                source=kwargs["source"],
            ),
        ]

    monkeypatch.setattr("inbox_vault.redaction._model_detect_candidates", fake_model_detect_candidates)
    table = PersistentRedactionMap()
    text = (
        'Participant Information:\n'
        '- Building Number: "58"\n'
        '- Street: "Kings Lane"\n'
        '- City: "Norwich"\n'
        '- State: "ENG"\n'
        '- Postcode: "NR1 3PS"\n'
    )
    out = redact_with_persistent_map(
        text,
        chunks=[text],
        mode="hybrid",
        llm_cfg=LLMConfig(enabled=True, endpoint="http://localhost:8080", model="local", timeout_seconds=1.0),
        profile="standard",
        instruction="",
        table=table,
    )

    originals = {entry["original_value"] for entry in out.inserted_entries}
    assert {"58", "Kings Lane", "Norwich", "ENG", "NR1 3PS"}.issubset(originals)
    assert [(entry["key_name"], entry["original_value"]) for entry in out.persisted_entries] == [
        ("ADDRESS", "Kings Lane"),
    ]


def test_hybrid_redaction_ignores_generic_role_labels_from_model(monkeypatch):
    from inbox_vault.redaction import RedactionCandidate

    def fake_model_detect_candidates(text: str, **kwargs):
        return [
            RedactionCandidate(key_name="PERSON", value="Student E", source=kwargs["source"]),
            RedactionCandidate(key_name="CUSTOM", value="Agent1", source=kwargs["source"]),
            RedactionCandidate(key_name="PERSON", value="Amy Doe", source=kwargs["source"]),
        ]

    monkeypatch.setattr("inbox_vault.redaction._model_detect_candidates", fake_model_detect_candidates)
    table = PersistentRedactionMap()
    out = redact_with_persistent_map(
        "Student E met Agent1 before Amy Doe joined.",
        chunks=["Student E met Agent1 before Amy Doe joined."],
        mode="hybrid",
        llm_cfg=LLMConfig(enabled=True, endpoint="http://localhost:8080", model="local", timeout_seconds=1.0),
        profile="standard",
        instruction="",
        table=table,
    )

    assert [(entry["key_name"], entry["original_value"]) for entry in out.inserted_entries] == [
        ("PERSON", "Amy Doe"),
    ]


def test_hybrid_redaction_accepts_single_token_person_list_context(monkeypatch):
    from inbox_vault.redaction import RedactionCandidate

    text = (
        "Proposal Details:\n"
        "1. **Animal Welfare Taskforce:** Establish a dedicated taskforce comprising of expert veterinarians "
        "([Cesnulis], [Kakiashvili], [Ponomarov], [Sobhan]) to oversee the protection and care of animals."
    )

    def fake_model_detect_candidates(chunk: str, **kwargs):
        assert chunk == text
        return [
            RedactionCandidate(key_name="PERSON", value="Cesnulis", source=kwargs["source"]),
            RedactionCandidate(key_name="PERSON", value="Kakiashvili", source=kwargs["source"]),
            RedactionCandidate(key_name="PERSON", value="Ponomarov", source=kwargs["source"]),
            RedactionCandidate(key_name="PERSON", value="Sobhan", source=kwargs["source"]),
        ]

    monkeypatch.setattr("inbox_vault.redaction._model_detect_candidates", fake_model_detect_candidates)
    table = PersistentRedactionMap()
    out = redact_with_persistent_map(
        text,
        chunks=[text],
        mode="hybrid",
        llm_cfg=LLMConfig(enabled=True, endpoint="http://localhost:8080", model="local", timeout_seconds=1.0),
        profile="standard",
        instruction="",
        table=table,
    )

    assert len(out.inserted_entries) == 4
    assert out.chunk_text_redacted[0].count("<REDACTED_PERSON_") == 4


def test_hybrid_redaction_remaps_person_handle_to_custom(monkeypatch):
    from inbox_vault.redaction import RedactionCandidate

    def fake_model_detect_candidates(text: str, **kwargs):
        return [
            RedactionCandidate(key_name="PERSON", value="neo-43CU", source=kwargs["source"]),
        ]

    monkeypatch.setattr("inbox_vault.redaction._model_detect_candidates", fake_model_detect_candidates)
    table = PersistentRedactionMap()
    out = redact_with_persistent_map(
        "User neo-43CU joined the private room.",
        chunks=["User neo-43CU joined the private room."],
        mode="hybrid",
        llm_cfg=LLMConfig(enabled=True, endpoint="http://localhost:8080", model="local", timeout_seconds=1.0),
        profile="standard",
        instruction="",
        table=table,
    )

    assert out.inserted_entries == [
        {
            "key_name": "CUSTOM",
            "placeholder": "<REDACTED_CUSTOM_A>",
            "value_norm": "neo-43cu",
            "original_value": "neo-43CU",
            "source_mode": "llm_document",
        }
    ]


def test_hybrid_redaction_remaps_phone_like_value_to_account_in_account_field(monkeypatch):
    from inbox_vault.redaction import RedactionCandidate

    text = 'Social Security Number: 940-965-2328\nTelephone Number: 617-555-1212\n'

    def fake_model_detect_candidates(chunk: str, **kwargs):
        assert chunk == text
        return [
            RedactionCandidate(key_name="PHONE", value="940-965-2328", source=kwargs["source"]),
            RedactionCandidate(key_name="PHONE", value="617-555-1212", source=kwargs["source"]),
        ]

    monkeypatch.setattr("inbox_vault.redaction._model_detect_candidates", fake_model_detect_candidates)
    table = PersistentRedactionMap()
    out = redact_with_persistent_map(
        text,
        chunks=[text],
        mode="hybrid",
        llm_cfg=LLMConfig(enabled=True, endpoint="http://localhost:8080", model="local", timeout_seconds=1.0),
        profile="standard",
        instruction="",
        table=table,
    )

    seen = [(entry["key_name"], entry["original_value"]) for entry in out.inserted_entries]
    assert ("ACCOUNT", "940-965-2328") in seen
    assert ("PHONE", "617-555-1212") in seen
    assert [(entry["key_name"], entry["original_value"]) for entry in out.persisted_entries] == [
        ("ACCOUNT", "940-965-2328"),
        ("PHONE", "617-555-1212"),
    ]


def test_prune_invalid_redaction_entries_marks_rejected(conn):
    conn.execute(
        """
        INSERT INTO redaction_entries (
          scope_type, scope_id, key_name, placeholder, value_norm, original_value, source_mode,
          policy_version, status, validator_name, detector_sources, modality, source_field,
          first_seen_at, last_seen_at, hit_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), ?)
        """,
        (
            "account",
            "acct@example.com",
            "ADDRESS",
            "<REDACTED_ADDRESS_A>",
            "ca",
            "CA",
            "llm_chunk",
            "",
            "active",
            "",
            "",
            "email",
            "body",
            1,
        ),
    )
    conn.commit()

    pruned = prune_invalid_redaction_entries(
        conn, scope_type="account", scope_id="acct@example.com"
    )
    conn.commit()

    assert pruned == 1
    row = conn.execute(
        "SELECT status FROM redaction_entries WHERE placeholder = ?",
        ("<REDACTED_ADDRESS_A>",),
    ).fetchone()
    assert row[0] == "rejected"
