from __future__ import annotations

from inbox_vault.prompts import (
    ENRICH_SYSTEM_PROMPT,
    PROFILE_SYSTEM_PROMPT,
    REDACTION_SYSTEM_PROMPT,
    build_enrichment_messages,
    build_profile_messages,
    build_redaction_messages,
)


def test_enrichment_prompt_has_schema_contract_and_truncation_note():
    messages = build_enrichment_messages(
        subject="Invoice",
        snippet="Please review",
        body_text="A" * 500,
        from_addr="a@example.com",
        to_addr="b@example.com",
        date_iso="2026-03-01",
        body_max_chars=200,
    )
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == ENRICH_SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert "JSON schema contract:" in messages[1]["content"]
    assert "Body (truncated, max_chars=200):" in messages[1]["content"]
    assert '"importance"' in messages[1]["content"]


def test_profile_prompt_has_chunk_awareness_and_output_rules():
    messages = build_profile_messages(
        contact_email="alice@example.com",
        samples=[
            (
                "Launch",
                "Checklist",
                "Long body",
                "alice@example.com",
                "me@example.com",
                "2026-03-01",
            )
        ],
    )
    assert messages[0]["content"] == PROFILE_SYSTEM_PROMPT
    assert "Chunk-awareness note" in messages[1]["content"]
    assert "Context note:" in messages[1]["content"]
    assert "Output rules:" in messages[1]["content"]
    assert '"common_topics"' in messages[1]["content"]
    assert "Control: /no_think" not in messages[1]["content"]
    assert "source=direct" in messages[1]["content"]
    assert "content=Long body" in messages[1]["content"]
    assert "snippet=Checklist" in messages[1]["content"]
    assert "body_excerpt=Long body" in messages[1]["content"]


def test_profile_prompt_bounds_excerpts_with_sample_chars():
    snippet = "s" * 200
    body = "b" * 200
    messages = build_profile_messages(
        contact_email="alice@example.com",
        samples=[
            (
                "Launch",
                snippet,
                body,
                "alice@example.com",
                "me@example.com",
                "2026-03-01",
            )
        ],
        sample_chars=120,
    )
    prompt = messages[1]["content"]
    assert f"content={'b' * 120}" in prompt
    assert f"snippet={'s' * 120}" in prompt
    assert f"body_excerpt={'b' * 120}" in prompt
    assert f"content={body}" not in prompt


def test_profile_prompt_respects_prompt_budget_chars():
    samples = [
        {
            "subject": f"Lease update {i}",
            "snippet": "s" * 180,
            "body_text": "b" * 220,
            "from_addr": "alice@example.com",
            "to_addr": "me@example.com",
            "date_iso": "2026-03-10",
            "thread_id": f"thread-{i}",
            "context_source": "thread",
        }
        for i in range(5)
    ]

    messages = build_profile_messages(
        contact_email="alice@example.com",
        samples=samples,
        max_samples=5,
        sample_chars=220,
        prompt_budget_chars=1800,
    )
    prompt = messages[1]["content"]
    assert len(prompt) <= 1800
    assert "Samples included: " in prompt


def test_profile_prompt_renders_thread_aware_sample_metadata():
    messages = build_profile_messages(
        contact_email="contact@example.com",
        samples=[
            {
                "subject": "Lease update",
                "snippet": "Need contractor availability",
                "body_text": "Thread follow-up details",
                "from_addr": "assistant@agency.example",
                "to_addr": "operator@example.com",
                "date_iso": "2026-03-10",
                "thread_id": "thread-landlord-1",
                "context_source": "thread",
            }
        ],
        sample_chars=160,
    )
    prompt = messages[1]["content"]
    assert "source=thread" in prompt
    assert "thread_id=thread-landlord-1" in prompt
    assert "contact=contact@example.com" in prompt


def test_redaction_prompt_includes_chunk_position_and_boundaries():
    messages = build_redaction_messages(
        chunk_text="Contact bob@example.com",
        profile="finance",
        instruction="Mask all names",
        chunk_index=2,
        chunk_total=5,
    )
    assert messages[0]["content"] == REDACTION_SYSTEM_PROMPT
    assert "Chunk position: 2/5" in messages[1]["content"]
    assert "---CHUNK---" in messages[1]["content"]
    assert "---END CHUNK---" in messages[1]["content"]
