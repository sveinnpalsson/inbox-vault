from __future__ import annotations

import json

import pytest
import requests

from inbox_vault.db import upsert_contact_profile, upsert_contact_seen, upsert_message
from inbox_vault.json_contracts import validate_profile_contract
from inbox_vault.profiles import build_profile_context_samples, build_profiles


def _seed_contact(
    conn,
    *,
    email: str,
    display_name: str,
    msg_id: str,
    subject: str,
    snippet: str,
    body_text: str,
    ts: int,
    from_addr: str,
    to_addr: str,
    thread_id: str | None = None,
):
    upsert_contact_seen(conn, email, display_name)
    upsert_message(
        conn,
        {
            "msg_id": msg_id,
            "account_email": "acct@example.com",
            "thread_id": thread_id or f"t-{msg_id}",
            "date_iso": "2023-12-04",
            "internal_ts": ts,
            "from_addr": from_addr,
            "to_addr": to_addr,
            "subject": subject,
            "snippet": snippet,
            "body_text": body_text,
            "labels": ["INBOX"],
            "history_id": 10,
        },
    )


def _seed_quick_contact_data(conn):
    _seed_contact(
        conn,
        email="alice@example.com",
        display_name="Alice",
        msg_id="m1",
        subject="Launch checklist ready",
        snippet="Review launch plan",
        body_text="Let's review launch details.",
        ts=1701684900000,
        from_addr="alice@example.com",
        to_addr="acct@example.com",
    )


def _seed_deep_bidirectional_contact(
    conn, *, email: str, display_name: str, prefix: str, total_messages: int = 10
):
    for idx in range(total_messages):
        from_addr = email if idx % 2 == 0 else "acct@example.com"
        to_addr = "acct@example.com" if idx % 2 == 0 else email
        _seed_contact(
            conn,
            email=email,
            display_name=display_name,
            msg_id=f"{prefix}-{idx}",
            subject=f"Launch thread {idx}",
            snippet=f"Roadmap update {idx}",
            body_text=f"Bidirectional planning note {idx}",
            ts=1701684900000 - idx,
            from_addr=from_addr,
            to_addr=to_addr,
        )


def _seed_human_like_one_way_contact(
    conn, *, email: str, display_name: str, prefix: str, total_messages: int = 3
):
    for idx in range(total_messages):
        _seed_contact(
            conn,
            email=email,
            display_name=display_name,
            msg_id=f"{prefix}-{idx}",
            subject=f"Project update {idx}",
            snippet=f"Following up on planning {idx}",
            body_text=f"Human-style note {idx}",
            ts=1701684901000 - idx,
            from_addr=email,
            to_addr="acct@example.com",
        )


def _seed_user_outbound_only_contact(
    conn, *, email: str, display_name: str, prefix: str, total_messages: int = 1
):
    for idx in range(total_messages):
        _seed_contact(
            conn,
            email=email,
            display_name=display_name,
            msg_id=f"{prefix}-out-{idx}",
            subject=f"Checking in {idx}",
            snippet=f"Following up with you {idx}",
            body_text=f"Wanted to reach out directly {idx}",
            ts=1701684901500 - idx,
            from_addr="acct@example.com",
            to_addr=email,
        )


def test_build_profile_context_samples_deep_includes_same_thread_context(conn):
    _seed_contact(
        conn,
        email="contact@example.com",
        display_name="Chris",
        msg_id="landlord-direct",
        subject="Lease plumbing follow-up",
        snippet="Can we schedule the contractor",
        body_text="Direct contact to user",
        ts=1701684903000,
        from_addr="contact@example.com",
        to_addr="acct@example.com",
        thread_id="thread-landlord-1",
    )
    upsert_message(
        conn,
        {
            "msg_id": "landlord-thread-extra",
            "account_email": "acct@example.com",
            "thread_id": "thread-landlord-1",
            "date_iso": "2023-12-04",
            "internal_ts": 1701684902999,
            "from_addr": "assistant@agency.example",
            "to_addr": "acct@example.com",
            "subject": "RE: Lease plumbing follow-up",
            "snippet": "Attached contractor estimate",
            "body_text": "Thread context from another participant",
            "labels": ["INBOX"],
            "history_id": 10,
        },
    )

    samples = build_profile_context_samples(
        conn,
        email="contact@example.com",
        tier="deep",
        max_threads=4,
        max_messages=10,
        max_chars=5000,
    )

    ids = [sample["msg_id"] for sample in samples]
    assert "landlord-direct" in ids
    assert "landlord-thread-extra" in ids

    direct = next(sample for sample in samples if sample["msg_id"] == "landlord-direct")
    extra = next(sample for sample in samples if sample["msg_id"] == "landlord-thread-extra")
    assert direct["context_source"] == "direct"
    assert extra["context_source"] == "thread"


def test_build_profile_context_samples_respects_caps(conn):
    _seed_contact(
        conn,
        email="contact@example.com",
        display_name="Chris",
        msg_id="d-1",
        subject="Lease item 1",
        snippet="A" * 120,
        body_text="B" * 400,
        ts=1701684905000,
        from_addr="contact@example.com",
        to_addr="acct@example.com",
        thread_id="thread-a",
    )
    _seed_contact(
        conn,
        email="contact@example.com",
        display_name="Chris",
        msg_id="d-2",
        subject="Lease item 2",
        snippet="A" * 120,
        body_text="B" * 400,
        ts=1701684904999,
        from_addr="acct@example.com",
        to_addr="contact@example.com",
        thread_id="thread-b",
    )
    upsert_message(
        conn,
        {
            "msg_id": "thread-a-extra",
            "account_email": "acct@example.com",
            "thread_id": "thread-a",
            "date_iso": "2023-12-04",
            "internal_ts": 1701684904998,
            "from_addr": "manager@agency.example",
            "to_addr": "acct@example.com",
            "subject": "RE Lease item 1",
            "snippet": "extra",
            "body_text": "context A",
            "labels": ["INBOX"],
            "history_id": 10,
        },
    )
    upsert_message(
        conn,
        {
            "msg_id": "thread-b-extra",
            "account_email": "acct@example.com",
            "thread_id": "thread-b",
            "date_iso": "2023-12-04",
            "internal_ts": 1701684904997,
            "from_addr": "manager@agency.example",
            "to_addr": "acct@example.com",
            "subject": "RE Lease item 2",
            "snippet": "extra",
            "body_text": "context B",
            "labels": ["INBOX"],
            "history_id": 10,
        },
    )

    samples = build_profile_context_samples(
        conn,
        email="contact@example.com",
        tier="deep",
        max_threads=1,
        max_messages=2,
        max_chars=700,
    )

    assert len(samples) <= 2
    used_threads = {sample["thread_id"] for sample in samples if sample["thread_id"]}
    assert used_threads <= {"thread-a"}


def test_build_profile_context_samples_prioritizes_target_relevant_thread_items(conn):
    _seed_contact(
        conn,
        email="contact@example.com",
        display_name="Chris",
        msg_id="seed-1",
        subject="Lease planning",
        snippet="Need timeline",
        body_text="direct",
        ts=1701684905100,
        from_addr="contact@example.com",
        to_addr="acct@example.com",
        thread_id="thread-a",
    )
    _seed_contact(
        conn,
        email="contact@example.com",
        display_name="Chris",
        msg_id="seed-2",
        subject="Lease docs",
        snippet="sharing docs",
        body_text="direct",
        ts=1701684905099,
        from_addr="acct@example.com",
        to_addr="contact@example.com",
        thread_id="thread-b",
    )
    upsert_message(
        conn,
        {
            "msg_id": "thread-a-target",
            "account_email": "acct@example.com",
            "thread_id": "thread-a",
            "date_iso": "2023-12-04",
            "internal_ts": 1701684905098,
            "from_addr": "contact@example.com",
            "to_addr": "acct@example.com",
            "subject": "RE Lease planning",
            "snippet": "target-focused context",
            "body_text": "target context",
            "labels": ["INBOX"],
            "history_id": 10,
        },
    )
    upsert_message(
        conn,
        {
            "msg_id": "thread-a-noise",
            "account_email": "acct@example.com",
            "thread_id": "thread-a",
            "date_iso": "2023-12-04",
            "internal_ts": 1701684905097,
            "from_addr": "assistant@agency.example",
            "to_addr": "acct@example.com",
            "subject": "RE Lease planning",
            "snippet": "non-target context",
            "body_text": "noise context",
            "labels": ["INBOX"],
            "history_id": 10,
        },
    )

    samples = build_profile_context_samples(
        conn,
        email="contact@example.com",
        tier="deep",
        max_threads=2,
        max_messages=3,
        max_chars=10000,
    )

    ids = [sample["msg_id"] for sample in samples]
    assert "thread-a-target" in ids
    assert "thread-a-noise" not in ids


def test_build_profiles_heuristic_mode(conn, app_cfg):
    _seed_quick_contact_data(conn)

    updated = build_profiles(conn, app_cfg, use_llm=False)
    assert updated == 1

    row = conn.execute(
        "SELECT profile_json, model FROM contact_profiles WHERE contact_email='alice@example.com'"
    ).fetchone()
    profile = json.loads(row[0])
    assert profile["role"] == "contact"
    assert "launch" in profile["common_topics"]
    assert profile["_meta"]["tier"] == "quick"
    assert profile["_meta"]["source"] == "heuristic"
    assert profile["_meta"]["message_count_at_build"] == 1
    assert isinstance(profile["_meta"]["updated_at"], str)
    assert row[1] is None


def test_build_profiles_heuristic_role_detection(conn, app_cfg):
    _seed_contact(
        conn,
        email="rentals@propertymanager.example",
        display_name="Property Desk",
        msg_id="m1",
        subject="Lease renewal options",
        snippet="Tenant must confirm rent update",
        body_text="Your apartment lease and rent terms for next month",
        ts=2,
        from_addr="rentals@propertymanager.example",
        to_addr="acct@example.com",
    )

    updated = build_profiles(conn, app_cfg, use_llm=False)
    assert updated == 1

    row = conn.execute(
        "SELECT profile_json FROM contact_profiles WHERE contact_email='rentals@propertymanager.example'"
    ).fetchone()
    profile = json.loads(row[0])
    assert profile["role"] == "landlord/property_manager"


def test_build_profiles_llm_override_for_deep_tier(conn, app_cfg, monkeypatch: pytest.MonkeyPatch):
    _seed_deep_bidirectional_contact(
        conn, email="alice@example.com", display_name="Alice", prefix="a"
    )
    monkeypatch.setattr(
        "inbox_vault.profiles.chat_json",
        lambda *_args, **_kwargs: {
            "role": "client",
            "common_topics": ["roadmap"],
            "tone": "direct",
            "relationship": "customer",
            "notes": "Synthetic profile",
        },
    )

    updated = build_profiles(conn, app_cfg, use_llm=True)
    assert updated == 1

    row = conn.execute(
        "SELECT profile_json, model FROM contact_profiles WHERE contact_email='alice@example.com'"
    ).fetchone()
    profile = json.loads(row[0])
    assert profile["role"] == "client"
    assert profile["_meta"]["tier"] == "deep"
    assert profile["_meta"]["source"] == "llm"
    assert row[1] == app_cfg.llm.model


def test_build_profiles_promotes_human_like_one_way_contact_to_deep_tier(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    _seed_human_like_one_way_contact(
        conn, email="ally@example.com", display_name="Ally", prefix="ally"
    )
    monkeypatch.setattr(
        "inbox_vault.profiles.chat_json",
        lambda *_args, **_kwargs: {
            "role": "peer",
            "common_topics": ["project"],
            "tone": "warm",
            "relationship": "collaborator",
            "notes": "Synthetic profile",
        },
    )

    updated = build_profiles(conn, app_cfg, use_llm=True)
    assert updated == 1

    row = conn.execute(
        "SELECT profile_json, model FROM contact_profiles WHERE contact_email='ally@example.com'"
    ).fetchone()
    profile = json.loads(row[0])
    assert profile["_meta"]["tier"] == "deep"
    assert profile["_meta"]["source"] == "llm"
    assert row[1] == app_cfg.llm.model


def test_build_profiles_promotes_user_outbound_contact_to_deep_tier(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    _seed_user_outbound_only_contact(
        conn, email="vendor@example.com", display_name="Vendor", prefix="vendor"
    )
    monkeypatch.setattr(
        "inbox_vault.profiles.chat_json",
        lambda *_args, **_kwargs: {
            "role": "vendor",
            "common_topics": ["follow-up"],
            "tone": "direct",
            "relationship": "outreach",
            "notes": "Synthetic profile",
        },
    )

    diagnostics: dict[str, int] = {}
    updated = build_profiles(conn, app_cfg, use_llm=True, diagnostics=diagnostics)
    assert updated == 1

    row = conn.execute(
        "SELECT profile_json, model FROM contact_profiles WHERE contact_email='vendor@example.com'"
    ).fetchone()
    profile = json.loads(row[0])
    assert profile["_meta"]["tier"] == "deep"
    assert profile["_meta"]["source"] == "llm"
    assert row[1] == app_cfg.llm.model
    assert diagnostics["attempted"] == 1


def test_build_profiles_keeps_low_signal_two_message_inbound_contact_quick(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    _seed_human_like_one_way_contact(
        conn,
        email="updates@service.example",
        display_name="Service Updates",
        prefix="svc",
        total_messages=2,
    )

    calls = {"n": 0}

    def _chat(*_args, **_kwargs):
        calls["n"] += 1
        return {
            "role": "notification",
            "common_topics": ["updates"],
            "tone": "neutral",
            "relationship": "service",
            "notes": "Synthetic profile",
        }

    monkeypatch.setattr("inbox_vault.profiles.chat_json", _chat)

    diagnostics: dict[str, int] = {}
    updated = build_profiles(conn, app_cfg, use_llm=True, diagnostics=diagnostics)

    assert updated == 1
    row = conn.execute(
        "SELECT profile_json, model FROM contact_profiles WHERE contact_email='updates@service.example'"
    ).fetchone()
    profile = json.loads(row[0])
    assert profile["_meta"]["tier"] == "quick"
    assert profile["_meta"]["source"] == "heuristic"
    assert row[1] is None
    assert calls["n"] == 0
    assert diagnostics["llm_skipped_quick_tier"] == 1


def test_build_profiles_falls_back_to_heuristics_when_llm_errors(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    _seed_deep_bidirectional_contact(
        conn, email="alice@example.com", display_name="Alice", prefix="a"
    )

    def _raise(*_args, **_kwargs):
        raise RuntimeError("synthetic llm error")

    monkeypatch.setattr("inbox_vault.profiles.chat_json", _raise)
    diagnostics: dict[str, int] = {}
    updated = build_profiles(conn, app_cfg, use_llm=True, diagnostics=diagnostics)

    assert updated == 1
    row = conn.execute(
        "SELECT profile_json, model FROM contact_profiles WHERE contact_email='alice@example.com'"
    ).fetchone()
    profile = json.loads(row[0])
    assert profile["role"] == "contact"
    assert profile["_meta"]["source"] == "heuristic"
    assert row[1] is None
    assert diagnostics["attempted"] == 1
    assert diagnostics["succeeded"] == 0
    assert diagnostics["fallback_used"] == 1
    assert diagnostics["stepA_attempted"] == 1
    assert diagnostics["stepA_succeeded"] == 1
    assert diagnostics["stepA_failed"] == 1
    assert diagnostics["stepB_attempted"] == 1


def test_build_profiles_uses_prompt_contract_in_messages(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    _seed_deep_bidirectional_contact(
        conn, email="alice@example.com", display_name="Alice", prefix="a"
    )
    captured_calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    def _chat(_cfg, messages, **kwargs):
        captured_calls.append((messages, kwargs))
        return {
            "role": "partner",
            "common_topics": ["launch"],
            "tone": "friendly",
            "relationship": "vendor",
            "notes": "Synthetic",
        }

    monkeypatch.setattr("inbox_vault.profiles.chat_json", _chat)
    updated = build_profiles(conn, app_cfg, use_llm=True)

    assert updated == 1
    prompts = [messages[1]["content"] for messages, _kwargs in captured_calls if len(messages) > 1]
    assert any("JSON schema contract:" in prompt for prompt in prompts)
    assert any("Context note:" in prompt for prompt in prompts)
    assert any("Chunk-awareness note" in prompt for prompt in prompts)
    assert any("Samples included:" in prompt for prompt in prompts)
    assert any("source=direct" in prompt for prompt in prompts)
    assert captured_calls[0][1]["temperature"] == 0.0


def test_build_profiles_repairs_empty_object(conn, app_cfg, monkeypatch: pytest.MonkeyPatch):
    _seed_deep_bidirectional_contact(
        conn, email="repair-profile@example.com", display_name="Repair", prefix="repair"
    )

    def _chat(_cfg, messages, **_kwargs):
        if "Original output:" in messages[1]["content"]:
            return {
                "role": "partner",
                "common_topics": ["launch"],
                "tone": "friendly",
                "relationship": "collaborator",
                "notes": "repaired",
            }
        return {}

    monkeypatch.setattr("inbox_vault.profiles.chat_json", _chat)

    diagnostics: dict[str, int] = {}
    updated = build_profiles(conn, app_cfg, use_llm=True, diagnostics=diagnostics)
    assert updated == 1

    row = conn.execute(
        "SELECT profile_json, model FROM contact_profiles WHERE contact_email='repair-profile@example.com'"
    ).fetchone()
    profile = json.loads(row[0])
    ok, _ = validate_profile_contract(profile)
    assert ok is True
    assert row[1] == app_cfg.llm.model
    assert diagnostics["repair_attempted"] == 1
    assert diagnostics["repair_succeeded"] == 1
    assert diagnostics["fallback_used"] == 1


def test_build_profiles_fallback_fill_when_repair_fails(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    _seed_deep_bidirectional_contact(
        conn,
        email="repair-fail-profile@example.com",
        display_name="Repair Fail",
        prefix="repair-fail",
    )

    def _chat(_cfg, messages, **_kwargs):
        if "Original output:" in messages[1]["content"]:
            return {"role": "", "common_topics": []}
        return {"role": "", "common_topics": []}

    monkeypatch.setattr("inbox_vault.profiles.chat_json", _chat)

    diagnostics: dict[str, int] = {}
    updated = build_profiles(conn, app_cfg, use_llm=True, diagnostics=diagnostics)
    assert updated == 1

    row = conn.execute(
        "SELECT profile_json, model FROM contact_profiles WHERE contact_email='repair-fail-profile@example.com'"
    ).fetchone()
    profile = json.loads(row[0])
    ok, _ = validate_profile_contract(profile)
    assert ok is True
    assert row[1] == app_cfg.llm.model
    assert diagnostics["repair_attempted"] == 2
    assert diagnostics["repair_succeeded"] == 0
    assert diagnostics["fallback_used"] == 1


def test_build_profiles_augment_with_gog_history_when_enabled(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    _seed_deep_bidirectional_contact(
        conn, email="alice@example.com", display_name="Alice", prefix="a", total_messages=4
    )

    app_cfg.profiles.gog_history_enabled = True
    app_cfg.profiles.gog_history_account = "acct@example.com"
    app_cfg.profiles.gog_history_command = "gog"
    app_cfg.profiles.gog_history_max_messages = 2
    app_cfg.profiles.deep_context_max_messages = 8
    app_cfg.profiles.deep_context_max_chars = 10000

    captured_calls: list[list[dict[str, str]]] = []

    def _fake_gog(_cfg, *, args):
        joined = " ".join(args)
        if "gmail messages search" in joined:
            return [{"id": "gog-1"}, {"id": "gog-2"}]
        if "gmail get gog-1" in joined:
            return {
                "headers": {
                    "subject": "Extra context 1",
                    "from": "alice@example.com",
                    "to": "acct@example.com",
                    "date": "Mon, 1 Jan 2026 10:00:00 -0500",
                },
                "message": {
                    "id": "gog-1",
                    "threadId": "g-thread-1",
                    "internalDate": "1772470413000",
                    "snippet": "extra snippet 1",
                },
            }
        if "gmail get gog-2" in joined:
            return {
                "headers": {
                    "subject": "Extra context 2",
                    "from": "acct@example.com",
                    "to": "alice@example.com",
                    "date": "Mon, 1 Jan 2026 10:01:00 -0500",
                },
                "message": {
                    "id": "gog-2",
                    "threadId": "g-thread-2",
                    "internalDate": "1772470414000",
                    "snippet": "extra snippet 2",
                },
            }
        raise AssertionError(f"unexpected gog args: {args}")

    def _chat(_cfg, messages, **_kwargs):
        captured_calls.append(messages)
        return {
            "role": "partner",
            "common_topics": ["launch"],
            "tone": "friendly",
            "relationship": "collaborator",
            "notes": "Synthetic",
        }

    monkeypatch.setattr("inbox_vault.profiles._run_gog_json", _fake_gog)
    monkeypatch.setattr("inbox_vault.profiles.chat_json", _chat)

    diagnostics: dict[str, int] = {}
    updated = build_profiles(conn, app_cfg, use_llm=True, diagnostics=diagnostics)

    assert updated == 1
    assert diagnostics["gog_attempted"] == 1
    assert diagnostics["gog_added"] == 2
    assert diagnostics["gog_failed"] == 0
    prompts = [messages[1]["content"] for messages in captured_calls if len(messages) > 1]
    assert any("source=gog" in prompt for prompt in prompts)


def test_build_profiles_populates_diagnostics(conn, app_cfg, monkeypatch: pytest.MonkeyPatch):
    _seed_deep_bidirectional_contact(
        conn, email="alice@example.com", display_name="Alice", prefix="a"
    )
    _seed_deep_bidirectional_contact(conn, email="bob@example.com", display_name="Bob", prefix="b")
    _seed_deep_bidirectional_contact(
        conn, email="carol@example.com", display_name="Carol", prefix="c"
    )

    def _chat(_cfg, messages, **_kwargs):
        user = messages[1]["content"]
        if "contact=alice@example.com" in user and "extract compact evidence" in user:
            return {
                "facts": ["launch planning"],
                "topics": ["launch"],
                "relationship_cues": ["client"],
                "tone_cues": ["direct"],
            }
        if "Contact email: alice@example.com" in user and "Evidence JSON" in user:
            return {
                "role": "client",
                "common_topics": ["launch"],
                "tone": "direct",
                "relationship": "customer",
                "notes": "Synthetic",
            }

        if "contact=bob@example.com" in user and "extract compact evidence" in user:
            return {
                "facts": ["billing status"],
                "topics": ["billing"],
                "relationship_cues": ["vendor"],
                "tone_cues": ["neutral"],
            }
        if "Contact email: bob@example.com" in user and "Evidence JSON" in user:
            raise requests.HTTPError("status=503")
        if "contact=bob@example.com" in user and "build/update a profile" in user:
            return {
                "role": "vendor",
                "common_topics": ["billing"],
                "tone": "neutral",
                "relationship": "partner",
                "notes": "retry",
            }

        if "contact=carol@example.com" in user and "extract compact evidence" in user:
            return None
        if "contact=carol@example.com" in user and "build/update a profile" in user:
            return None
        return None

    monkeypatch.setattr("inbox_vault.profiles.chat_json", _chat)

    diagnostics: dict[str, int] = {}
    updated = build_profiles(conn, app_cfg, use_llm=True, diagnostics=diagnostics)

    assert updated == 3
    assert diagnostics["attempted"] == 3
    assert diagnostics["succeeded"] == 2
    assert diagnostics["http_failed"] == 1
    assert diagnostics["fallback_used"] >= 2
    assert diagnostics["stepA_attempted"] == 3
    assert diagnostics["stepA_succeeded"] == 3
    assert diagnostics["stepB_attempted"] == 3
    assert diagnostics["stepB_succeeded"] == 2
    assert diagnostics["stepB_failed"] == 1

    rows = conn.execute(
        "SELECT contact_email, model FROM contact_profiles ORDER BY contact_email"
    ).fetchall()
    assert rows == [
        ("alice@example.com", app_cfg.llm.model),
        ("bob@example.com", app_cfg.llm.model),
        ("carol@example.com", None),
    ]


def test_build_profiles_skips_when_no_meaningful_new_evidence(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    _seed_deep_bidirectional_contact(
        conn, email="alice@example.com", display_name="Alice", prefix="a"
    )
    upsert_contact_profile(
        conn,
        "alice@example.com",
        {
            "role": "client",
            "common_topics": ["roadmap"],
            "tone": "direct",
            "relationship": "customer",
            "notes": "Existing",
            "_meta": {
                "evidence_count": 64,
                "tier": "deep",
                "source": "llm",
                "message_count_at_build": 10,
                "updated_at": "2025-01-01T00:00:00+00:00",
            },
        },
        model=app_cfg.llm.model,
    )

    calls = {"n": 0}

    def _chat(*_args, **_kwargs):
        calls["n"] += 1
        return {}

    monkeypatch.setattr("inbox_vault.profiles.chat_json", _chat)

    diagnostics: dict[str, int] = {}
    updated = build_profiles(conn, app_cfg, use_llm=True, diagnostics=diagnostics)

    assert updated == 0
    assert calls["n"] == 0
    assert diagnostics["skipped_no_new_evidence"] == 1


def test_build_profiles_allows_deep_heuristic_replace_when_evidence_is_significantly_better(
    conn, app_cfg
):
    _seed_deep_bidirectional_contact(
        conn, email="alice@example.com", display_name="Alice", prefix="a", total_messages=20
    )
    upsert_contact_profile(
        conn,
        "alice@example.com",
        {
            "role": "client",
            "common_topics": ["roadmap"],
            "tone": "direct",
            "relationship": "customer",
            "notes": "Old LLM profile",
            "_meta": {
                "evidence_count": 4,
                "tier": "deep",
                "source": "llm",
                "message_count_at_build": 4,
                "updated_at": "2025-01-01T00:00:00+00:00",
            },
        },
        model=app_cfg.llm.model,
    )

    diagnostics: dict[str, int] = {}
    updated = build_profiles(conn, app_cfg, use_llm=False, diagnostics=diagnostics)

    assert updated == 1
    row = conn.execute(
        "SELECT profile_json, model FROM contact_profiles WHERE contact_email='alice@example.com'"
    ).fetchone()
    profile = json.loads(row[0])
    assert profile["_meta"]["source"] == "heuristic"
    assert profile["_meta"]["tier"] == "deep"
    assert int(profile["_meta"]["evidence_count"]) > 4
    assert row[1] is None
    assert diagnostics["skipped_quality_guard"] == 0


def test_build_profiles_keeps_obvious_newsletter_in_quick_tier(
    conn, app_cfg, monkeypatch: pytest.MonkeyPatch
):
    _seed_human_like_one_way_contact(
        conn,
        email="digest@mail.beehiiv.com",
        display_name="Weekly Digest",
        prefix="digest",
        total_messages=3,
    )

    calls = {"n": 0}

    def _chat(*_args, **_kwargs):
        calls["n"] += 1
        return {
            "role": "newsletter",
            "common_topics": ["digest"],
            "tone": "neutral",
            "relationship": "subscription",
            "notes": "Synthetic profile",
        }

    monkeypatch.setattr("inbox_vault.profiles.chat_json", _chat)

    diagnostics: dict[str, int] = {}
    updated = build_profiles(conn, app_cfg, use_llm=True, diagnostics=diagnostics)

    assert updated == 1
    row = conn.execute(
        "SELECT profile_json, model FROM contact_profiles WHERE contact_email='digest@mail.beehiiv.com'"
    ).fetchone()
    profile = json.loads(row[0])
    assert profile["_meta"]["tier"] == "quick"
    assert profile["_meta"]["source"] == "heuristic"
    assert row[1] is None
    assert calls["n"] == 0
    assert diagnostics["llm_skipped_quick_tier"] == 1


def test_build_profiles_quality_guard_keeps_existing_llm_on_quick_tier(conn, app_cfg):
    _seed_contact(
        conn,
        email="noreply@service.example",
        display_name="Service",
        msg_id="m1",
        subject="Do not reply",
        snippet="Newsletter and alerts",
        body_text="Manage preferences / unsubscribe",
        ts=1701684900000,
        from_addr="noreply@service.example",
        to_addr="acct@example.com",
    )

    upsert_contact_profile(
        conn,
        "noreply@service.example",
        {
            "role": "notification",
            "common_topics": ["alerts"],
            "tone": "neutral",
            "relationship": "service",
            "notes": "Existing stronger profile",
            "_meta": {
                "evidence_count": 20,
                "tier": "deep",
                "source": "llm",
                "message_count_at_build": 20,
                "updated_at": "2025-01-01T00:00:00+00:00",
            },
        },
        model=app_cfg.llm.model,
    )

    diagnostics: dict[str, int] = {}
    updated = build_profiles(conn, app_cfg, use_llm=True, diagnostics=diagnostics)

    assert updated == 0
    assert diagnostics["skipped_quality_guard"] == 1
    assert diagnostics["llm_skipped_quick_tier"] == 1
