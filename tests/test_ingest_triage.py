from __future__ import annotations

import json

from inbox_vault.ingest_triage import derive_ingest_triage, normalize_subject_family
from tests.factories import gmail_message_payload


def test_normalize_subject_family_strips_prefixes_and_noise():
    assert normalize_subject_family("Re: Weekly Digest 2026-04-17 09:30 #981245") == "weekly digest"


def test_derive_ingest_triage_prefers_stream_fingerprint_over_sender_only():
    digest_raw = gmail_message_payload(
        "m-digest",
        from_addr="no-reply@example.com",
        subject="Weekly Digest 2026-04-17",
        body_text="Top stories for you",
        labels=["INBOX", "CATEGORY_PROMOTIONS"],
    )
    digest_raw["payload"]["headers"].extend(
        [
            {"name": "List-Id", "value": "digest.example.com"},
            {"name": "List-Unsubscribe", "value": "<https://example.com/unsub?id=1>"},
            {"name": "Precedence", "value": "bulk"},
        ]
    )
    digest_rec = {
        "msg_id": "m-digest",
        "account_email": "recipient@example.com",
        "from_addr": "no-reply@example.com",
        "to_addr": "recipient@example.com",
        "subject": "Weekly Digest 2026-04-17",
        "snippet": "Top stories for you",
        "labels": ["INBOX", "CATEGORY_PROMOTIONS"],
    }

    security_raw = gmail_message_payload(
        "m-sec",
        from_addr="no-reply@example.com",
        subject="Verify your sign-in",
        body_text="Verify your sign-in attempt now.",
        labels=["INBOX"],
    )
    security_rec = {
        "msg_id": "m-sec",
        "account_email": "recipient@example.com",
        "from_addr": "no-reply@example.com",
        "to_addr": "recipient@example.com",
        "subject": "Verify your sign-in",
        "snippet": "Verify your sign-in attempt now.",
        "labels": ["INBOX"],
    }

    digest = derive_ingest_triage(digest_rec, raw_payload=digest_raw)
    security = derive_ingest_triage(security_rec, raw_payload=security_raw)

    assert digest.sender_domain == security.sender_domain == "example.com"
    assert digest.subject_family != security.subject_family
    assert digest.stream_id != security.stream_id
    assert digest.triage_tier in {"light", "minimal"}
    assert security.triage_tier == "full"

    digest_signals = json.loads(digest.signals_json)
    assert digest_signals["list_id"] is True
    assert digest_signals["bulk_label"] is True
