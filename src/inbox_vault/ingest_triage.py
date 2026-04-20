from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

_SUBJECT_PREFIX_RE = re.compile(r"^(?:(?:re|fw|fwd)\s*:\s*)+", re.IGNORECASE)
_SUBJECT_NOISE_RE = re.compile(
    r"\b(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\d{1,2}:\d{2}(?::\d{2})?|\d{5,}|[a-f0-9]{8,})\b",
    re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

_BULK_LABELS = {"CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_FORUMS"}
_IMPORTANT_KEYWORDS = (
    "verify",
    "sign-in",
    "signin",
    "password",
    "passcode",
    "mfa",
    "2fa",
    "security alert",
    "billing",
    "invoice",
    "receipt",
    "statement",
    "payment failed",
    "recovery",
)


@dataclass(slots=True)
class IngestTriageResult:
    msg_id: str
    account_email: str
    stream_id: str
    stream_kind: str
    subject_family: str
    sender_domain: str
    triage_tier: str
    decision_source: str
    bulk_score: int
    importance_score: int
    novelty_score: float
    signals_json: str


def _headers_from_raw(raw: dict[str, Any] | None) -> dict[str, str]:
    headers = (raw or {}).get("payload", {}).get("headers", [])
    out: dict[str, str] = {}
    for item in headers:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip().lower()
        value = str(item.get("value") or "").strip()
        if name and value:
            out[name] = value
    return out


def normalize_sender_domain(from_addr: str | None) -> str:
    addr = (from_addr or "").strip().lower()
    if "@" not in addr:
        return ""
    return addr.rsplit("@", 1)[-1]


def normalize_subject_family(subject: str | None) -> str:
    text = _SUBJECT_PREFIX_RE.sub("", (subject or "").strip().lower())
    text = _SUBJECT_NOISE_RE.sub(" ", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "no subject"


def _unsubscribe_host(value: str) -> str:
    for chunk in value.split(","):
        chunk = chunk.strip().strip("<>").strip()
        if not chunk:
            continue
        parsed = urlparse(chunk)
        if parsed.netloc:
            return parsed.netloc.lower()
    return ""


def derive_ingest_triage(
    rec: dict[str, Any],
    *,
    raw_payload: dict[str, Any] | None = None,
    prior_observation_count: int = 0,
) -> IngestTriageResult:
    headers = _headers_from_raw(raw_payload)
    labels = {str(label).strip().upper() for label in rec.get("labels", []) if str(label).strip()}
    subject_family = normalize_subject_family(rec.get("subject"))
    sender_domain = normalize_sender_domain(rec.get("from_addr"))
    list_id = headers.get("list-id", "").lower()
    list_unsubscribe = headers.get("list-unsubscribe", "")
    unsubscribe_host = _unsubscribe_host(list_unsubscribe)
    precedence = headers.get("precedence", "").lower()
    auto_submitted = headers.get("auto-submitted", "").lower()
    sender_local = (rec.get("from_addr") or "").strip().lower().split("@", 1)[0]
    preview_text = " ".join(
        part.strip().lower() for part in [rec.get("subject") or "", rec.get("snippet") or ""] if part
    )

    important_hits = [keyword for keyword in _IMPORTANT_KEYWORDS if keyword in preview_text]
    has_bulk_label = bool(labels & _BULK_LABELS)
    has_list_headers = bool(list_id or list_unsubscribe)
    has_bulk_precedence = precedence in {"bulk", "list", "junk"}
    has_auto_submitted = bool(auto_submitted and auto_submitted != "no")
    has_no_reply_sender = sender_local.startswith(("no-reply", "noreply", "donotreply"))

    bulk_score = int(has_bulk_label) * 2
    bulk_score += int(bool(list_id)) * 2
    bulk_score += int(bool(list_unsubscribe))
    bulk_score += int(has_bulk_precedence)
    bulk_score += int(has_auto_submitted)
    bulk_score += int(has_no_reply_sender)

    importance_score = min(3, len(important_hits))
    if (
        not has_bulk_label
        and not has_list_headers
        and not has_bulk_precedence
        and rec.get("to_addr")
        and str(rec.get("to_addr")).strip().lower() == rec.get("account_email", "").lower()
    ):
        importance_score += 1

    if importance_score >= 2:
        triage_tier = "full"
    elif bulk_score >= 4:
        triage_tier = "minimal"
    elif bulk_score >= 2:
        triage_tier = "light"
    else:
        triage_tier = "full"

    stream_kind = "bulk" if (has_bulk_label or has_list_headers or has_bulk_precedence) else "direct"
    stream_key = {
        "account_email": (rec.get("account_email") or "").strip().lower(),
        "sender_domain": sender_domain,
        "subject_family": subject_family,
        "list_id": list_id,
        "unsubscribe_host": unsubscribe_host,
        "stream_kind": stream_kind,
    }
    stream_id = "stream_" + hashlib.sha1(
        json.dumps(stream_key, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]

    novelty_score = 1.0 if int(prior_observation_count or 0) <= 0 else 0.0
    signals = {
        "labels": sorted(labels),
        "list_id": bool(list_id),
        "list_unsubscribe": bool(list_unsubscribe),
        "unsubscribe_host": unsubscribe_host,
        "precedence": precedence,
        "auto_submitted": auto_submitted,
        "bulk_label": has_bulk_label,
        "no_reply_sender": has_no_reply_sender,
        "important_keywords": important_hits,
        "prior_observation_count": int(prior_observation_count or 0),
    }

    return IngestTriageResult(
        msg_id=str(rec.get("msg_id") or ""),
        account_email=str(rec.get("account_email") or ""),
        stream_id=stream_id,
        stream_kind=stream_kind,
        subject_family=subject_family,
        sender_domain=sender_domain,
        triage_tier=triage_tier,
        decision_source="rules",
        bulk_score=bulk_score,
        importance_score=importance_score,
        novelty_score=novelty_score,
        signals_json=json.dumps(signals, sort_keys=True),
    )
