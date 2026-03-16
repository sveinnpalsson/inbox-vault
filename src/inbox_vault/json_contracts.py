from __future__ import annotations

from typing import Any

PROFILE_REQUIRED_KEYS = ("role", "common_topics", "tone", "relationship", "notes")
ENRICH_REQUIRED_KEYS = ("category", "importance", "action", "summary")


def profile_contract_text() -> str:
    return (
        "Required JSON object contract for contact profiles:\n"
        "- role: non-empty string\n"
        "- common_topics: array of non-empty strings (at least one item)\n"
        "- tone: non-empty string\n"
        "- relationship: non-empty string\n"
        "- notes: non-empty string"
    )


def enrich_contract_text() -> str:
    return (
        "Required JSON object contract for message enrichment:\n"
        "- category: non-empty string\n"
        "- importance: integer 1..10\n"
        "- action: non-empty string\n"
        "- summary: non-empty string"
    )


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _as_non_empty_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            token = item.strip()
            if token:
                out.append(token)
    return out


def validate_profile_contract(payload: dict[str, Any] | None) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not isinstance(payload, dict):
        return False, ["payload_not_object"]

    for key in PROFILE_REQUIRED_KEYS:
        if key not in payload:
            issues.append(f"missing:{key}")

    if "role" in payload and not _is_non_empty_string(payload.get("role")):
        issues.append("invalid:role")
    if "common_topics" in payload and not _as_non_empty_string_list(payload.get("common_topics")):
        issues.append("invalid:common_topics")
    if "tone" in payload and not _is_non_empty_string(payload.get("tone")):
        issues.append("invalid:tone")
    if "relationship" in payload and not _is_non_empty_string(payload.get("relationship")):
        issues.append("invalid:relationship")
    if "notes" in payload and not _is_non_empty_string(payload.get("notes")):
        issues.append("invalid:notes")

    return not issues, issues


def validate_enrich_contract(payload: dict[str, Any] | None) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not isinstance(payload, dict):
        return False, ["payload_not_object"]

    for key in ENRICH_REQUIRED_KEYS:
        if key not in payload:
            issues.append(f"missing:{key}")

    if "category" in payload and not _is_non_empty_string(payload.get("category")):
        issues.append("invalid:category")

    importance = payload.get("importance") if isinstance(payload, dict) else None
    if "importance" in payload:
        if not isinstance(importance, int) or not (1 <= importance <= 10):
            issues.append("invalid:importance")

    if "action" in payload and not _is_non_empty_string(payload.get("action")):
        issues.append("invalid:action")
    if "summary" in payload and not _is_non_empty_string(payload.get("summary")):
        issues.append("invalid:summary")

    return not issues, issues


def fill_profile_defaults(payload: dict[str, Any] | None, *, email: str) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    topics = _as_non_empty_string_list(source.get("common_topics"))
    if not topics:
        topics = ["general"]

    role = str(source.get("role") or "contact").strip() or "contact"
    tone = str(source.get("tone") or "unknown").strip() or "unknown"
    relationship = str(
        source.get("relationship") or f"email contact ({email or 'unknown'})"
    ).strip()
    if not relationship:
        relationship = f"email contact ({email or 'unknown'})"
    notes = (
        str(source.get("notes") or "fallback-filled profile fields").strip()
        or "fallback-filled profile fields"
    )

    return {
        "role": role,
        "common_topics": topics,
        "tone": tone,
        "relationship": relationship,
        "notes": notes,
    }


def fill_enrich_defaults(
    payload: dict[str, Any] | None, *, subject: str, snippet: str
) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}

    category = str(source.get("category") or "general").strip() or "general"
    action = str(source.get("action") or "review").strip() or "review"

    raw_importance = source.get("importance")
    importance = 4
    if isinstance(raw_importance, int):
        importance = raw_importance
    else:
        try:
            importance = int(raw_importance)
        except (TypeError, ValueError):
            importance = 4
    importance = max(1, min(10, importance))

    summary_seed = str(source.get("summary") or "").strip()
    if not summary_seed:
        subject_text = (subject or "").strip()
        snippet_text = (snippet or "").strip()
        summary_seed = f"{subject_text} — {snippet_text}".strip(" —") or "Email update"

    return {
        "category": category,
        "importance": importance,
        "action": action,
        "summary": summary_seed,
    }
