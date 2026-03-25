from __future__ import annotations

import json
from typing import Any, Sequence

_JSON_DETERMINISTIC_RULES = (
    "Output rules:\n"
    "- Return exactly one JSON object and nothing else.\n"
    "- Use double-quoted keys/strings (valid JSON).\n"
    "- No markdown, no code fences, no prose outside JSON.\n"
    "- Do not output reasoning traces or thinking blocks.\n"
    "- If a field is unknown, use an empty string or empty array instead of omitting keys."
)

_ENRICHMENT_SCHEMA = {
    "type": "object",
    "required": ["category", "importance", "action", "summary"],
    "properties": {
        "category": {
            "type": "string",
            "description": "Short category label, e.g. work, personal, billing",
        },
        "importance": {"type": "integer", "minimum": 1, "maximum": 10},
        "action": {
            "type": "string",
            "description": "Primary next action: reply, review, schedule, archive, none",
        },
        "summary": {
            "type": "string",
            "description": "One concise sentence preserving actionable details",
        },
    },
}

_PROFILE_SCHEMA = {
    "type": "object",
    "required": ["role", "common_topics", "tone", "relationship", "notes"],
    "properties": {
        "role": {"type": "string"},
        "common_topics": {"type": "array", "items": {"type": "string"}},
        "tone": {"type": "string"},
        "relationship": {"type": "string"},
        "notes": {"type": "string"},
    },
}

_PROFILE_EVIDENCE_SCHEMA = {
    "type": "object",
    "required": ["facts", "topics", "relationship_cues", "tone_cues"],
    "properties": {
        "facts": {"type": "array", "items": {"type": "string"}},
        "topics": {"type": "array", "items": {"type": "string"}},
        "relationship_cues": {"type": "array", "items": {"type": "string"}},
        "tone_cues": {"type": "array", "items": {"type": "string"}},
    },
}


ENRICH_SYSTEM_PROMPT = (
    "You are an email triage classifier for a privacy-first local system. "
    "Classify one email and summarize it with high utility. "
    "Use only the provided message content. "
    "Do not reveal hidden reasoning or infer facts not present in the input.\n\n"
    "Safety and privacy constraints:\n"
    "- Never include secrets not present in the source email.\n"
    "- Preserve operational meaning while avoiding unnecessary sensitive detail in summary text.\n"
    "- If uncertain, choose conservative labels and keep summary factual."
)

PROFILE_SYSTEM_PROMPT = (
    "You are a contact-profile extractor for a privacy-first local system. "
    "Build a compact profile from sample exchanges only.\n\n"
    "Safety and privacy constraints:\n"
    "- Use only supplied samples, no outside knowledge.\n"
    "- Do not invent personal attributes not evidenced in messages.\n"
    "- Keep notes useful for email triage; avoid unnecessary sensitive details."
)

REDACTION_SYSTEM_PROMPT = (
    "You are a deterministic PII detection engine for email text. "
    "Return structured JSON redaction candidates for the provided chunk/document.\n\n"
    "Safety and behavior constraints:\n"
    "- Identify only concrete sensitive values present in the provided text.\n"
    "- Never add new facts, summaries, or commentary.\n"
    "- Allowed key names are only: EMAIL, PHONE, URL, ACCOUNT, PERSON, ADDRESS.\n"
    "- Do not emit generic categories such as NAME, LOCATION, SECRET, ID, CUSTOM, or OTHER.\n"
    "- Do not emit field labels, headers, or generic words such as 'name', 'last name', 'address', or state abbreviations by themselves.\n"
    "- If uncertain, abstain and return no candidate for that span.\n"
    "- Return key/name + value list pairs so downstream systems can validate and persist mappings."
)


def _json_schema_contract(schema: dict) -> str:
    return "JSON schema contract:\n" + json.dumps(schema, separators=(",", ":"), sort_keys=True)


def build_enrichment_messages(
    *,
    subject: str,
    snippet: str,
    body_text: str,
    from_addr: str,
    to_addr: str,
    date_iso: str,
    body_max_chars: int = 3000,
    compact: bool = False,
) -> list[dict[str, str]]:
    safe_max = max(200, int(body_max_chars))
    body = body_text or ""
    body_truncated = body[:safe_max]
    trunc_note = "truncated" if len(body) > safe_max else "full"

    if compact:
        user_prompt = (
            "Control: /no_think\n"
            "Task: classify and summarize this email (compact mode).\n"
            f"Subject: {subject}\n"
            f"Snippet: {snippet}\n\n"
            "Return JSON with keys: category (string), importance (1-10 int), action (string), summary (string).\n"
            "No markdown, no code fences, no extra prose."
        )
    else:
        user_prompt = (
            "Control: /no_think\n"
            "Task: classify and summarize this email.\n"
            f"Date: {date_iso}\n"
            f"From: {from_addr}\n"
            f"To: {to_addr}\n"
            f"Subject: {subject}\n"
            f"Snippet: {snippet}\n"
            f"Body ({trunc_note}, max_chars={safe_max}): {body_truncated}\n\n"
            f"{_json_schema_contract(_ENRICHMENT_SCHEMA)}\n"
            f"{_JSON_DETERMINISTIC_RULES}"
        )

    return [
        {"role": "system", "content": ENRICH_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _profile_sample_lines(
    *, samples: Sequence[Any], safe_chars: int, max_samples: int
) -> list[str]:
    selected = samples[: max(1, int(max_samples))]
    convo_lines: list[str] = []
    for idx, sample in enumerate(selected, start=1):
        if isinstance(sample, dict):
            subject = str(sample.get("subject") or "")
            snippet = str(sample.get("snippet") or "")
            body_text = str(sample.get("body_text") or "")
            from_addr = str(sample.get("from_addr") or "")
            to_addr = str(sample.get("to_addr") or "")
            date_iso = str(sample.get("date_iso") or "")
            thread_id = str(sample.get("thread_id") or "")
            context_source = str(sample.get("context_source") or "direct")
        else:
            subject, snippet, body_text, from_addr, to_addr, date_iso = sample
            thread_id = ""
            context_source = "direct"

        snippet_excerpt = (snippet or "")[:safe_chars]
        body_excerpt = (body_text or "")[:safe_chars]
        context_payload = body_excerpt or snippet_excerpt

        fields = [
            f"[{idx}] source={context_source} date={date_iso} from={from_addr} to={to_addr} subject={subject}",
            f"content={context_payload}",
        ]
        if thread_id:
            fields.append(f"thread_id={thread_id}")
        if snippet_excerpt:
            fields.append(f"snippet={snippet_excerpt}")
        if body_excerpt:
            fields.append(f"body_excerpt={body_excerpt}")

        convo_lines.append("\n".join(fields))
    return convo_lines


def _bounded_lines(
    *, lines: list[str], static_prefix: str, static_suffix: str, budget_chars: int | None
) -> list[str]:
    if budget_chars is None:
        return lines

    safe_budget = max(1200, int(budget_chars))
    budget_for_lines = max(0, safe_budget - len(static_prefix) - len(static_suffix))
    if budget_for_lines <= 0:
        return []

    out: list[str] = []
    running = 0
    for line in lines:
        cost = len(line) + (2 if out else 0)
        if out and (running + cost > budget_for_lines):
            break
        if not out and cost > budget_for_lines:
            out.append(line[:budget_for_lines])
            break
        out.append(line)
        running += cost
    return out


def build_profile_messages(
    *,
    contact_email: str,
    samples: Sequence[Any],
    max_samples: int = 10,
    sample_chars: int = 320,
    prompt_budget_chars: int | None = None,
) -> list[dict[str, str]]:
    safe_chars = max(120, int(sample_chars))
    convo_lines = _profile_sample_lines(
        samples=samples, safe_chars=safe_chars, max_samples=max_samples
    )

    schema_and_rules = f"{_json_schema_contract(_PROFILE_SCHEMA)}\n{_JSON_DETERMINISTIC_RULES}"
    static_prefix = (
        f"Task: build/update a profile for contact={contact_email}.\n"
        "Samples included: 0\n"
        "Context note: source=direct means messages directly to/from this contact; "
        "source=thread means same thread_id context added for deeper history.\n"
        "Chunk-awareness note: these are partial snippets from larger threads; avoid overfitting to one sample.\n\n"
    )
    bounded = _bounded_lines(
        lines=convo_lines,
        static_prefix=static_prefix,
        static_suffix=f"\n\n{schema_and_rules}",
        budget_chars=prompt_budget_chars,
    )

    joined = "\n\n".join(bounded)
    user_prompt = (
        f"Task: build/update a profile for contact={contact_email}.\n"
        f"Samples included: {len(bounded)}\n"
        "Context note: source=direct means messages directly to/from this contact; "
        "source=thread means same thread_id context added for deeper history.\n"
        "Chunk-awareness note: these are partial snippets from larger threads; avoid overfitting to one sample.\n\n"
        f"{joined}\n\n"
        f"{schema_and_rules}"
    )

    return [
        {"role": "system", "content": PROFILE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_profile_evidence_messages(
    *,
    contact_email: str,
    samples: Sequence[Any],
    max_samples: int = 10,
    sample_chars: int = 320,
    prompt_budget_chars: int | None = None,
) -> list[dict[str, str]]:
    safe_chars = max(120, int(sample_chars))
    convo_lines = _profile_sample_lines(
        samples=samples, safe_chars=safe_chars, max_samples=max_samples
    )

    schema_and_rules = (
        f"{_json_schema_contract(_PROFILE_EVIDENCE_SCHEMA)}\n{_JSON_DETERMINISTIC_RULES}"
    )
    static_prefix = (
        f"Task: extract compact evidence for contact={contact_email}.\n"
        "Samples included: 0\n"
        "Context note: source=direct means messages directly to/from this contact; "
        "source=thread means same thread_id context added for deeper history.\n"
        "Chunk-awareness note: these are partial snippets from larger threads; avoid overfitting to one sample.\n"
        "Focus: prioritize concrete message-grounded clues, not speculation.\n\n"
    )
    bounded = _bounded_lines(
        lines=convo_lines,
        static_prefix=static_prefix,
        static_suffix=f"\n\n{schema_and_rules}",
        budget_chars=prompt_budget_chars,
    )

    joined = "\n\n".join(bounded)
    user_prompt = (
        f"Task: extract compact evidence for contact={contact_email}.\n"
        f"Samples included: {len(bounded)}\n"
        "Context note: source=direct means messages directly to/from this contact; "
        "source=thread means same thread_id context added for deeper history.\n"
        "Chunk-awareness note: these are partial snippets from larger threads; avoid overfitting to one sample.\n"
        "Focus: prioritize concrete message-grounded clues, not speculation.\n\n"
        f"{joined}\n\n"
        f"{schema_and_rules}"
    )
    return [
        {"role": "system", "content": PROFILE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_redaction_messages(
    *,
    chunk_text: str,
    profile: str,
    instruction: str,
    chunk_index: int,
    chunk_total: int,
) -> list[dict[str, str]]:
    user_prompt = (
        f"Clearance profile: {profile or 'standard'}\n"
        f"Additional redaction instruction: {instruction or 'none'}\n"
        f"Chunk position: {chunk_index}/{chunk_total}\n"
        "Chunk handling rule: analyze this chunk independently, but include any full sensitive value visible in this chunk.\n\n"
        "Return JSON only with this exact shape:\n"
        "{\"redactions\":[{\"key_name\":\"EMAIL\",\"values\":[\"alice@example.com\"]}]}\n"
        "- key_name must be one of: EMAIL, PHONE, URL, ACCOUNT, PERSON, ADDRESS.\n"
        "- values must be concrete sensitive strings found verbatim in the chunk.\n"
        "- Never emit labels, placeholders, summaries, or guessed values.\n"
        "- If nothing sensitive is found, return {\"redactions\":[]}.\n\n"
        "Analyze the following chunk exactly once:\n"
        "---CHUNK---\n"
        f"{chunk_text}\n"
        "---END CHUNK---"
    )
    return [
        {"role": "system", "content": REDACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
