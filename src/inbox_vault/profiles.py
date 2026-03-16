from __future__ import annotations

import json
import logging
import math
import re
import subprocess
from collections.abc import Sequence
from html import unescape
from typing import Any

import requests

from .config import AppConfig
from .db import contact_directional_counts, profile_candidates, upsert_contact_profile, utc_now
from .json_contracts import fill_profile_defaults, profile_contract_text, validate_profile_contract
from .llm import chat_json
from .prompts import build_profile_evidence_messages, build_profile_messages

LOG = logging.getLogger(__name__)

LLMDiagnostics = dict[str, int]

_NO_REPLY_HINTS = (
    "noreply",
    "no-reply",
    "do-not-reply",
    "donotreply",
    "newsletter",
    "mailer-daemon",
    "notifications",
    "beehiiv",
    "substack",
    "mailchimp",
    "list-manage",
)

_DEEP_MIN_MESSAGES = 2
_DEEP_MIN_BIDIRECTIONAL = 1
_DEEP_MIN_ONE_WAY_INBOUND_MESSAGES = 3
_SIGNIFICANT_EVIDENCE_GROWTH_ABS = 3
_SIGNIFICANT_EVIDENCE_GROWTH_REL = 0.25

_LOW_SIGNAL_LOCALPART_HINTS = (
    "update",
    "updates",
    "news",
    "digest",
    "newsletter",
    "notify",
    "notifications",
    "alert",
    "alerts",
    "receipt",
    "receipts",
    "billing",
)

_DIRECT_PROFILE_SAMPLE_LIMIT = 40
_DEEP_PROFILE_SAMPLE_CHARS = 700
_GOG_SEARCH_CANDIDATE_MULTIPLIER = 3
_GOG_SEARCH_CANDIDATE_CAP = 40

ProfileSample = dict[str, Any]


def _empty_diag() -> LLMDiagnostics:
    return {
        "attempted": 0,
        "succeeded": 0,
        "http_failed": 0,
        "parse_failed": 0,
        "contract_failed": 0,
        "repair_attempted": 0,
        "repair_succeeded": 0,
        "fallback_used": 0,
        "skipped_no_new_evidence": 0,
        "skipped_quality_guard": 0,
        "llm_skipped_quick_tier": 0,
        "gog_attempted": 0,
        "gog_added": 0,
        "gog_failed": 0,
        "stepA_attempted": 0,
        "stepA_succeeded": 0,
        "stepA_failed": 0,
        "stepB_attempted": 0,
        "stepB_succeeded": 0,
        "stepB_failed": 0,
    }


def _sample_field(sample: Any, index: int, key: str) -> str:
    if isinstance(sample, dict):
        value = sample.get(key)
        return str(value or "")
    if (
        isinstance(sample, Sequence)
        and not isinstance(sample, (str, bytes, bytearray))
        and len(sample) > index
    ):
        value = sample[index]
        return str(value or "")
    return ""


def _sample_haystack(samples) -> str:
    bits: list[str] = []
    for sample in samples:
        bits.extend(
            [
                _sample_field(sample, 0, "subject"),
                _sample_field(sample, 1, "snippet"),
                _sample_field(sample, 2, "body_text"),
                _sample_field(sample, 3, "from_addr"),
                _sample_field(sample, 4, "to_addr"),
            ]
        )
    return " ".join(bits).lower()


def _looks_noreply_or_newsletter(email: str, samples) -> bool:
    lowered_email = (email or "").strip().lower()
    if any(hint in lowered_email for hint in _NO_REPLY_HINTS):
        return True

    text = _sample_haystack(samples)
    return any(
        hint in text
        for hint in (
            "do not reply",
            "unsubscribe",
            "manage preferences",
            "view in browser",
        )
    )


def _text_contains_keyword(text: str, keyword: str) -> bool:
    token = str(keyword or "").strip().lower()
    if not token:
        return False
    if " " in token or "-" in token or "#" in token:
        return token in text
    return re.search(rf"\b{re.escape(token)}\b", text) is not None


def _contains_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    return any(_text_contains_keyword(text, keyword) for keyword in keywords)


def _heuristic_role(email: str, samples) -> str:
    text = _sample_haystack(samples)
    domain = (email.split("@")[-1] if "@" in email else email).lower()

    if _contains_any_keyword(
        text, ("landlord", "property manager", "lease", "rent", "tenant", "apartment")
    ):
        return "landlord/property_manager"
    if _contains_any_keyword(
        text, ("bank", "statement", "transaction", "checking", "savings", "credit card")
    ) or any(d in domain for d in ("bank", "chase", "wellsfargo", "citi", "capitalone", "amex")):
        return "bank"
    if _contains_any_keyword(
        text, ("support ticket", "helpdesk", "customer support", "case #", "issue resolved")
    ) or any(d in domain for d in ("support", "help", "zendesk", "freshdesk")):
        return "support"
    if _contains_any_keyword(
        text, ("noreply", "do not reply", "notification", "alert", "reminder", "receipt")
    ) or any(d in domain for d in ("notifications", "mailer", "noreply")):
        return "notification"
    return "contact"


def _is_low_signal_one_way_inbound(email: str, samples) -> bool:
    localpart = (email.split("@", 1)[0] if "@" in email else email).lower().strip()
    if any(
        localpart == hint or localpart.startswith(f"{hint}.") or localpart.startswith(f"{hint}-")
        for hint in _LOW_SIGNAL_LOCALPART_HINTS
    ):
        return True

    text = _sample_haystack(samples)
    return any(
        phrase in text
        for phrase in (
            "weekly update",
            "daily update",
            "unsubscribe",
            "manage preferences",
            "view in browser",
            "privacy policy",
        )
    )


def _compute_signal(
    *, email: str, message_count: int, inbound_count: int, outbound_count: int, samples
) -> dict[str, int | str | bool]:
    noreply_like = _looks_noreply_or_newsletter(email, samples)
    bidirectional = min(inbound_count, outbound_count)
    user_reached_out = outbound_count > 0
    low_signal_inbound = _is_low_signal_one_way_inbound(email, samples)

    is_deep = False
    if not noreply_like:
        if bidirectional >= _DEEP_MIN_BIDIRECTIONAL:
            is_deep = True
        elif user_reached_out:
            # Prioritize contacts the user actively writes/replies to.
            is_deep = True
        elif message_count >= _DEEP_MIN_ONE_WAY_INBOUND_MESSAGES and not low_signal_inbound:
            # Allow one-way inbound promotion only when there is enough volume and
            # the sender does not look like bulk/update traffic.
            is_deep = True

    tier = "deep" if is_deep else "quick"

    evidence_count = inbound_count + outbound_count
    if evidence_count <= 0:
        evidence_count = max(1, min(int(message_count), 3))
    if tier == "deep":
        evidence_count += 2

    return {
        "tier": tier,
        "noreply_like": noreply_like,
        "inbound_count": int(inbound_count),
        "outbound_count": int(outbound_count),
        "evidence_count": int(evidence_count),
    }


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _row_to_sample(row) -> ProfileSample:
    return {
        "msg_id": row[0],
        "thread_id": row[1],
        "internal_ts": _coerce_int(row[2]),
        "date_iso": row[3] or "",
        "subject": row[4] or "",
        "snippet": row[5] or "",
        "body_text": row[6] or "",
        "from_addr": row[7] or "",
        "to_addr": row[8] or "",
        "context_source": "direct",
    }


def _fetch_direct_profile_samples(
    conn, *, email: str, limit: int = _DIRECT_PROFILE_SAMPLE_LIMIT
) -> list[ProfileSample]:
    safe_limit = max(1, int(limit))
    rows = conn.execute(
        """
        SELECT msg_id, thread_id, internal_ts, date_iso, subject, snippet, body_text, from_addr, to_addr
        FROM messages
        WHERE lower(from_addr) = lower(?) OR lower(to_addr) = lower(?)
        ORDER BY COALESCE(internal_ts, 0) DESC
        LIMIT ?
        """,
        (email, email, safe_limit),
    ).fetchall()
    return [_row_to_sample(row) for row in rows]


def _sample_char_estimate(sample: ProfileSample) -> int:
    return (
        len(sample.get("subject", ""))
        + len(sample.get("snippet", ""))
        + len(sample.get("body_text", ""))
    )


def _is_target_focused_sample(sample: ProfileSample, *, contact_email: str) -> bool:
    contact = (contact_email or "").strip().lower()
    from_addr = str(sample.get("from_addr") or "").strip().lower()
    to_addr = str(sample.get("to_addr") or "").strip().lower()
    return bool(contact and (from_addr == contact or to_addr == contact))


def _sample_relevance_score(sample: ProfileSample, *, contact_email: str) -> int:
    score = 0
    if _is_target_focused_sample(sample, contact_email=contact_email):
        score += 20
    if str(sample.get("context_source") or "") == "direct":
        score += 6
    if str(sample.get("from_addr") or "").strip().lower() == (contact_email or "").strip().lower():
        score += 4
    if str(sample.get("to_addr") or "").strip().lower() == (contact_email or "").strip().lower():
        score += 3
    if str(sample.get("body_text") or "").strip():
        score += 1
    return score


def _rank_samples_for_profile(
    samples: list[ProfileSample], *, contact_email: str
) -> list[ProfileSample]:
    ranked = sorted(
        samples,
        key=lambda s: (
            _sample_relevance_score(s, contact_email=contact_email),
            int(s.get("internal_ts") or 0),
            str(s.get("msg_id") or ""),
        ),
        reverse=True,
    )
    return ranked


def _trim_ranked_samples(
    *, samples: list[ProfileSample], max_messages: int, max_chars: int, contact_email: str
) -> list[ProfileSample]:
    safe_max_messages = max(1, int(max_messages))
    safe_max_chars = max(500, int(max_chars))

    selected: list[ProfileSample] = []
    seen_msg_ids: set[str] = set()
    chars_used = 0

    for sample in _rank_samples_for_profile(samples, contact_email=contact_email):
        msg_id = str(sample.get("msg_id") or "")
        if msg_id and msg_id in seen_msg_ids:
            continue
        char_cost = _sample_char_estimate(sample)
        if selected and chars_used + char_cost > safe_max_chars:
            continue
        selected.append(sample)
        chars_used += char_cost
        if msg_id:
            seen_msg_ids.add(msg_id)
        if len(selected) >= safe_max_messages:
            break

    if not selected and samples:
        selected.append(samples[0])
    return selected


def _expand_deep_context_samples(
    conn,
    *,
    contact_email: str,
    direct_samples: list[ProfileSample],
    max_threads: int,
    max_messages: int,
    max_chars: int,
) -> list[ProfileSample]:
    if not direct_samples:
        return []

    safe_max_threads = max(1, int(max_threads))
    safe_max_messages = max(1, int(max_messages))
    safe_max_chars = max(500, int(max_chars))

    direct_ranked = _rank_samples_for_profile(direct_samples, contact_email=contact_email)
    selected = _trim_ranked_samples(
        samples=direct_ranked,
        max_messages=safe_max_messages,
        max_chars=safe_max_chars,
        contact_email=contact_email,
    )
    seen_msg_ids: set[str] = {
        str(sample.get("msg_id") or "") for sample in selected if str(sample.get("msg_id") or "")
    }
    chars_used = sum(_sample_char_estimate(sample) for sample in selected)

    thread_ids: list[str] = []
    seen_threads: set[str] = set()
    for sample in direct_ranked:
        thread_id = str(sample.get("thread_id") or "").strip()
        if not thread_id or thread_id in seen_threads:
            continue
        thread_ids.append(thread_id)
        seen_threads.add(thread_id)
        if len(thread_ids) >= safe_max_threads:
            break

    if not thread_ids or len(selected) >= safe_max_messages or chars_used >= safe_max_chars:
        return selected

    ts_values = [
        int(sample.get("internal_ts") or 0)
        for sample in direct_samples
        if int(sample.get("internal_ts") or 0) > 0
    ]
    lower_bound = min(ts_values) if ts_values else 0
    upper_bound = max(ts_values) if ts_values else 0
    if lower_bound > 0 and upper_bound > 0:
        span = max(24 * 60 * 60 * 1000, upper_bound - lower_bound)
        lower_bound = max(0, lower_bound - span)
        upper_bound = upper_bound + span

    placeholders = ",".join("?" for _ in thread_ids)
    where = f"thread_id IN ({placeholders})"
    params: list[Any] = [*thread_ids]

    if lower_bound > 0:
        where += " AND COALESCE(internal_ts, 0) >= ?"
        params.append(lower_bound)
    if upper_bound > 0:
        where += " AND COALESCE(internal_ts, 0) <= ?"
        params.append(upper_bound)

    rows = conn.execute(
        f"""
        SELECT msg_id, thread_id, internal_ts, date_iso, subject, snippet, body_text, from_addr, to_addr
        FROM messages
        WHERE {where}
        ORDER BY COALESCE(internal_ts, 0) DESC
        LIMIT ?
        """,
        [*params, max(safe_max_messages * 4, safe_max_messages)],
    ).fetchall()

    thread_candidates: list[ProfileSample] = []
    for row in rows:
        sample = _row_to_sample(row)
        msg_id = str(sample.get("msg_id") or "")
        if msg_id and msg_id in seen_msg_ids:
            continue
        sample["context_source"] = "thread"
        sample["target_relevant"] = _is_target_focused_sample(sample, contact_email=contact_email)
        thread_candidates.append(sample)

    if not thread_candidates:
        return selected

    remaining_slots = max(0, safe_max_messages - len(selected))
    remaining_chars = max(0, safe_max_chars - chars_used)
    if remaining_slots <= 0 or remaining_chars <= 0:
        return selected

    extra = _trim_ranked_samples(
        samples=thread_candidates,
        max_messages=remaining_slots,
        max_chars=remaining_chars,
        contact_email=contact_email,
    )
    return [*selected, *extra]


def _run_gog_json(cfg: AppConfig, *, args: list[str]) -> Any:
    cmd = [cfg.profiles.gog_history_command, *args]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=max(1.0, float(cfg.profiles.gog_history_timeout_seconds)),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gog command failed rc={proc.returncode} stderr={proc.stderr.strip()}")
    payload = (proc.stdout or "").strip()
    if not payload:
        return None
    return json.loads(payload)


def _gog_message_to_sample(payload: dict[str, Any]) -> ProfileSample | None:
    if not isinstance(payload, dict):
        return None

    message = payload.get("message")
    if not isinstance(message, dict):
        return None

    headers = payload.get("headers") if isinstance(payload.get("headers"), dict) else {}
    internal_ts = _coerce_int(message.get("internalDate"))

    return {
        "msg_id": str(message.get("id") or ""),
        "thread_id": str(message.get("threadId") or ""),
        "internal_ts": internal_ts,
        "date_iso": str(headers.get("date") or ""),
        "subject": str(headers.get("subject") or ""),
        "snippet": unescape(str(message.get("snippet") or "")),
        "body_text": "",
        "from_addr": str(headers.get("from") or ""),
        "to_addr": str(headers.get("to") or ""),
        "context_source": "gog",
    }


def _augment_profile_samples_with_gog(
    cfg: AppConfig,
    *,
    email: str,
    samples: list[ProfileSample],
    max_messages: int,
    max_chars: int,
) -> tuple[list[ProfileSample], int, bool]:
    if not cfg.profiles.gog_history_enabled:
        return samples, 0, False

    safe_max_messages = max(1, int(max_messages))
    safe_max_chars = max(500, int(max_chars))

    selected = list(samples)
    chars_used = sum(_sample_char_estimate(item) for item in selected)

    remaining_messages = safe_max_messages - len(selected)
    remaining_messages = min(remaining_messages, max(1, int(cfg.profiles.gog_history_max_messages)))
    if remaining_messages <= 0:
        return selected, 0, False

    remaining_chars = safe_max_chars - chars_used
    if remaining_chars <= 0:
        return selected, 0, False

    account = (
        cfg.profiles.gog_history_account or (cfg.accounts[0].email if cfg.accounts else "")
    ).strip()
    if not account:
        return selected, 0, False

    existing_ids = {
        str(item.get("msg_id") or "") for item in selected if str(item.get("msg_id") or "")
    }
    query = f"from:{email} OR to:{email}"
    search_cap = min(
        _GOG_SEARCH_CANDIDATE_CAP,
        max(remaining_messages * _GOG_SEARCH_CANDIDATE_MULTIPLIER, remaining_messages),
    )

    try:
        search_rows = _run_gog_json(
            cfg,
            args=[
                "-a",
                account,
                "-j",
                "--results-only",
                "gmail",
                "messages",
                "search",
                query,
                "--max",
                str(search_cap),
            ],
        )
    except Exception:
        LOG.warning("gog profile context search failed for contact=%s", email, exc_info=True)
        return selected, 0, True

    if not isinstance(search_rows, list) or not search_rows:
        return selected, 0, False

    added = 0
    for row in search_rows:
        if not isinstance(row, dict):
            continue
        msg_id = str(row.get("id") or "").strip()
        if not msg_id or msg_id in existing_ids:
            continue

        try:
            message_payload = _run_gog_json(
                cfg,
                args=[
                    "-a",
                    account,
                    "-j",
                    "--results-only",
                    "gmail",
                    "get",
                    msg_id,
                    "--format",
                    "metadata",
                    "--headers",
                    "Subject,From,To,Date",
                ],
            )
        except Exception:
            LOG.warning(
                "gog profile context get failed for contact=%s msg_id=%s",
                email,
                msg_id,
                exc_info=True,
            )
            continue

        sample = _gog_message_to_sample(message_payload)
        if not sample:
            continue

        char_cost = _sample_char_estimate(sample)
        if selected and chars_used + char_cost > safe_max_chars:
            break

        selected.append(sample)
        chars_used += char_cost
        existing_ids.add(msg_id)
        added += 1

        if added >= remaining_messages or len(selected) >= safe_max_messages:
            break

    return selected, added, False


def build_profile_context_samples(
    conn,
    *,
    email: str,
    tier: str,
    max_threads: int,
    max_messages: int,
    max_chars: int,
) -> list[ProfileSample]:
    direct = _fetch_direct_profile_samples(conn, email=email, limit=_DIRECT_PROFILE_SAMPLE_LIMIT)
    if tier != "deep":
        return direct
    return _expand_deep_context_samples(
        conn,
        contact_email=email,
        direct_samples=direct,
        max_threads=max_threads,
        max_messages=max_messages,
        max_chars=max_chars,
    )


def _heuristic_profile(
    email: str, stats_row, samples, *, signal: dict[str, int | str | bool]
) -> dict:
    _, display_name, message_count, first_seen, last_seen, *_rest = stats_row
    domains = sorted(
        {
            (_sample_field(s, 3, "from_addr") or _sample_field(s, 4, "to_addr")).split("@")[-1]
            for s in samples
            if (_sample_field(s, 3, "from_addr") or _sample_field(s, 4, "to_addr"))
        }
    )
    topic_cap = 3 if signal["tier"] == "quick" else 5
    topics = sorted(
        {
            _sample_field(s, 0, "subject").split(" ")[0].lower()
            for s in samples
            if _sample_field(s, 0, "subject")
        }
    )[:topic_cap]
    role = _heuristic_role(email, samples)
    relationship_prefix = (
        "low-signal email contact" if signal["tier"] == "quick" else "email contact"
    )
    return {
        "role": role,
        "common_topics": topics,
        "tone": "unknown",
        "relationship": f"{relationship_prefix} ({display_name or email})",
        "notes": (
            f"messages={message_count}, first_seen={first_seen}, last_seen={last_seen}, "
            f"domains={domains}, heuristic_role={role}, tier={signal['tier']}, "
            f"inbound={signal['inbound_count']}, outbound={signal['outbound_count']}"
        ),
    }


def _repair_profile_json(
    cfg: AppConfig,
    *,
    email: str,
    candidate: dict[str, Any],
    stats: LLMDiagnostics,
    attempts: int = 2,
) -> tuple[dict[str, Any], bool]:
    valid, _issues = validate_profile_contract(candidate)
    if valid:
        return fill_profile_defaults(candidate, email=email), False

    stats["contract_failed"] += 1
    original_output = json.dumps(candidate, ensure_ascii=False, sort_keys=True)

    for _ in range(max(1, attempts)):
        stats["repair_attempted"] += 1
        repaired = chat_json(
            cfg.llm,
            [
                {
                    "role": "system",
                    "content": "Repair malformed contact-profile JSON. Return JSON object only.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Contact email: {email}\n"
                        f"Original output:\n{original_output}\n\n"
                        f"{profile_contract_text()}\n"
                        "Rules:\n"
                        "- Keep original intent when possible.\n"
                        "- Fill all required keys.\n"
                        "- common_topics must be an array of non-empty strings.\n"
                        "- JSON object only; no markdown/prose."
                    ),
                },
            ],
            max_tokens=220,
            temperature=0.0,
        )

        if repaired is None:
            stats["parse_failed"] += 1
            continue

        repaired_valid, _ = validate_profile_contract(repaired)
        if repaired_valid:
            stats["repair_succeeded"] += 1
            return fill_profile_defaults(repaired, email=email), True

        stats["contract_failed"] += 1
        original_output = json.dumps(repaired, ensure_ascii=False, sort_keys=True)

    return fill_profile_defaults(candidate, email=email), True


def _normalize_profile_evidence(candidate: dict[str, Any] | None) -> dict[str, list[str]] | None:
    if not isinstance(candidate, dict):
        return None

    if {"role", "relationship", "notes"}.issubset(candidate.keys()):
        topics_raw = candidate.get("common_topics")
        topics = [str(item).strip() for item in topics_raw] if isinstance(topics_raw, list) else []
        relationship_cues = [
            str(candidate.get("role") or "").strip(),
            str(candidate.get("relationship") or "").strip(),
        ]
        return {
            "facts": [str(candidate.get("notes") or "").strip()]
            if str(candidate.get("notes") or "").strip()
            else [],
            "topics": [item for item in topics if item][:10],
            "relationship_cues": [item for item in relationship_cues if item][:10],
            "tone_cues": [str(candidate.get("tone") or "").strip()]
            if str(candidate.get("tone") or "").strip()
            else [],
        }

    out: dict[str, list[str]] = {}
    for key in ("facts", "topics", "relationship_cues", "tone_cues"):
        raw = candidate.get(key)
        if isinstance(raw, list):
            values = [str(item).strip() for item in raw if str(item).strip()]
        elif isinstance(raw, str):
            values = [raw.strip()] if raw.strip() else []
        else:
            values = []
        out[key] = values[:10]

    if not any(out.values()):
        return None
    return out


def _heuristic_evidence_from_samples(*, email: str, samples) -> dict[str, list[str]] | None:
    topic_candidates: list[str] = []
    fact_candidates: list[str] = []
    for sample in samples[:12]:
        subject = _sample_field(sample, 0, "subject").strip()
        snippet = _sample_field(sample, 1, "snippet").strip()
        if subject:
            topic_candidates.append(subject.split(" ")[0].lower())
        if snippet:
            fact_candidates.append(snippet[:140])

    topics = [item for item in dict.fromkeys(topic_candidates) if item][:6]
    facts = [item for item in dict.fromkeys(fact_candidates) if item][:6]
    relationship_cues = [f"contact={email}"]
    tone_cues = ["unknown"]
    if not topics and not facts:
        return None
    return {
        "facts": facts,
        "topics": topics,
        "relationship_cues": relationship_cues,
        "tone_cues": tone_cues,
    }


def _profile_with_llm_retry(
    cfg: AppConfig,
    *,
    email: str,
    samples,
    stats: LLMDiagnostics,
    max_samples: int,
    sample_chars: int,
) -> tuple[dict | None, bool]:
    fallback_used = False
    stats["attempted"] += 1

    prompt_budget = int(cfg.profiles.deep_prompt_budget_chars)
    evidence: dict[str, list[str]] | None = None

    stats["stepA_attempted"] += 1
    try:
        evidence_candidate = chat_json(
            cfg.llm,
            build_profile_evidence_messages(
                contact_email=email,
                samples=samples,
                max_samples=max_samples,
                sample_chars=sample_chars,
                prompt_budget_chars=prompt_budget,
            ),
            max_tokens=260,
            temperature=0.0,
        )
        evidence = _normalize_profile_evidence(evidence_candidate)
        if evidence is None:
            stats["parse_failed"] += 1
            fallback_used = True
    except requests.RequestException:
        LOG.exception("LLM profile stepA HTTP failure for contact=%s", email)
        stats["http_failed"] += 1
        fallback_used = True
    except Exception:
        LOG.exception("LLM profile stepA parse/format failure for contact=%s", email)
        stats["parse_failed"] += 1
        fallback_used = True

    if evidence is None:
        retry_samples = min(max_samples, 8)
        retry_chars = min(sample_chars, 220)
        try:
            evidence_retry = chat_json(
                cfg.llm,
                build_profile_evidence_messages(
                    contact_email=email,
                    samples=samples,
                    max_samples=retry_samples,
                    sample_chars=retry_chars,
                    prompt_budget_chars=max(1200, int(prompt_budget * 0.7)),
                ),
                max_tokens=220,
                temperature=0.0,
            )
            evidence = _normalize_profile_evidence(evidence_retry)
            if evidence is None:
                stats["parse_failed"] += 1
                evidence = _heuristic_evidence_from_samples(email=email, samples=samples)
                if evidence is None:
                    stats["stepA_failed"] += 1
                    return None, True
                stats["stepA_failed"] += 1
                fallback_used = True
            else:
                fallback_used = True
        except requests.RequestException:
            LOG.exception("LLM profile stepA HTTP failure (compact retry) for contact=%s", email)
            stats["http_failed"] += 1
            evidence = _heuristic_evidence_from_samples(email=email, samples=samples)
            if evidence is None:
                stats["stepA_failed"] += 1
                return None, True
            stats["stepA_failed"] += 1
            fallback_used = True
        except Exception:
            LOG.exception(
                "LLM profile stepA parse/format failure (compact retry) for contact=%s", email
            )
            stats["parse_failed"] += 1
            evidence = _heuristic_evidence_from_samples(email=email, samples=samples)
            if evidence is None:
                stats["stepA_failed"] += 1
                return None, True
            stats["stepA_failed"] += 1
            fallback_used = True

    stats["stepA_succeeded"] += 1

    stats["stepB_attempted"] += 1
    evidence_json = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    try:
        step_b = chat_json(
            cfg.llm,
            [
                {
                    "role": "system",
                    "content": "Synthesize final contact profile JSON from compact evidence. Return JSON object only.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Contact email: {email}\n"
                        f"Evidence JSON:\n{evidence_json}\n\n"
                        f"{profile_contract_text()}\n"
                        "Rules:\n"
                        "- Use evidence only; do not invent outside facts.\n"
                        "- Keep notes concise and useful for inbox triage.\n"
                        "- JSON object only; no markdown/prose."
                    ),
                },
            ],
            max_tokens=320,
            temperature=0.0,
        )
        if step_b is None:
            stats["parse_failed"] += 1
            fallback_used = True
        else:
            repaired_profile, repaired = _repair_profile_json(
                cfg, email=email, candidate=step_b, stats=stats
            )
            if repaired:
                fallback_used = True
            stats["stepB_succeeded"] += 1
            return repaired_profile, fallback_used
    except requests.RequestException:
        LOG.exception("LLM profile stepB HTTP failure for contact=%s", email)
        stats["http_failed"] += 1
        fallback_used = True
    except Exception:
        LOG.exception("LLM profile stepB parse/format failure for contact=%s", email)
        stats["parse_failed"] += 1
        fallback_used = True

    try:
        step_b_retry = chat_json(
            cfg.llm,
            build_profile_messages(
                contact_email=email,
                samples=samples,
                max_samples=min(max_samples, 6),
                sample_chars=min(sample_chars, 220),
                prompt_budget_chars=max(1200, int(prompt_budget * 0.6)),
            ),
            max_tokens=240,
            temperature=0.0,
        )
        if step_b_retry is not None:
            repaired_profile, _repaired = _repair_profile_json(
                cfg, email=email, candidate=step_b_retry, stats=stats
            )
            stats["stepB_succeeded"] += 1
            return repaired_profile, True
        stats["parse_failed"] += 1
        stats["stepB_failed"] += 1
        return None, True
    except requests.RequestException:
        LOG.exception("LLM profile stepB HTTP failure (compact retry) for contact=%s", email)
        stats["http_failed"] += 1
        stats["stepB_failed"] += 1
        return None, True
    except Exception:
        LOG.exception(
            "LLM profile stepB parse/format failure (compact retry) for contact=%s", email
        )
        stats["parse_failed"] += 1
        stats["stepB_failed"] += 1
        return None, True


def _parse_existing_profile(profile_json: str | None) -> dict[str, object]:
    if not profile_json:
        return {}
    try:
        value = json.loads(profile_json)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _normalize_existing_meta(
    existing_profile: dict[str, object], existing_model: str | None
) -> dict[str, object]:
    raw_meta = existing_profile.get("_meta")
    meta = raw_meta if isinstance(raw_meta, dict) else {}
    source = str(meta.get("source") or ("llm" if existing_model else "heuristic"))
    tier = str(meta.get("tier") or ("deep" if source == "llm" else "quick"))
    try:
        evidence = int(meta.get("evidence_count") or 0)
    except (TypeError, ValueError):
        evidence = 0
    return {"source": source, "tier": tier, "evidence_count": evidence}


def _has_meaningful_new_evidence(
    *, old_evidence: int, new_evidence: int, old_tier: str, new_tier: str
) -> bool:
    if old_evidence <= 0:
        return True
    if new_tier == "deep" and old_tier != "deep":
        return True
    if new_evidence <= old_evidence:
        return False
    required_delta = max(
        _SIGNIFICANT_EVIDENCE_GROWTH_ABS,
        int(math.ceil(old_evidence * _SIGNIFICANT_EVIDENCE_GROWTH_REL)),
    )
    return (new_evidence - old_evidence) >= required_delta


def _should_block_llm_to_heuristic_replacement(
    *,
    existing_meta: dict[str, object],
    candidate_tier: str,
    candidate_evidence: int,
    meaningful_new_evidence: bool,
) -> bool:
    """Return True when replacing an LLM profile with heuristic output would be a demotion.

    Policy default: keep existing LLM quality, especially against quick-tier heuristic rewrites.
    Allow replacement only when the heuristic candidate is deep-tier and carries equal-or-better
    evidence with meaningful growth.
    """

    if str(existing_meta.get("source")) != "llm":
        return False

    if candidate_tier != "deep":
        return True

    existing_tier = str(existing_meta.get("tier") or "deep")
    try:
        old_evidence = int(existing_meta.get("evidence_count") or 0)
    except (TypeError, ValueError):
        old_evidence = 0

    if existing_tier == "deep" and not meaningful_new_evidence:
        return True

    return old_evidence > 0 and candidate_evidence < old_evidence


def _with_profile_meta(
    profile: dict,
    *,
    source: str,
    tier: str,
    evidence_count: int,
    message_count_at_build: int,
) -> dict:
    payload = dict(profile)
    payload["_meta"] = {
        "evidence_count": int(evidence_count),
        "tier": tier,
        "source": source,
        "message_count_at_build": int(message_count_at_build),
        "updated_at": utc_now(),
    }
    return payload


def build_profiles(
    conn,
    cfg: AppConfig,
    use_llm: bool = False,
    limit: int = 200,
    diagnostics: LLMDiagnostics | None = None,
) -> int:
    rows = profile_candidates(conn)[: max(1, int(limit))]
    updated = 0
    stats = _empty_diag()
    user_emails = [account.email.lower() for account in cfg.accounts]

    for row in rows:
        email = row[0]
        message_count = int(row[2] or 0)
        existing_profile_json = row[5] if len(row) > 5 else None
        existing_model = row[6] if len(row) > 6 else None

        direct_samples = _fetch_direct_profile_samples(
            conn, email=email, limit=_DIRECT_PROFILE_SAMPLE_LIMIT
        )
        if not direct_samples:
            continue

        inbound_count, outbound_count = contact_directional_counts(
            conn, email=email, user_emails=user_emails
        )
        signal = _compute_signal(
            email=email,
            message_count=message_count,
            inbound_count=inbound_count,
            outbound_count=outbound_count,
            samples=direct_samples,
        )

        should_try_llm = bool(use_llm and cfg.llm.enabled and signal["tier"] == "deep")
        if use_llm and cfg.llm.enabled and signal["tier"] != "deep":
            stats["llm_skipped_quick_tier"] += 1

        existing_profile = _parse_existing_profile(existing_profile_json)
        existing_meta = _normalize_existing_meta(existing_profile, existing_model)
        meaningful_new_evidence = _has_meaningful_new_evidence(
            old_evidence=int(existing_meta["evidence_count"]),
            new_evidence=int(signal["evidence_count"]),
            old_tier=str(existing_meta["tier"]),
            new_tier=str(signal["tier"]),
        )

        candidate_source = "llm" if should_try_llm else "heuristic"

        if candidate_source == "heuristic" and _should_block_llm_to_heuristic_replacement(
            existing_meta=existing_meta,
            candidate_tier=str(signal["tier"]),
            candidate_evidence=int(signal["evidence_count"]),
            meaningful_new_evidence=meaningful_new_evidence,
        ):
            stats["skipped_quality_guard"] += 1
            continue

        if existing_meta["evidence_count"] and not meaningful_new_evidence:
            stats["skipped_no_new_evidence"] += 1
            continue

        profile = _heuristic_profile(email, row, direct_samples, signal=signal)
        final_source = "heuristic"
        model = None
        fallback_used = False

        if should_try_llm:
            llm_samples = build_profile_context_samples(
                conn,
                email=email,
                tier=str(signal["tier"]),
                max_threads=cfg.profiles.deep_context_max_threads,
                max_messages=cfg.profiles.deep_context_max_messages,
                max_chars=cfg.profiles.deep_context_max_chars,
            )
            if cfg.profiles.gog_history_enabled and str(signal["tier"]) == "deep":
                stats["gog_attempted"] += 1
                llm_samples, gog_added, gog_failed = _augment_profile_samples_with_gog(
                    cfg,
                    email=email,
                    samples=llm_samples,
                    max_messages=cfg.profiles.deep_context_max_messages,
                    max_chars=cfg.profiles.deep_context_max_chars,
                )
                if gog_failed:
                    stats["gog_failed"] += 1
                if gog_added > 0:
                    stats["gog_added"] += int(gog_added)

            llm_samples = _trim_ranked_samples(
                samples=llm_samples,
                max_messages=cfg.profiles.deep_context_max_messages,
                max_chars=cfg.profiles.deep_context_max_chars,
                contact_email=email,
            )

            llm_result, used_fallback = _profile_with_llm_retry(
                cfg,
                email=email,
                samples=llm_samples,
                stats=stats,
                max_samples=max(1, min(cfg.profiles.deep_context_max_messages, len(llm_samples))),
                sample_chars=_DEEP_PROFILE_SAMPLE_CHARS,
            )
            fallback_used = used_fallback
            if llm_result:
                profile = llm_result
                final_source = "llm"
                model = cfg.llm.model
                stats["succeeded"] += 1
            else:
                fallback_used = True
                if existing_meta["source"] == "llm":
                    stats["skipped_quality_guard"] += 1
                    stats["fallback_used"] += 1
                    continue

        if fallback_used:
            stats["fallback_used"] += 1

        profile_with_meta = _with_profile_meta(
            profile,
            source=final_source,
            tier=str(signal["tier"]),
            evidence_count=int(signal["evidence_count"]),
            message_count_at_build=message_count,
        )
        upsert_contact_profile(conn, email, profile_with_meta, model=model)
        updated += 1

    conn.commit()
    if diagnostics is not None:
        diagnostics.update(stats)
    LOG.info(
        "profiles diagnostics attempted=%s succeeded=%s http_failed=%s parse_failed=%s contract_failed=%s "
        "repair_attempted=%s repair_succeeded=%s fallback_used=%s skipped_no_new_evidence=%s "
        "skipped_quality_guard=%s llm_skipped_quick_tier=%s gog_attempted=%s gog_added=%s gog_failed=%s "
        "stepA_attempted=%s stepA_succeeded=%s stepA_failed=%s stepB_attempted=%s stepB_succeeded=%s stepB_failed=%s",
        stats["attempted"],
        stats["succeeded"],
        stats["http_failed"],
        stats["parse_failed"],
        stats["contract_failed"],
        stats["repair_attempted"],
        stats["repair_succeeded"],
        stats["fallback_used"],
        stats["skipped_no_new_evidence"],
        stats["skipped_quality_guard"],
        stats["llm_skipped_quick_tier"],
        stats["gog_attempted"],
        stats["gog_added"],
        stats["gog_failed"],
        stats["stepA_attempted"],
        stats["stepA_succeeded"],
        stats["stepA_failed"],
        stats["stepB_attempted"],
        stats["stepB_succeeded"],
        stats["stepB_failed"],
    )
    return updated
