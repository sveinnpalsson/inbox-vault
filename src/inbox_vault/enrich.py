from __future__ import annotations

import json
import logging
from typing import Any

import requests

from .config import AppConfig
from .db import enrichment_repair_candidates, upsert_enrichment
from .json_contracts import enrich_contract_text, fill_enrich_defaults, validate_enrich_contract
from .llm import chat_json
from .prompts import build_enrichment_messages

LOG = logging.getLogger(__name__)


LLMDiagnostics = dict[str, int]


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
    }


def _short_text(value: str | None, *, max_chars: int = 140) -> str:
    text = (value or "").replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 1)].rstrip() + "…"


def _heuristic_enrichment(subject: str, snippet: str, body_text: str) -> dict[str, object]:
    haystack = " ".join([subject or "", snippet or "", body_text or ""]).lower()

    category = "general"
    action = "review"
    importance = 4

    if any(
        k in haystack
        for k in ("rent", "lease", "landlord", "tenant", "apartment", "property manager")
    ):
        category = "housing"
        action = "reply"
        importance = 9
    elif any(k in haystack for k in ("invoice", "payment", "billing", "statement", "due")):
        category = "billing"
        action = "review"
        importance = 7
    elif any(k in haystack for k in ("verify", "security", "password", "alert", "suspicious")):
        category = "security"
        action = "review"
        importance = 8
    elif any(k in haystack for k in ("meeting", "schedule", "calendar", "appointment")):
        category = "scheduling"
        action = "schedule"
        importance = 6

    evidence = _short_text(snippet) or _short_text(subject) or "Email triage fallback summary"
    summary = f"{_short_text(subject) or 'Email update'} — {evidence}".strip(" —")

    return {
        "category": category,
        "importance": importance,
        "action": action,
        "summary": _short_text(summary, max_chars=180),
    }


def _call_enrichment_llm(
    cfg: AppConfig,
    *,
    subject: str,
    snippet: str,
    body_text: str,
    from_addr: str,
    to_addr: str,
    date_iso: str,
    compact: bool,
) -> dict | None:
    body_max = 1200 if compact else 3000
    max_tokens = 180 if compact else 260
    return chat_json(
        cfg.llm,
        build_enrichment_messages(
            subject=subject,
            snippet=snippet,
            body_text=(snippet if compact else body_text),
            from_addr=from_addr,
            to_addr=to_addr,
            date_iso=date_iso,
            body_max_chars=body_max,
            compact=compact,
        ),
        max_tokens=max_tokens,
        temperature=0.0,
    )


def _repair_enrichment_json(
    cfg: AppConfig,
    *,
    candidate: dict[str, Any],
    subject: str,
    snippet: str,
    stats: LLMDiagnostics,
    attempts: int = 2,
) -> tuple[dict[str, Any], bool]:
    valid, _issues = validate_enrich_contract(candidate)
    if valid:
        return fill_enrich_defaults(candidate, subject=subject, snippet=snippet), False

    stats["contract_failed"] += 1

    original_output = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
    for _ in range(max(1, attempts)):
        stats["repair_attempted"] += 1
        repaired = chat_json(
            cfg.llm,
            [
                {
                    "role": "system",
                    "content": (
                        "Repair malformed JSON output. "
                        "Return exactly one corrected JSON object and nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original output:\n{original_output}\n\n"
                        f"{enrich_contract_text()}\n"
                        "Rules:\n"
                        "- Keep existing intent when possible.\n"
                        "- Fill missing required keys.\n"
                        "- importance must be an integer 1..10.\n"
                        "- JSON object only; no markdown/prose."
                    ),
                },
            ],
            max_tokens=180,
            temperature=0.0,
        )
        if repaired is None:
            stats["parse_failed"] += 1
            continue

        repaired_valid, _ = validate_enrich_contract(repaired)
        if repaired_valid:
            stats["repair_succeeded"] += 1
            return fill_enrich_defaults(repaired, subject=subject, snippet=snippet), True

        stats["contract_failed"] += 1
        original_output = json.dumps(repaired, ensure_ascii=False, sort_keys=True)

    return fill_enrich_defaults(candidate, subject=subject, snippet=snippet), True


def enrich_pending(
    conn,
    cfg: AppConfig,
    limit: int = 200,
    diagnostics: LLMDiagnostics | None = None,
    *,
    include_degraded: bool = False,
    progress_callback=None,
) -> int:
    stats = _empty_diag()
    if not cfg.llm.enabled:
        if diagnostics is not None:
            diagnostics.update(stats)
        return 0

    rows = enrichment_repair_candidates(conn, limit=limit, include_degraded=include_degraded)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "stage",
                "stage": "enrich_start",
                "total": len(rows),
                "include_degraded": include_degraded,
            }
        )
    count = 0
    for msg_id, subject, snippet, body_text, from_addr, to_addr, date_iso in rows:
        stats["attempted"] += 1
        fallback_path_used = False
        result: dict[str, Any] | None = None

        try:
            candidate = _call_enrichment_llm(
                cfg,
                subject=subject,
                snippet=snippet,
                body_text=body_text,
                from_addr=from_addr,
                to_addr=to_addr,
                date_iso=date_iso,
                compact=False,
            )
            if candidate is None:
                stats["parse_failed"] += 1
                fallback_path_used = True
            else:
                result, repaired = _repair_enrichment_json(
                    cfg,
                    candidate=candidate,
                    subject=subject or "",
                    snippet=snippet or "",
                    stats=stats,
                )
                if repaired:
                    fallback_path_used = True
        except requests.RequestException:
            stats["http_failed"] += 1
            fallback_path_used = True
            LOG.exception("LLM enrichment HTTP failure (primary) for message_id=%s", msg_id)
        except Exception:
            stats["parse_failed"] += 1
            fallback_path_used = True
            LOG.exception("LLM enrichment parse/format failure (primary) for message_id=%s", msg_id)

        if not result:
            try:
                candidate = _call_enrichment_llm(
                    cfg,
                    subject=subject,
                    snippet=snippet,
                    body_text=body_text,
                    from_addr=from_addr,
                    to_addr=to_addr,
                    date_iso=date_iso,
                    compact=True,
                )
                fallback_path_used = True
                if candidate is None:
                    stats["parse_failed"] += 1
                else:
                    result, _repaired = _repair_enrichment_json(
                        cfg,
                        candidate=candidate,
                        subject=subject or "",
                        snippet=snippet or "",
                        stats=stats,
                    )
            except requests.RequestException:
                stats["http_failed"] += 1
                fallback_path_used = True
                LOG.exception(
                    "LLM enrichment HTTP failure (compact retry) for message_id=%s", msg_id
                )
            except Exception:
                stats["parse_failed"] += 1
                fallback_path_used = True
                LOG.exception(
                    "LLM enrichment parse/format failure (compact retry) for message_id=%s", msg_id
                )

        model_name = cfg.llm.model
        if not result:
            result = _heuristic_enrichment(subject or "", snippet or "", body_text or "")
            model_name = "heuristic-fallback"
            fallback_path_used = True

        upsert_enrichment(conn, msg_id, result, model_name)
        count += 1
        stats["succeeded"] += 1
        if fallback_path_used:
            stats["fallback_used"] += 1
        if progress_callback is not None and (
            count == 1 or count == len(rows) or count % 10 == 0
        ):
            progress_callback(
                {
                    "event": "progress",
                    "stage": "enrich_progress",
                    "completed": count,
                    "total": len(rows),
                    "fallback_used": stats["fallback_used"],
                    "http_failed": stats["http_failed"],
                    "parse_failed": stats["parse_failed"],
                    "contract_failed": stats["contract_failed"],
                }
            )

    conn.commit()
    if diagnostics is not None:
        diagnostics.update(stats)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "stage",
                "stage": "enrich_done",
                "updated": count,
                "total": len(rows),
                "diagnostics": dict(stats),
            }
        )
    LOG.info(
        "enrich diagnostics attempted=%s succeeded=%s http_failed=%s parse_failed=%s contract_failed=%s "
        "repair_attempted=%s repair_succeeded=%s fallback_used=%s",
        stats["attempted"],
        stats["succeeded"],
        stats["http_failed"],
        stats["parse_failed"],
        stats["contract_failed"],
        stats["repair_attempted"],
        stats["repair_succeeded"],
        stats["fallback_used"],
    )
    return count
