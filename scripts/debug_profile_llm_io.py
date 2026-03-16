#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from inbox_vault.config import load_config, resolve_password
from inbox_vault.db import contact_directional_counts, get_conn
from inbox_vault.llm import extract_first_json
from inbox_vault.profiles import (
    _augment_profile_samples_with_gog,
    _compute_signal,
    _heuristic_evidence_from_samples,
    _normalize_profile_evidence,
    build_profile_context_samples,
)
from inbox_vault.prompts import build_profile_evidence_messages

# Keep these in sync with the deep-tier profile LLM path in profiles.build_profiles().
_DEEP_DEFAULT_SAMPLE_CHARS = 500


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _coerce_choice_text(choice: dict[str, Any]) -> str:
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        message = {}

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    pieces.append(text)
        merged = "".join(pieces)
        if merged.strip():
            return merged

    text_field = choice.get("text")
    if isinstance(text_field, str) and text_field.strip():
        return text_field

    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning

    return ""


def _extract_text_from_response_json(response_json: dict[str, Any] | None) -> str:
    if not isinstance(response_json, dict):
        return ""
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    return _coerce_choice_text(first)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Debug one contact profile LLM request/response with full prompt capture"
    )
    parser.add_argument("--config", default="config.toml", help="Config path")
    parser.add_argument("--contact", required=True, help="Contact email")
    parser.add_argument("--db", default=None, help="Optional DB override path")
    parser.add_argument("--key-env", default="INBOX_VAULT_DB_PASSWORD", help="Env var name for DB key")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Samples included in prompt (default from config [profiles].deep_context_max_messages)",
    )
    parser.add_argument(
        "--sample-chars",
        type=int,
        default=_DEEP_DEFAULT_SAMPLE_CHARS,
        help=f"Chars per sample in prompt (default {_DEEP_DEFAULT_SAMPLE_CHARS})",
    )
    parser.add_argument(
        "--force-llm",
        action="store_true",
        help="Force one LLM call even when quick-tier would normally skip",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.key_env:
        cfg.db.password_env = args.key_env
    if args.db:
        cfg.db.path = args.db

    db_password = resolve_password(cfg.db)
    contact = args.contact.strip().lower()

    report_dir = Path("reports") / f"profile_debug_{_timestamp_slug()}"
    report_dir.mkdir(parents=True, exist_ok=True)

    with get_conn(cfg.db.path, db_password) as conn:
        user_emails = [acct.email.lower() for acct in cfg.accounts]
        inbound_count, outbound_count = contact_directional_counts(
            conn, email=contact, user_emails=user_emails
        )
        direct_samples = build_profile_context_samples(
            conn,
            email=contact,
            tier="quick",
            max_threads=cfg.profiles.deep_context_max_threads,
            max_messages=40,
            max_chars=cfg.profiles.deep_context_max_chars,
        )
        signal = _compute_signal(
            email=contact,
            message_count=len(direct_samples),
            inbound_count=inbound_count,
            outbound_count=outbound_count,
            samples=direct_samples,
        )
        samples = build_profile_context_samples(
            conn,
            email=contact,
            tier=str(signal["tier"]),
            max_threads=cfg.profiles.deep_context_max_threads,
            max_messages=cfg.profiles.deep_context_max_messages,
            max_chars=cfg.profiles.deep_context_max_chars,
        )

    gog_added = 0
    gog_failed = False
    if signal["tier"] == "deep" and cfg.profiles.gog_history_enabled:
        samples, gog_added, gog_failed = _augment_profile_samples_with_gog(
            cfg,
            email=contact,
            samples=samples,
            max_messages=cfg.profiles.deep_context_max_messages,
            max_chars=cfg.profiles.deep_context_max_chars,
        )

    print(
        f"contact={contact} tier={signal['tier']} inbound={signal['inbound_count']} "
        f"outbound={signal['outbound_count']} evidence={signal['evidence_count']} "
        f"gog_added={gog_added} gog_failed={gog_failed}"
    )

    max_samples = (
        cfg.profiles.deep_context_max_messages
        if args.max_samples is None
        else max(1, int(args.max_samples))
    )

    evidence_messages = build_profile_evidence_messages(
        contact_email=contact,
        samples=samples,
        max_samples=max_samples,
        sample_chars=max(120, int(args.sample_chars)),
        prompt_budget_chars=cfg.profiles.deep_prompt_budget_chars,
    )

    step_a_payload: dict[str, Any] = {
        "model": cfg.llm.model,
        "messages": evidence_messages,
        "temperature": 0.0,
        "max_tokens": 260,
        "response_format": {"type": "json_object"},
    }

    should_call_llm = bool(cfg.llm.enabled and (signal["tier"] == "deep" or args.force_llm))

    step_a_response: dict[str, Any] = {
        "called": should_call_llm,
        "status_code": None,
        "ok": None,
        "headers": {},
        "json": None,
        "text": "",
        "error": None,
    }
    step_b_response: dict[str, Any] = {
        "called": False,
        "status_code": None,
        "ok": None,
        "headers": {},
        "json": None,
        "text": "",
        "error": None,
    }

    evidence_json: dict[str, Any] | None = None
    step_a_parse_success = False
    step_b_parse_success = False
    parsed_json: dict[str, Any] | None = None
    fallback_used = False
    step_b_payload: dict[str, Any] | None = None
    step_b_messages: list[dict[str, str]] = []

    if should_call_llm:
        try:
            resp = requests.post(
                f"{cfg.llm.endpoint.rstrip('/')}/v1/chat/completions",
                json=step_a_payload,
                timeout=cfg.llm.timeout_seconds,
            )
            step_a_response["status_code"] = int(resp.status_code)
            step_a_response["ok"] = bool(resp.ok)
            step_a_response["headers"] = dict(resp.headers)
            step_a_response["text"] = resp.text
            try:
                response_json = resp.json()
            except ValueError:
                response_json = None
            step_a_response["json"] = response_json

            model_text = _extract_text_from_response_json(response_json)
            parse_target = model_text if model_text.strip() else (resp.text or "")
            step_a_raw = extract_first_json(parse_target)
            evidence_json = _normalize_profile_evidence(
                step_a_raw if isinstance(step_a_raw, dict) else None
            )
            if evidence_json is None:
                evidence_json = _heuristic_evidence_from_samples(email=contact, samples=samples)
                fallback_used = True
            step_a_parse_success = evidence_json is not None
        except Exception as exc:  # broad on purpose for debug capture
            step_a_response["error"] = f"{type(exc).__name__}: {exc}"
            evidence_json = _heuristic_evidence_from_samples(email=contact, samples=samples)
            fallback_used = evidence_json is not None

        if evidence_json is not None:
            step_b_messages = [
                {
                    "role": "system",
                    "content": "Synthesize final contact profile JSON from compact evidence. Return JSON object only.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Contact email: {contact}\n"
                        f"Evidence JSON:\n{json.dumps(evidence_json, ensure_ascii=False, sort_keys=True)}\n\n"
                        "Required keys: role, common_topics (array), tone, relationship, notes.\n"
                        "Use evidence only. JSON object only."
                    ),
                },
            ]
            step_b_payload: dict[str, Any] = {
                "model": cfg.llm.model,
                "messages": step_b_messages,
                "temperature": 0.0,
                "max_tokens": 320,
                "response_format": {"type": "json_object"},
            }
            step_b_response["called"] = True
            try:
                resp_b = requests.post(
                    f"{cfg.llm.endpoint.rstrip('/')}/v1/chat/completions",
                    json=step_b_payload,
                    timeout=cfg.llm.timeout_seconds,
                )
                step_b_response["status_code"] = int(resp_b.status_code)
                step_b_response["ok"] = bool(resp_b.ok)
                step_b_response["headers"] = dict(resp_b.headers)
                step_b_response["text"] = resp_b.text
                try:
                    response_json_b = resp_b.json()
                except ValueError:
                    response_json_b = None
                step_b_response["json"] = response_json_b

                model_text_b = _extract_text_from_response_json(response_json_b)
                parse_target_b = model_text_b if model_text_b.strip() else (resp_b.text or "")
                parsed_json = extract_first_json(parse_target_b)
                step_b_parse_success = parsed_json is not None
            except Exception as exc:  # broad on purpose for debug capture
                step_b_response["error"] = f"{type(exc).__name__}: {exc}"
    else:
        step_a_response["error"] = (
            "LLM call skipped: tier=quick and --force-llm not set, "
            "matching build_profiles quick-tier behavior."
        )
        step_b_payload = None

    report = {
        "created_at": _iso_utc_now(),
        "contact": contact,
        "config_path": args.config,
        "db_path": cfg.db.path,
        "llm_enabled": bool(cfg.llm.enabled),
        "llm_model": cfg.llm.model,
        "llm_endpoint": cfg.llm.endpoint,
        "max_samples": int(max_samples),
        "sample_chars": int(args.sample_chars),
        "prompt_budget_chars": int(cfg.profiles.deep_prompt_budget_chars),
        "force_llm": bool(args.force_llm),
        "signal": signal,
        "tier": signal["tier"],
        "sample_count": len(samples),
        "gog_history_enabled": bool(cfg.profiles.gog_history_enabled),
        "gog_added_samples": int(gog_added),
        "gog_failed": bool(gog_failed),
        "stepA_prompt_user_chars": len(evidence_messages[1]["content"])
        if len(evidence_messages) > 1
        else 0,
        "stepB_prompt_user_chars": len(step_b_messages[1]["content"])
        if len(step_b_messages) > 1
        else 0,
        "stepA_parse_success": step_a_parse_success,
        "stepB_parse_success": step_b_parse_success,
        "fallback_used": fallback_used,
        "evidence_json": evidence_json,
        "parsed_json": parsed_json,
    }

    _json_dump(report_dir / "report.json", report)
    _json_dump(report_dir / "stepA_request_payload.json", step_a_payload)
    _json_dump(report_dir / "stepA_response_raw.json", step_a_response)
    _json_dump(report_dir / "stepB_request_payload.json", step_b_payload or {})
    _json_dump(report_dir / "stepB_response_raw.json", step_b_response)

    prompts = {
        "stepA_system": evidence_messages[0]["content"] if evidence_messages else "",
        "stepA_user": evidence_messages[1]["content"] if len(evidence_messages) > 1 else "",
        "stepB_system": step_b_messages[0]["content"] if step_b_messages else "",
        "stepB_user": step_b_messages[1]["content"] if len(step_b_messages) > 1 else "",
    }
    _json_dump(report_dir / "prompt_messages.json", prompts)

    print(str(report_dir / "report.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
