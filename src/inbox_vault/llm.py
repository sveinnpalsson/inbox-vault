from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from typing import Any

import requests

from .config import EmbeddingConfig, LLMConfig

LOG = logging.getLogger(__name__)


def extract_first_json(text: str) -> dict[str, Any] | None:
    in_string = False
    escaped = False
    depth = 0
    start: int | None = None

    for idx, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue

        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : idx + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(parsed, dict):
                    return parsed
                start = None

    return None


def _coerce_chat_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
                    continue
                content_value = item.get("content")
                if isinstance(content_value, str):
                    parts.append(content_value)
                    continue
        return "".join(parts)
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            text_value = value.get(key)
            if isinstance(text_value, str):
                return text_value
    return ""


def _extract_choice_text(choice: dict[str, Any], *, allow_reasoning_fallback: bool) -> str:
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        message = {}

    content_text = _coerce_chat_text(message.get("content"))
    if content_text.strip():
        return content_text

    direct_choice_text = _coerce_chat_text(choice.get("text"))
    if direct_choice_text.strip():
        return direct_choice_text

    if allow_reasoning_fallback:
        reasoning_text = _coerce_chat_text(message.get("reasoning_content"))
        if reasoning_text.strip():
            return reasoning_text

    return ""


def chat_text(
    cfg: LLMConfig,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 300,
    temperature: float = 0.2,
    response_format: dict[str, Any] | None = None,
    allow_reasoning_fallback: bool = False,
) -> str:
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Match llm-vault's local llama.cpp/Qwen behavior by disabling thinking so
        # structured output lands in content instead of reasoning-specific fields.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if response_format is not None:
        payload["response_format"] = response_format

    try:
        resp = requests.post(
            f"{cfg.endpoint.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=cfg.timeout_seconds,
        )
        resp.raise_for_status()
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 400:
            # Some OpenAI-compatible servers reject response_format and/or
            # chat_template_kwargs. Retry compatibly before giving up.
            if response_format is not None:
                payload.pop("response_format", None)
                try:
                    resp = requests.post(
                        f"{cfg.endpoint.rstrip('/')}/v1/chat/completions",
                        json=payload,
                        timeout=cfg.timeout_seconds,
                    )
                    resp.raise_for_status()
                except requests.HTTPError as exc2:
                    if exc2.response is None or exc2.response.status_code != 400:
                        raise
                    payload.pop("chat_template_kwargs", None)
                    resp = requests.post(
                        f"{cfg.endpoint.rstrip('/')}/v1/chat/completions",
                        json=payload,
                        timeout=cfg.timeout_seconds,
                    )
                    resp.raise_for_status()
            else:
                payload.pop("chat_template_kwargs", None)
                resp = requests.post(
                    f"{cfg.endpoint.rstrip('/')}/v1/chat/completions",
                    json=payload,
                    timeout=cfg.timeout_seconds,
                )
                resp.raise_for_status()
        else:
            raise

    body = resp.json()
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat completion response missing choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("chat completion response has invalid first choice")

    text = _extract_choice_text(first_choice, allow_reasoning_fallback=allow_reasoning_fallback)
    if not text.strip():
        raise ValueError("chat completion returned empty text in content/reasoning_content")
    return text


def chat_json(
    cfg: LLMConfig,
    messages: list[dict[str, str]],
    max_tokens: int = 300,
    temperature: float = 0.0,
) -> dict[str, Any] | None:
    text = chat_text(
        cfg,
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"},
        allow_reasoning_fallback=True,
    )
    parsed = extract_first_json(text)
    if parsed is not None:
        return parsed

    retry_messages = [
        *messages,
        {
            "role": "user",
            "content": (
                "Your previous reply was not valid JSON. "
                "Return exactly one JSON object that matches the JSON schema contract. "
                "No markdown, no prose, no code fences."
            ),
        },
    ]
    retry_text = chat_text(
        cfg,
        retry_messages,
        max_tokens=max_tokens,
        temperature=0.0,
        response_format={"type": "json_object"},
        allow_reasoning_fallback=True,
    )
    return extract_first_json(retry_text)


def _retryable_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _hash_fallback_embedding(text: str, dim: int) -> list[float]:
    safe_dim = max(8, int(dim))
    out = [0.0] * safe_dim
    for token in re.findall(r"\w+", text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % safe_dim
        sign = 1.0 if (digest[4] % 2 == 0) else -1.0
        out[idx] += sign

    norm = math.sqrt(sum(v * v for v in out))
    if norm == 0:
        return out
    return [v / norm for v in out]


def embedding_vector(cfg: EmbeddingConfig, text: str) -> list[float]:
    payload = {
        "model": cfg.model,
        "input": text,
    }
    attempts = max(1, int(cfg.max_retries) + 1)
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(
                f"{cfg.endpoint.rstrip('/')}/v1/embeddings",
                json=payload,
                timeout=cfg.timeout_seconds,
            )
            if _retryable_status(resp.status_code):
                raise requests.HTTPError(
                    f"retryable status from embedding endpoint: {resp.status_code}",
                    response=resp,
                )

            resp.raise_for_status()
            data = resp.json()["data"][0]["embedding"]
            return [float(v) for v in data]
        except Exception as exc:
            retryable = isinstance(exc, requests.RequestException)
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                retryable = _retryable_status(exc.response.status_code)

            if not retryable:
                last_exc = exc
                break

            last_exc = exc
            if attempt >= attempts:
                break

            backoff = min(
                float(cfg.backoff_max_seconds),
                float(cfg.backoff_base_seconds) * (2 ** (attempt - 1)),
            )
            if backoff > 0:
                time.sleep(backoff)

    if cfg.fallback == "hash":
        LOG.warning("Embedding endpoint unavailable; using hash fallback vectors")
        return _hash_fallback_embedding(text, cfg.fallback_dim)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Embedding generation failed without explicit error")
