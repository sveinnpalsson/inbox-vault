from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from .config import EmbeddingConfig, LLMConfig

LOG = logging.getLogger(__name__)
_EMBEDDING_BATCH_SIZE = 16
_EMBEDDING_BATCH_TOKEN_BUDGET = 3000
_EMBEDDING_MIN_TEXT_CHARS = 256


@dataclass(slots=True)
class _PreparedEmbeddingText:
    original_index: int
    text: str


class _RetryableEmbeddingSizeError(RuntimeError):
    def __init__(self, *, status_code: int, approx_tokens: int, batch_items: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.approx_tokens = approx_tokens
        self.batch_items = batch_items
        self.message = message


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


def _estimate_text_tokens(text: str) -> int:
    clean = str(text or "").strip()
    if not clean:
        return 1
    token_like = re.findall(r"\w+|[^\w\s]", clean)
    estimate_chars = math.ceil(len(clean) / 4)
    estimate_tokens = len(token_like)
    if " " not in clean:
        estimate_tokens = max(estimate_tokens, math.ceil(len(clean) / 1.5))
    return max(1, estimate_chars, estimate_tokens)


def _normalize_text_for_embedding(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _is_retryable_size_error(message: str) -> bool:
    lowered = str(message or "").lower()
    hints = (
        "context size has been exceeded",
        "too large to process",
        "free space in the kv cache",
        "kv cache",
        "physical batch size",
        "input is too large",
    )
    return any(hint in lowered for hint in hints)


def _shrink_text_for_retry(prepared_text: str) -> str:
    clean = _normalize_text_for_embedding(prepared_text)
    if len(clean) <= _EMBEDDING_MIN_TEXT_CHARS:
        return clean
    next_limit = max(_EMBEDDING_MIN_TEXT_CHARS, int(len(clean) * 0.6))
    clipped = clean[:next_limit].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0].rstrip() or clipped
    return clipped


def _build_embedding_batches(
    texts: list[_PreparedEmbeddingText],
    *,
    batch_size: int,
    token_budget: int,
    single_text_budget: int,
) -> list[list[_PreparedEmbeddingText]]:
    max_items = max(1, int(batch_size))
    safe_token_budget = max(_EMBEDDING_MIN_TEXT_CHARS, int(token_budget))
    safe_single_text_budget = max(_EMBEDDING_MIN_TEXT_CHARS, int(single_text_budget))
    batches: list[list[_PreparedEmbeddingText]] = []
    current: list[_PreparedEmbeddingText] = []
    current_tokens = 0
    for entry in texts:
        adjusted = entry
        token_estimate = _estimate_text_tokens(adjusted.text)
        while (
            token_estimate > min(safe_token_budget, safe_single_text_budget)
            and len(adjusted.text) > _EMBEDDING_MIN_TEXT_CHARS
        ):
            shrunk = _shrink_text_for_retry(adjusted.text)
            if shrunk == adjusted.text:
                break
            adjusted = _PreparedEmbeddingText(original_index=entry.original_index, text=shrunk)
            token_estimate = _estimate_text_tokens(adjusted.text)
        if current and (len(current) >= max_items or (current_tokens + token_estimate) > safe_token_budget):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(adjusted)
        current_tokens += token_estimate
    if current:
        batches.append(current)
    return batches


def _request_embedding_batch(
    cfg: EmbeddingConfig,
    batch: list[_PreparedEmbeddingText],
    *,
    attempts: int,
) -> list[list[float]]:
    payload = {
        "model": cfg.model,
        "input": [entry.text for entry in batch],
    }
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
            body = resp.json()
            data = body.get("data") if isinstance(body, dict) else None
            if not isinstance(data, list):
                raise RuntimeError("embedding response missing data list")
            if len(data) != len(batch):
                raise RuntimeError(
                    f"embedding response item count mismatch: expected {len(batch)} got {len(data)}"
                )
            out: list[list[float]] = []
            expected_dim: int | None = None
            for entry in data:
                embedding = entry.get("embedding") if isinstance(entry, dict) else None
                if not isinstance(embedding, list):
                    raise RuntimeError("embedding entry missing list")
                vector = [float(v) for v in embedding]
                if expected_dim is None:
                    expected_dim = len(vector)
                elif len(vector) != expected_dim:
                    raise RuntimeError(
                        f"embedding dimension mismatch in batch: expected {expected_dim} got {len(vector)}"
                    )
                out.append(vector)
            return out
        except requests.HTTPError as exc:
            response = exc.response
            response_text = ""
            if response is not None:
                try:
                    response_text = response.text
                except Exception:
                    response_text = ""
            if response is not None and _is_retryable_size_error(response_text):
                approx_tokens = sum(_estimate_text_tokens(entry.text) for entry in batch)
                raise _RetryableEmbeddingSizeError(
                    status_code=int(response.status_code),
                    approx_tokens=approx_tokens,
                    batch_items=len(batch),
                    message=(
                        f"embedding HTTP {response.status_code}: batch_items={len(batch)} "
                        f"approx_tokens={approx_tokens} {response_text[:600]}"
                    ),
                ) from exc
            last_exc = exc
            retryable = response is not None and _retryable_status(response.status_code)
        except requests.RequestException as exc:
            last_exc = exc
            retryable = True
        except Exception as exc:
            last_exc = exc
            retryable = False

        if not retryable or attempt >= attempts:
            break

        backoff = min(
            float(cfg.backoff_max_seconds),
            float(cfg.backoff_base_seconds) * (2 ** (attempt - 1)),
        )
        if backoff > 0:
            time.sleep(backoff)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Embedding batch failed without explicit error")


def embedding_vectors(cfg: EmbeddingConfig, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    prepared = [
        _PreparedEmbeddingText(original_index=idx, text=_normalize_text_for_embedding(text))
        for idx, text in enumerate(texts)
    ]
    results: list[list[float] | None] = [None] * len(prepared)
    batch_token_budget = _EMBEDDING_BATCH_TOKEN_BUDGET
    single_text_budget = _EMBEDDING_BATCH_TOKEN_BUDGET
    pending_batches = _build_embedding_batches(
        prepared,
        batch_size=_EMBEDDING_BATCH_SIZE,
        token_budget=batch_token_budget,
        single_text_budget=single_text_budget,
    )
    attempts = max(1, int(cfg.max_retries) + 1)
    expected_dim: int | None = None

    try:
        while pending_batches:
            batch = pending_batches.pop(0)
            try:
                vectors = _request_embedding_batch(cfg, batch, attempts=attempts)
            except _RetryableEmbeddingSizeError as exc:
                batch_token_budget = min(
                    batch_token_budget,
                    max(_EMBEDDING_MIN_TEXT_CHARS, int(exc.approx_tokens * 0.75)),
                )
                if len(batch) > 1:
                    midpoint = max(1, len(batch) // 2)
                    pending_batches = [batch[:midpoint], batch[midpoint:], *pending_batches]
                    continue

                entry = batch[0]
                shrunk_text = _shrink_text_for_retry(entry.text)
                if shrunk_text == entry.text:
                    raise RuntimeError(exc.message) from exc
                single_text_budget = min(
                    single_text_budget,
                    max(_EMBEDDING_MIN_TEXT_CHARS, int(_estimate_text_tokens(shrunk_text) * 1.25)),
                )
                pending_batches = [
                    [_PreparedEmbeddingText(original_index=entry.original_index, text=shrunk_text)],
                    *pending_batches,
                ]
                continue

            batch_dim = len(vectors[0]) if vectors else 0
            if batch_dim <= 0:
                raise RuntimeError("embedding endpoint returned zero-dimension vectors")
            if expected_dim is None:
                expected_dim = batch_dim
            elif batch_dim != expected_dim:
                raise RuntimeError(
                    f"embedding dimension mismatch across batches: expected {expected_dim} got {batch_dim}"
                )
            for entry, vector in zip(batch, vectors):
                results[entry.original_index] = vector
    except Exception:
        if cfg.fallback == "hash":
            LOG.warning("Embedding endpoint unavailable; using hash fallback vectors")
            return [_hash_fallback_embedding(text, cfg.fallback_dim) for text in texts]
        raise

    if any(vector is None for vector in results):
        raise RuntimeError("Embedding pipeline returned incomplete results")
    return [vector for vector in results if vector is not None]


def embedding_vector(cfg: EmbeddingConfig, text: str) -> list[float]:
    return embedding_vectors(cfg, [text])[0]
