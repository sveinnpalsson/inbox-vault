from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from functools import lru_cache

from .config import LLMConfig
from .llm import chat_text, extract_first_json
from .prompts import build_redaction_messages
from .redaction_map import DeterministicRedactionMap, apply_deterministic_redaction

_RE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", flags=re.I), "EMAIL"),
    (
        re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"),
        "PHONE",
    ),
    (re.compile(r"\bhttps?://[^\s)]+", flags=re.I), "URL"),
    (
        re.compile(r"\b(?:acct|account|iban|routing|card|ssn)[:\s#-]*[A-Z0-9-]{5,}\b", flags=re.I),
        "ACCOUNT",
    ),
    (re.compile(r"\b\d{10,19}\b"), "ACCOUNT"),
]

_ALLOWED_MODES = {"regex", "model", "hybrid"}


@lru_cache(maxsize=1)
def _scrubadub_cleaner():
    try:
        import scrubadub  # type: ignore
    except Exception:
        return None
    return scrubadub


def _normalize_placeholders(text: str) -> str:
    return (
        text.replace("{{EMAIL}}", "[REDACTED_EMAIL]")
        .replace("{{PHONE}}", "[REDACTED_PHONE]")
        .replace("{{URL}}", "[REDACTED_URL]")
        .replace("{{IP}}", "[REDACTED_IP]")
    )


def regex_redact_text(text: str) -> str:
    if not text:
        return ""

    candidate = text
    scrubadub = _scrubadub_cleaner()
    if scrubadub is not None:
        try:
            candidate = str(scrubadub.clean(text))
        except Exception:
            candidate = text

    redacted = _normalize_placeholders(candidate)
    for pattern, key_name in _RE_PATTERNS:
        redacted = pattern.sub(f"[REDACTED_{key_name}]", redacted)
    return redacted


def deterministic_map_redact_text(
    text: str,
    *,
    redaction_map: DeterministicRedactionMap | None = None,
    scope_id: str | None = None,
) -> tuple[str, DeterministicRedactionMap]:
    return apply_deterministic_redaction(text, redaction_map=redaction_map, scope_id=scope_id)


def _chunk_text(text: str, *, chunk_chars: int) -> list[str]:
    safe_size = max(1, int(chunk_chars))
    return [text[i : i + safe_size] for i in range(0, len(text), safe_size)] or [""]


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def _model_redact_chunk(
    text: str,
    *,
    llm_cfg: LLMConfig,
    profile: str,
    instruction: str,
    chunk_index: int,
    chunk_total: int,
) -> str:
    output = chat_text(
        llm_cfg,
        build_redaction_messages(
            chunk_text=text,
            profile=profile,
            instruction=instruction,
            chunk_index=chunk_index,
            chunk_total=chunk_total,
        ),
        max_tokens=max(300, min(2000, len(text) * 2)),
        temperature=0.0,
    )
    return _strip_code_fences(output)


@dataclass(slots=True)
class RedactionCandidate:
    key_name: str
    value: str
    source: str


@dataclass(slots=True)
class PersistentRedactionMap:
    value_to_placeholder: dict[str, str] = field(default_factory=dict)
    placeholder_to_value: dict[str, str] = field(default_factory=dict)
    placeholder_to_key: dict[str, str] = field(default_factory=dict)
    key_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_rows(cls, rows: list[tuple[str, str, str, str]]):
        # rows: (key_name, placeholder, value_norm, original_value)
        obj = cls()
        for key_name, placeholder, value_norm, original_value in rows:
            key = _normalize_key_name(key_name)
            obj.value_to_placeholder[str(value_norm)] = str(placeholder)
            obj.placeholder_to_value[str(placeholder)] = str(original_value)
            obj.placeholder_to_key[str(placeholder)] = key
            m = re.search(r"_([A-Z]+)>$", str(placeholder))
            if m:
                idx = _alpha_token_to_int(m.group(1))
                obj.key_counts[key] = max(obj.key_counts.get(key, 0), idx)
        return obj

    def register(self, key_name: str, value: str) -> tuple[str, str, bool]:
        key = _normalize_key_name(key_name)
        normalized = _normalize_value(key, value)
        if not normalized:
            return "", "", False

        existing = self.value_to_placeholder.get(normalized)
        if existing:
            return existing, normalized, False

        next_count = self.key_counts.get(key, 0) + 1
        self.key_counts[key] = next_count
        placeholder = f"<REDACTED_{key}_{_ordinal_token(next_count)}>"
        self.value_to_placeholder[normalized] = placeholder
        self.placeholder_to_value[placeholder] = value
        self.placeholder_to_key[placeholder] = key
        return placeholder, normalized, True

    def apply(self, text: str) -> str:
        if not text:
            return ""
        out = text
        for placeholder, value in sorted(
            self.placeholder_to_value.items(), key=lambda item: len(item[1]), reverse=True
        ):
            pattern = re.compile(re.escape(value), flags=re.I)
            out = pattern.sub(placeholder, out)
            out = _replace_partial_boundary(out, value, placeholder)
        return out

    def unredact(self, text: str) -> str:
        if not text:
            return ""
        out = text
        for placeholder in sorted(self.placeholder_to_value, key=len, reverse=True):
            out = out.replace(placeholder, self.placeholder_to_value[placeholder])
        return out


@dataclass(slots=True)
class RedactionRunResult:
    source_text_redacted: str
    chunk_text_redacted: list[str]
    inserted_entries: list[dict[str, str]]


def _normalize_key_name(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", (raw or "").strip().upper()).strip("_")
    return cleaned or "CUSTOM"


def _normalize_value(key_name: str, value: str) -> str:
    stripped = (value or "").strip()
    if not stripped:
        return ""
    if key_name in {"PHONE"}:
        digits = re.sub(r"\D", "", stripped)
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        return digits
    if key_name in {"URL", "EMAIL"}:
        return stripped.lower().rstrip("/")
    return re.sub(r"\s+", " ", stripped).lower()


def _ordinal_token(n: int) -> str:
    chars: list[str] = []
    x = n
    while x > 0:
        x -= 1
        chars.append(chr(ord("A") + (x % 26)))
        x //= 26
    return "".join(reversed(chars))


def _alpha_token_to_int(token: str) -> int:
    out = 0
    for char in token:
        if not ("A" <= char <= "Z"):
            return 0
        out = (out * 26) + (ord(char) - ord("A") + 1)
    return out


def _replace_partial_boundary(text: str, value: str, placeholder: str) -> str:
    if not text or not value:
        return text

    target = value.lower()
    best = min(4, len(target) - 1) if len(target) > 1 else 0
    if best <= 0:
        return text

    lowered = text.lower()
    for k in range(len(target) - 1, best - 1, -1):
        prefix = target[:k]
        if lowered.endswith(prefix):
            text = text[: len(text) - k] + placeholder
            lowered = text.lower()
            break

    for k in range(len(target) - 1, best - 1, -1):
        suffix = target[len(target) - k :]
        if lowered.startswith(suffix):
            text = placeholder + text[k:]
            break
    return text


def _regex_detect_candidates(text: str) -> list[RedactionCandidate]:
    out: list[RedactionCandidate] = []
    for pattern, key_name in _RE_PATTERNS:
        for match in pattern.finditer(text or ""):
            value = match.group(0).strip()
            if value:
                out.append(RedactionCandidate(key_name=key_name, value=value, source="regex"))
    return out


def _parse_llm_redaction_json(raw_text: str) -> list[tuple[str, str]]:
    cleaned = _strip_code_fences(raw_text)
    parsed = extract_first_json(cleaned)
    if not isinstance(parsed, dict):
        return []

    items = parsed.get("redactions")
    if not isinstance(items, list):
        return []

    out: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        key_name = str(item.get("key_name") or item.get("placeholder_key") or "CUSTOM").strip()
        values = item.get("values")
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            continue
        for value in values:
            sval = str(value).strip()
            if sval:
                out.append((key_name, sval))
    return out


def _model_detect_candidates(
    text: str,
    *,
    llm_cfg: LLMConfig,
    profile: str,
    instruction: str,
    chunk_index: int,
    chunk_total: int,
    source: str,
) -> list[RedactionCandidate]:
    output = chat_text(
        llm_cfg,
        build_redaction_messages(
            chunk_text=text,
            profile=profile,
            instruction=instruction,
            chunk_index=chunk_index,
            chunk_total=chunk_total,
        ),
        max_tokens=max(300, min(1800, len(text) * 2)),
        temperature=0.0,
    )
    parsed = _parse_llm_redaction_json(output)
    return [RedactionCandidate(key_name=k, value=v, source=source) for k, v in parsed]


def redact_with_persistent_map(
    source_text: str,
    *,
    chunks: list[str],
    mode: str,
    llm_cfg: LLMConfig | None,
    profile: str,
    instruction: str,
    table: PersistentRedactionMap,
) -> RedactionRunResult:
    selected_mode = (mode or "hybrid").strip().lower()
    if selected_mode not in _ALLOWED_MODES:
        selected_mode = "hybrid"

    new_entries: list[dict[str, str]] = []

    def _register(candidates: list[RedactionCandidate]):
        for cand in candidates:
            placeholder, value_norm, is_new = table.register(cand.key_name, cand.value)
            if placeholder and is_new:
                new_entries.append(
                    {
                        "key_name": _normalize_key_name(cand.key_name),
                        "placeholder": placeholder,
                        "value_norm": value_norm,
                        "original_value": cand.value,
                        "source_mode": cand.source,
                    }
                )

    _register(_regex_detect_candidates(source_text))
    for chunk_text in chunks:
        _register(_regex_detect_candidates(chunk_text))

    if selected_mode in {"model", "hybrid"} and llm_cfg is not None:
        try:
            _register(
                _model_detect_candidates(
                    source_text,
                    llm_cfg=llm_cfg,
                    profile=profile,
                    instruction=instruction,
                    chunk_index=1,
                    chunk_total=1,
                    source="llm_document",
                )
            )
        except Exception:
            pass

        chunk_total = max(1, len(chunks))
        for idx, chunk_text in enumerate(chunks, start=1):
            try:
                _register(
                    _model_detect_candidates(
                        chunk_text,
                        llm_cfg=llm_cfg,
                        profile=profile,
                        instruction=instruction,
                        chunk_index=idx,
                        chunk_total=chunk_total,
                        source="llm_chunk",
                    )
                )
            except Exception:
                continue

    source_redacted = table.apply(source_text)
    chunk_redacted = [table.apply(chunk_text) for chunk_text in chunks]

    if selected_mode == "hybrid":
        source_redacted = regex_redact_text(source_redacted)
        chunk_redacted = [regex_redact_text(item) for item in chunk_redacted]
    elif selected_mode == "regex":
        source_redacted = regex_redact_text(source_redacted)
        chunk_redacted = [regex_redact_text(item) for item in chunk_redacted]

    return RedactionRunResult(
        source_text_redacted=source_redacted,
        chunk_text_redacted=chunk_redacted,
        inserted_entries=new_entries,
    )


def model_redact_text(
    text: str,
    *,
    llm_cfg: LLMConfig,
    profile: str,
    instruction: str,
    chunk_chars: int,
    model: str | None = None,
) -> str:
    if not text:
        return ""

    cfg = replace(llm_cfg, model=model) if model else llm_cfg
    chunks = _chunk_text(text, chunk_chars=chunk_chars)
    out_chunks: list[str] = []
    chunk_total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        out_chunks.append(
            _model_redact_chunk(
                chunk,
                llm_cfg=cfg,
                profile=profile,
                instruction=instruction,
                chunk_index=index,
                chunk_total=chunk_total,
            )
        )
    return "".join(out_chunks)


def redact_text(
    text: str,
    *,
    mode: str = "regex",
    llm_cfg: LLMConfig | None = None,
    profile: str = "standard",
    instruction: str = "",
    chunk_chars: int = 1200,
    model: str | None = None,
) -> str:
    if not text:
        return ""

    selected_mode = (mode or "regex").strip().lower()
    if selected_mode not in _ALLOWED_MODES:
        raise ValueError(f"Unsupported redaction mode: {mode}")

    if selected_mode == "regex":
        return regex_redact_text(text)

    if llm_cfg is None:
        return regex_redact_text(text)

    try:
        model_redacted = model_redact_text(
            text,
            llm_cfg=llm_cfg,
            profile=profile,
            instruction=instruction,
            chunk_chars=chunk_chars,
            model=model,
        )
    except Exception:
        model_redacted = ""

    if not model_redacted:
        return regex_redact_text(text)
    if selected_mode == "hybrid":
        return regex_redact_text(model_redacted)
    return model_redacted
