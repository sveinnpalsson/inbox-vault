from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from functools import lru_cache

from .config import LLMConfig
from .llm import chat_text, extract_first_json
from .prompts import build_redaction_messages
from .redaction_map import DeterministicRedactionMap, apply_deterministic_redaction

REDACTION_POLICY_VERSION = "2026-03-22-precision-2"

_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", flags=re.I)
_PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")
_URL_PATTERN = re.compile(r"\bhttps?://[^\s)]+", flags=re.I)
_ACCOUNT_INLINE_PATTERN = re.compile(
    r"\b(?:acct|account|iban|routing|card|ssn)[:\s#-]*[A-Z0-9-]{5,}\b", flags=re.I
)
_LABELED_ACCOUNT_PATTERN = re.compile(
    r"\b(?:acct|account|iban|routing|card|ssn)(?:\s+(?:number|no|num))?[:#\s-]*([A-Z0-9][A-Z0-9 -]{2,}\d[A-Z0-9 -]{1,})\b",
    flags=re.I,
)
_LONG_DIGITS_PATTERN = re.compile(r"\b\d{12,19}\b")
_RE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_EMAIL_PATTERN, "EMAIL"),
    (
        _PHONE_PATTERN,
        "PHONE",
    ),
    (_URL_PATTERN, "URL"),
    (_ACCOUNT_INLINE_PATTERN, "ACCOUNT"),
    (_LONG_DIGITS_PATTERN, "ACCOUNT"),
]

_ALLOWED_MODES = {"regex", "model", "hybrid"}
_KNOWN_KEYS = {"EMAIL", "PHONE", "URL", "ACCOUNT", "PERSON", "ADDRESS", "CUSTOM"}
_GENERIC_LABELS = {
    "",
    ".",
    "..",
    "...",
    "-",
    "--",
    "---",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "unspecified",
    "redacted",
    "placeholder",
    "value",
    "field",
    "form",
    "entry",
    "text",
    "data",
    "name",
    "full name",
    "first name",
    "last name",
    "middle name",
    "surname",
    "given name",
    "company name",
    "account",
    "account number",
    "routing number",
    "card number",
    "phone",
    "phone number",
    "mobile",
    "email",
    "email address",
    "address",
    "street",
    "city",
    "state",
    "zip",
    "zip code",
    "postal code",
    "country",
    "signature",
    "dob",
    "date of birth",
}
_GENERIC_PERSON_TOKENS = {
    "name",
    "first",
    "last",
    "middle",
    "full",
    "surname",
    "given",
    "applicant",
    "insured",
    "beneficiary",
    "patient",
    "signature",
    "person",
    "people",
    "individual",
    "individuals",
}
_GENERIC_CUSTOM_TOKENS = {
    "username",
    "user",
    "userid",
    "user_id",
    "user-id",
    "handle",
    "nickname",
    "forumid",
    "forum_id",
    "forum-id",
    "customid",
    "custom_id",
    "custom-id",
    "employee",
    "employeeid",
    "employee_id",
    "employee-id",
    "speaker",
    "contact",
    "participant",
}
_WRAPPING_DELIMITER_PAIRS = {
    '"': '"',
    "'": "'",
    "`": "`",
    "[": "]",
    "{": "}",
    "(": ")",
}
_PERSON_FIELD_CONTEXT_PATTERN = re.compile(
    r"(?i)(?:first\s*name|last\s*name|full\s*name|given\s*name|middle\s*name|family\s*name|surname|lastname|firstname|fullname|givenname|middlename|familyname)"
)
_PERSON_LIST_CONTEXT_PATTERN = re.compile(
    r"(?i)(?:such\s+as|including|include|comprising(?:\s+of)?|consisting\s+of|comprised\s+of|team\s+member|expert\s+veterinarian|veterinarians|members?)"
)
_ADDRESS_FIELD_CONTEXT_PATTERN = re.compile(
    r"(?i)(?:country|building(?:\s+number)?|street|city|state|postcode|postal\s*code|zip|address)"
)
_ACCOUNT_FIELD_CONTEXT_PATTERN = re.compile(
    r"(?i)(?:account(?:\s+(?:number|no|num))?|acct(?:\s+(?:number|no|num))?|iban|routing(?:\s+number)?|card(?:\s+number)?|ssn|social\s+security(?:\s+number)?|passport(?:\s+number)?|id\s+number|id\s+card|student\s+id|application\s+id|identifier|tax\s*id|tin)"
)
_ANY_FIELD_CONTEXT_PATTERN = re.compile(
    r"(?i)(?:first\s*name|last\s*name|full\s*name|given\s*name|middle\s*name|family\s*name|surname|lastname|firstname|fullname|givenname|middlename|familyname|"
    r"country|building(?:\s+number)?|street|city|state|postcode|postal\s*code|zip|address|"
    r"email|e-mail|phone(?:\s+number)?|telephone(?:\s+number)?|contact\s+number|mobile|fax|url|website|"
    r"account(?:\s+(?:number|no|num))?|acct(?:\s+(?:number|no|num))?|iban|routing(?:\s+number)?|card(?:\s+number)?|ssn|social\s+security(?:\s+number)?|passport(?:\s+number)?|id\s+number|id\s+card|student\s+id|application\s+id|identifier|tax\s*id|tin)"
)
_ACCOUNT_PREFIX_PATTERN = re.compile(
    r"(?i)^(?:acct|account|iban|routing|card|ssn|passport|idcard|social|taxid|tax-id|tin)[\s:_#-]*[A-Z0-9]"
)
_ADDRESS_HINTS = {
    "street",
    "st",
    "avenue",
    "ave",
    "road",
    "rd",
    "drive",
    "dr",
    "lane",
    "ln",
    "boulevard",
    "blvd",
    "court",
    "ct",
    "circle",
    "cir",
    "highway",
    "hwy",
    "parkway",
    "pkwy",
    "suite",
    "ste",
    "unit",
    "apt",
    "apartment",
    "box",
    "po",
}
_STATE_OR_REGION_CODES = {
    "al",
    "ak",
    "az",
    "ar",
    "ca",
    "co",
    "ct",
    "dc",
    "de",
    "fl",
    "ga",
    "hi",
    "ia",
    "id",
    "il",
    "in",
    "ks",
    "ky",
    "la",
    "ma",
    "md",
    "me",
    "mi",
    "mn",
    "mo",
    "ms",
    "mt",
    "nc",
    "nd",
    "ne",
    "nh",
    "nj",
    "nm",
    "nv",
    "ny",
    "oh",
    "ok",
    "or",
    "pa",
    "ri",
    "sc",
    "sd",
    "tn",
    "tx",
    "ut",
    "va",
    "vt",
    "wa",
    "wi",
    "wv",
    "wy",
}
_GENERIC_ROLE_PREFIXES = {
    "agent",
    "student",
    "customer",
    "applicant",
    "presenter",
    "learner",
    "participant",
    "representative",
    "user",
    "employee",
    "member",
    "client",
    "patient",
}


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
            if not is_persistent_redaction_value_allowed(key, original_value):
                continue
            obj.value_to_placeholder[str(value_norm)] = str(placeholder)
            obj.placeholder_to_value[str(placeholder)] = str(original_value)
            obj.placeholder_to_key[str(placeholder)] = key
            m = re.search(r"_([A-Z]+)>$", str(placeholder))
            if m:
                idx = _alpha_token_to_int(m.group(1))
                obj.key_counts[key] = max(obj.key_counts.get(key, 0), idx)
        return obj

    def register(
        self,
        key_name: str,
        value: str,
        *,
        source_text: str | None = None,
    ) -> tuple[str, str, bool]:
        key = _normalize_key_name(key_name)
        if not is_redaction_value_allowed(key, value, source_text=source_text):
            return "", "", False
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
            exact_pattern = _compile_exact_value_pattern(value)
            out = exact_pattern.sub(placeholder, out)
            whitespace_pattern = _compile_whitespace_tolerant_pattern(value)
            if whitespace_pattern is not None:
                out = whitespace_pattern.sub(placeholder, out)
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
    persisted_entries: list[dict[str, str]]


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


def _normalize_candidate_display(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", (value or "").strip())
    while len(cleaned) >= 2:
        closer = _WRAPPING_DELIMITER_PAIRS.get(cleaned[0])
        if closer is None or cleaned[-1] != closer:
            break
        inner = cleaned[1:-1].strip()
        if not inner:
            break
        cleaned = inner
    return cleaned.strip()


def _looks_like_generic_label(value: str) -> bool:
    cleaned = _normalize_candidate_display(value).lower().strip(" .,:;")
    if not cleaned:
        return True
    if cleaned in _GENERIC_LABELS:
        return True
    if re.fullmatch(r"[_.\- ]+", cleaned):
        return True
    if cleaned.endswith(":") and cleaned[:-1].strip() in _GENERIC_LABELS:
        return True
    return False


def _candidate_present_in_text(key_name: str, value: str, text: str) -> bool:
    source = str(text or "")
    if not source:
        return False
    display = _normalize_candidate_display(value)
    if not display:
        return False
    if key_name in {"PHONE", "ACCOUNT"}:
        digits = re.sub(r"\D", "", display)
        if digits:
            compact_source = re.sub(r"\D", "", source)
            return digits in compact_source
    if key_name in {"EMAIL", "URL"}:
        return _normalize_value(key_name, display) in _normalize_value(key_name, source)
    return display.lower() in source.lower()


def _has_strong_person_field_context(display: str, source_text: str | None) -> bool:
    source = str(source_text or "")
    if not source or not display:
        return False
    for match in re.finditer(re.escape(display), source, flags=re.I):
        prefix = _field_local_prefix(source, match.start())
        if _has_field_local_context(prefix, _PERSON_FIELD_CONTEXT_PATTERN):
            return True
    return False


def _has_person_tabular_context(display: str, source_text: str | None) -> bool:
    source = str(source_text or "")
    if not source or not display:
        return False
    for match in re.finditer(re.escape(display), source, flags=re.I):
        line = _line_window(source, match.start(), match.end())
        quoted_tokens = re.findall(r'"([^"\n]{1,64})"', line)
        if len(quoted_tokens) >= 3 and any(display.lower() == token.strip().lower() for token in quoted_tokens):
            return True
        if line.count(",") >= 4 and re.search(
            rf'(^|[,\t])\s*"?{re.escape(display)}"?\s*($|[,\t])',
            line,
            flags=re.I,
        ):
            return True
    return False


def _has_person_list_context(display: str, source_text: str | None) -> bool:
    source = str(source_text or "")
    if not source or not display:
        return False
    for match in re.finditer(re.escape(display), source, flags=re.I):
        start = max(0, match.start() - 96)
        end = min(len(source), match.end() + 96)
        window = source[start:end]
        unique_titleish = {token.lower() for token in re.findall(r"[A-Z][A-Za-z'’-]{2,}", window)}
        if len(unique_titleish) < 2:
            continue
        if _PERSON_LIST_CONTEXT_PATTERN.search(window):
            if window.count("[") >= 2 and window.count("]") >= 2:
                return True
            if window.count(",") >= 2 and (window.count('"') >= 4 or window.count("(") >= 1):
                return True
        if window.count("[") >= 2 and window.count("]") >= 2:
            return True
        if window.count('"') >= 4 and window.count(",") >= 4 and len(unique_titleish) >= 3:
            return True
    return False


def _has_strong_address_field_context(display: str, source_text: str | None) -> bool:
    source = str(source_text or "")
    if not source or not display:
        return False
    for match in re.finditer(re.escape(display), source, flags=re.I):
        prefix = _field_local_prefix(source, match.start())
        if _has_field_local_context(prefix, _ADDRESS_FIELD_CONTEXT_PATTERN):
            return True
    return False


def _has_strong_account_field_context(display: str, source_text: str | None) -> bool:
    source = str(source_text or "")
    if not source or not display:
        return False
    for match in re.finditer(re.escape(display), source, flags=re.I):
        prefix = _field_local_prefix(source, match.start())
        if _has_field_local_context(prefix, _ACCOUNT_FIELD_CONTEXT_PATTERN):
            return True
    return False


def _line_window(source: str, start: int, end: int) -> str:
    line_start = source.rfind("\n", 0, start) + 1
    line_end = source.find("\n", end)
    if line_end < 0:
        line_end = len(source)
    return source[line_start:line_end]


def _field_local_prefix(source: str, match_start: int) -> str:
    line_start = source.rfind("\n", 0, match_start) + 1
    prefix = source[line_start:match_start]
    return re.sub(r"<[^>]+>", "", prefix)


def _has_field_local_context(prefix: str, target_pattern: re.Pattern[str]) -> bool:
    labels = list(_ANY_FIELD_CONTEXT_PATTERN.finditer(prefix))
    if not labels:
        return False
    target_labels = list(target_pattern.finditer(prefix))
    if not target_labels:
        return False
    last_label = labels[-1]
    target_label = target_labels[-1]
    if target_label.start() != last_label.start():
        return False
    trailing = prefix[target_label.end() :]
    trailing = re.sub(r"\b[A-Za-z_][A-Za-z0-9_:-]*\s*=", "", trailing)
    return bool(re.fullmatch(r'[\s:=#\-"\'\[\]\(\)<>/]*', trailing))


def _looks_like_generic_role_identifier(display: str) -> bool:
    compact = display.strip()
    if not compact:
        return False
    lower = compact.lower()
    if re.fullmatch(r"(?:" + "|".join(sorted(_GENERIC_ROLE_PREFIXES)) + r")\d{1,3}", lower):
        return True
    parts = compact.split()
    if len(parts) != 2:
        return False
    prefix = parts[0].lower()
    suffix = parts[1]
    if prefix not in _GENERIC_ROLE_PREFIXES:
        return False
    return bool(re.fullmatch(r"(?:[A-Z]|\d{1,3}|[A-Z]\d{1,3}|\d{1,3}[A-Z])", suffix))


def _is_persistent_address_value(display: str) -> bool:
    lowered = display.lower().strip(" .,:;")
    if not lowered:
        return False
    if lowered in _STATE_OR_REGION_CODES:
        return False
    if re.fullmatch(r"\d+[A-Za-z0-9-]*", display):
        return False
    if re.fullmatch(r"[A-Za-z][A-Za-z'’-]+", display):
        return False
    word_tokens = {token.lower().strip(".,") for token in re.findall(r"[A-Za-z0-9#]+", display)}
    if "po" in word_tokens and "box" in word_tokens:
        return True
    if word_tokens & _ADDRESS_HINTS:
        return True
    if "," in display and len(word_tokens) >= 2:
        return True
    return False


def _is_persistent_person_value(display: str) -> bool:
    if any(ch.isdigit() for ch in display):
        return False
    lowered = display.lower()
    if lowered in _GENERIC_LABELS or _looks_like_generic_role_identifier(display):
        return False
    if ":" in display or "@" in display or "/" in display:
        return False
    tokens = re.findall(r"[A-Za-z][A-Za-z'’-]*", display)
    if not tokens:
        return False
    if any(token.lower() in _GENERIC_PERSON_TOKENS for token in tokens):
        return False
    return sum(len(token) for token in tokens) >= 5


def is_persistent_redaction_value_allowed(
    key_name: str,
    value: str,
    *,
    source_text: str | None = None,
) -> bool:
    key = _normalize_key_name(key_name)
    display = _normalize_candidate_display(value)
    if not display:
        return False
    if key == "ADDRESS":
        return is_redaction_value_allowed(key, value, source_text=source_text) and _is_persistent_address_value(display)
    if key == "PERSON":
        return _is_persistent_person_value(display)
    return is_redaction_value_allowed(key, value, source_text=source_text)


def is_redaction_value_allowed(
    key_name: str,
    value: str,
    *,
    source_text: str | None = None,
) -> bool:
    key = _normalize_key_name(key_name)
    if key not in _KNOWN_KEYS:
        return False

    display = _normalize_candidate_display(value)
    if not display or _looks_like_generic_label(display):
        return False
    if len(display) < 3:
        if key != "ADDRESS" or not _has_strong_address_field_context(display, source_text):
            return False
    if re.fullmatch(r"[Xx*#._-]+", display):
        return False
    if source_text is not None and not _candidate_present_in_text(key, display, source_text):
        return False

    if key == "EMAIL":
        return bool(_EMAIL_PATTERN.fullmatch(display))

    if key == "PHONE":
        digits = _normalize_value(key, display)
        if not digits.isdigit():
            return False
        if len(digits) == 10:
            return digits[0] in "23456789" and digits[3] in "23456789"
        return 11 <= len(digits) <= 15

    if key == "URL":
        return bool(_URL_PATTERN.fullmatch(display))

    if key == "ACCOUNT":
        digits = re.sub(r"\D", "", display)
        compact = re.sub(r"[\s-]+", "", display)
        if len(digits) >= 6:
            return True
        if len(compact) >= 8 and sum(ch.isdigit() for ch in compact) >= 4:
            return True
        return False

    if key == "PERSON":
        if any(ch.isdigit() for ch in display):
            return False
        lowered = display.lower()
        if lowered in _GENERIC_LABELS or _looks_like_generic_role_identifier(display):
            return False
        if ":" in display or "@" in display or "/" in display:
            return False
        tokens = re.findall(r"[A-Za-z][A-Za-z'’-]*", display)
        has_context = (
            _has_strong_person_field_context(display, source_text)
            or _has_person_tabular_context(display, source_text)
            or _has_person_list_context(display, source_text)
        )
        if len(tokens) < 2 and not has_context:
            return False
        if any(token.lower() in _GENERIC_PERSON_TOKENS for token in tokens):
            return False
        if sum(len(token) for token in tokens) < 5:
            return False
        return True

    if key == "ADDRESS":
        lowered = display.lower().strip(" .,:;")
        if lowered in _GENERIC_LABELS:
            return False
        contextual = _has_strong_address_field_context(display, source_text)
        if lowered in _STATE_OR_REGION_CODES:
            return contextual
        if re.fullmatch(r"[A-Za-z]{2,3}", display):
            return contextual
        if len(display) < 8 and not contextual:
            return False
        has_digit = any(ch.isdigit() for ch in display)
        word_tokens = {token.lower().strip(".,") for token in re.findall(r"[A-Za-z0-9#]+", display)}
        if has_digit:
            return True
        if "po" in word_tokens and "box" in word_tokens:
            return True
        if word_tokens & _ADDRESS_HINTS:
            return True
        return contextual and any(ch.isalpha() for ch in display)

    if key == "CUSTOM":
        if _looks_like_generic_role_identifier(display):
            return False
        return _is_allowed_custom_value(display)

    return False


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


def _compile_exact_value_pattern(value: str) -> re.Pattern[str]:
    normalized = _normalize_candidate_display(value)
    prefix = r"(?<!\w)" if normalized and re.match(r"^\w", normalized) else ""
    suffix = r"(?!\w)" if normalized and re.search(r"\w$", normalized) else ""
    return re.compile(f"{prefix}{re.escape(value)}{suffix}", flags=re.I)


def _compile_whitespace_tolerant_pattern(value: str) -> re.Pattern[str] | None:
    normalized = _normalize_candidate_display(value)
    if not normalized or len(normalized.split()) < 2:
        return None
    if not re.search(r"\s", value):
        return None

    parts = [part for part in normalized.split(" ") if part]
    if len(parts) < 2:
        return None

    body = r"\s+".join(re.escape(part) for part in parts)
    prefix = r"(?<!\w)" if re.match(r"^\w", parts[0]) else ""
    suffix = r"(?!\w)" if re.search(r"\w$", parts[-1]) else ""
    return re.compile(f"{prefix}{body}{suffix}", flags=re.I)


def _expand_address_literals(display: str, source_text: str | None) -> list[str]:
    source = str(source_text or "")
    if not source or not display:
        return []
    if _candidate_present_in_text("ADDRESS", display, source):
        return [display]

    candidates: list[str] = []
    for raw_part in re.split(r"\s*,\s*", display):
        part = raw_part.strip()
        if not part:
            continue
        candidates.append(part)
        if not _candidate_present_in_text("ADDRESS", part, source):
            m_num = re.match(r"^(\d+[A-Za-z0-9-]*)\s+(.+)$", part)
            if m_num:
                candidates.extend([m_num.group(1), m_num.group(2).strip()])
            m_code = re.match(r"^([A-Za-z]{2,3})\s+(\d[\w -]*)$", part)
            if m_code:
                candidates.extend([m_code.group(1), m_code.group(2).strip()])

    out: list[str] = []
    seen: set[str] = set()
    for part in candidates:
        cleaned = _normalize_candidate_display(part)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        if not _candidate_present_in_text("ADDRESS", cleaned, source):
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _is_obvious_account_value(display: str) -> bool:
    if not is_redaction_value_allowed("ACCOUNT", display):
        return False

    compact = re.sub(r"\s+", "", display)
    digits = re.sub(r"\D", "", compact)
    if len(digits) < 6:
        return False

    if not any(ch.isalpha() for ch in compact):
        return True

    if _ACCOUNT_PREFIX_PATTERN.match(compact):
        return True

    if any(ch in "._@" for ch in compact):
        return False

    if any(ch.islower() for ch in compact):
        return False

    return len(compact) >= 8 and sum(ch.isdigit() for ch in compact) >= 4


def _is_allowed_custom_value(display: str) -> bool:
    compact = display.strip()
    if " " in compact:
        return False

    lowered = compact.lower()
    if lowered in _GENERIC_LABELS or lowered in _GENERIC_CUSTOM_TOKENS:
        return False

    has_alpha = any(ch.isalpha() for ch in compact)
    has_digit = any(ch.isdigit() for ch in compact)
    has_identifier_punct = any(ch in "@._-" for ch in compact)
    if not has_alpha:
        return False

    if compact.startswith("@"):
        body = compact[1:]
        if len(body) < 3:
            return False
        return bool(re.fullmatch(r"[A-Za-z0-9._-]+", body)) and any(ch.isalpha() for ch in body)

    if len(compact) < 4:
        return False

    if has_digit:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", compact):
            return False
        alpha_count = sum(ch.isalpha() for ch in compact)
        digit_count = sum(ch.isdigit() for ch in compact)
        if alpha_count < 1 or digit_count < 1:
            return False
        return (
            len(compact) >= 5
            or compact[0].isdigit()
            or any(ch.isupper() for ch in compact)
            or has_identifier_punct
        )

    if has_identifier_punct:
        token_parts = [part for part in re.split(r"[@._-]+", compact) if part]
        if len(token_parts) < 2:
            return False
        if not all(re.fullmatch(r"[A-Za-z0-9]+", part) for part in token_parts):
            return False
        if not any(any(ch.isalpha() for ch in part) for part in token_parts):
            return False
        punct_chars = {ch for ch in compact if ch in "@._-"}
        if punct_chars == {"-"} and all(re.fullmatch(r"[A-Z][a-z]+", part) for part in token_parts):
            return False
        return sum(len(part) for part in token_parts) >= 4

    if re.fullmatch(r"[A-Za-z]+", compact):
        return bool(re.search(r"[a-z][A-Z]", compact) or re.search(r"[A-Z]{2,}", compact))

    return False


def _remap_model_candidate_key_name(
    key_name: str,
    value: str,
    *,
    source_text: str | None = None,
) -> str:
    key = _normalize_key_name(key_name)
    display = _normalize_candidate_display(value)
    if not display:
        return key

    if _has_strong_account_field_context(display, source_text) and is_redaction_value_allowed(
        "ACCOUNT", display, source_text=source_text
    ):
        return "ACCOUNT"

    if key == "PERSON" and _is_allowed_custom_value(display):
        return "CUSTOM"

    if key == "CUSTOM":
        for canonical_key in ("EMAIL", "PHONE", "URL"):
            if is_redaction_value_allowed(canonical_key, display, source_text=source_text):
                return canonical_key
        if _is_obvious_account_value(display):
            return "ACCOUNT"

    return key


def _regex_detect_candidates(text: str) -> list[RedactionCandidate]:
    out: list[RedactionCandidate] = []
    occupied_spans: list[tuple[int, int]] = []

    def _overlaps(span: tuple[int, int]) -> bool:
        start, end = span
        for seen_start, seen_end in occupied_spans:
            if start < seen_end and end > seen_start:
                return True
        return False

    def _normalize_labeled_account_value(value: str) -> str:
        tokens = [token for token in re.split(r"\s+", (value or "").strip()) if token]
        kept: list[str] = []
        for token in tokens:
            if any(ch.isdigit() for ch in token):
                kept.append(token)
                continue
            break
        return " ".join(kept).strip()

    for match in _LABELED_ACCOUNT_PATTERN.finditer(text or ""):
        value = _normalize_labeled_account_value(match.group(1) or "")
        span = match.span(1)
        if value:
            out.append(RedactionCandidate(key_name="ACCOUNT", value=value, source="regex"))
            occupied_spans.append(span)

    for pattern, key_name in _RE_PATTERNS:
        for match in pattern.finditer(text or ""):
            span = match.span(0)
            if _overlaps(span):
                continue
            value = match.group(0).strip()
            mapped_key_name = key_name
            if value and key_name in {"PHONE", "ACCOUNT"}:
                mapped_key_name = _remap_model_candidate_key_name(
                    key_name,
                    value,
                    source_text=text,
                )
            if value and is_redaction_value_allowed(mapped_key_name, value, source_text=text):
                out.append(
                    RedactionCandidate(
                        key_name=mapped_key_name,
                        value=value,
                        source="regex",
                    )
                )
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
    out: list[RedactionCandidate] = []
    for key_name, value in parsed:
        display = _normalize_candidate_display(value)
        mapped_key_name = _remap_model_candidate_key_name(
            key_name,
            display,
            source_text=text,
        )
        candidate_values = [display]
        if mapped_key_name == "ADDRESS" and display and not _candidate_present_in_text("ADDRESS", display, text):
            expanded = _expand_address_literals(display, text)
            if expanded:
                candidate_values = expanded
        for candidate_value in candidate_values:
            if candidate_value and is_redaction_value_allowed(
                mapped_key_name,
                candidate_value,
                source_text=text,
            ):
                out.append(
                    RedactionCandidate(
                        key_name=mapped_key_name,
                        value=candidate_value,
                        source=source,
                    )
                )
    return out


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
    persisted_entries: list[dict[str, str]] = []
    regex_candidate_cache: dict[str, list[RedactionCandidate]] = {}
    redacted_text_cache: dict[str, str] = {}

    def _register(candidates: list[RedactionCandidate], *, candidate_text: str):
        for cand in candidates:
            key_name = cand.key_name
            if str(cand.source or "").startswith("llm"):
                key_name = _remap_model_candidate_key_name(
                    cand.key_name,
                    cand.value,
                    source_text=candidate_text,
                )
            candidate_values = [cand.value]
            if (
                str(cand.source or "").startswith("llm")
                and key_name == "ADDRESS"
                and not _candidate_present_in_text("ADDRESS", cand.value, candidate_text)
            ):
                expanded = _expand_address_literals(cand.value, candidate_text)
                if expanded:
                    candidate_values = expanded
            for candidate_value in candidate_values:
                placeholder, value_norm, is_new = table.register(
                    key_name,
                    candidate_value,
                    source_text=candidate_text if str(cand.source or "").startswith("llm") else None,
                )
                if placeholder and is_new:
                    entry = {
                        "key_name": _normalize_key_name(key_name),
                        "placeholder": placeholder,
                        "value_norm": value_norm,
                        "original_value": candidate_value,
                        "source_mode": cand.source,
                    }
                    new_entries.append(entry)
                    if is_persistent_redaction_value_allowed(
                        key_name,
                        candidate_value,
                        source_text=candidate_text,
                    ):
                        persisted_entries.append(entry)

    def _regex_candidates(text: str) -> list[RedactionCandidate]:
        cached = regex_candidate_cache.get(text)
        if cached is not None:
            return cached
        detected = _regex_detect_candidates(text)
        regex_candidate_cache[text] = detected
        return detected

    def _render_redacted(text: str) -> str:
        cached = redacted_text_cache.get(text)
        if cached is not None:
            return cached
        redacted = table.apply(text)
        if selected_mode in {"hybrid", "regex"}:
            redacted = regex_redact_text(redacted)
        redacted_text_cache[text] = redacted
        return redacted

    _register(_regex_candidates(source_text), candidate_text=source_text)
    for chunk_text in chunks:
        _register(_regex_candidates(chunk_text), candidate_text=chunk_text)

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
                ),
                candidate_text=source_text,
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
                    ),
                    candidate_text=chunk_text,
                )
            except Exception:
                continue

    source_redacted = _render_redacted(source_text)
    chunk_redacted = [_render_redacted(chunk_text) for chunk_text in chunks]

    return RedactionRunResult(
        source_text_redacted=source_redacted,
        chunk_text_redacted=chunk_redacted,
        inserted_entries=new_entries,
        persisted_entries=persisted_entries,
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
