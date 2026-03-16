from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

RedactionCategory = Literal["EMAIL", "PHONE", "URL", "ACCOUNT"]

_URL_PATTERN = re.compile(r"\bhttps?://[^\s<>\"]+", flags=re.I)
_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", flags=re.I)
_PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})(?!\d)")
_ACCOUNT_KEYWORD_PATTERN = re.compile(
    r"\b(?:acct|account|iban|routing|card|ssn)[:\s#-]*[A-Z0-9-]{5,}\b", flags=re.I
)
_ACCOUNT_LONG_NUMBER_PATTERN = re.compile(r"\b\d{10,19}\b")

_TRAILING_URL_PUNCTUATION = ".,;:!?)]}"


@dataclass
class DeterministicRedactionMap:
    """Stable category-aware placeholder mapping for deterministic redaction.

    Reuse one instance across chunks/documents in the same scope to preserve
    coreference (same sensitive value => same placeholder).
    """

    scope_id: str | None = None
    entity_to_placeholder: dict[str, dict[str, str]] = field(default_factory=dict)
    placeholder_to_entity: dict[str, str] = field(default_factory=dict)
    placeholder_to_category: dict[str, RedactionCategory] = field(default_factory=dict)
    category_counts: dict[str, int] = field(default_factory=dict)

    def placeholder_for(self, category: RedactionCategory, value: str) -> str:
        normalized = _normalize_value(category, value)
        if not normalized:
            return value

        by_category = self.entity_to_placeholder.setdefault(category, {})
        existing = by_category.get(normalized)
        if existing:
            return existing

        next_count = self.category_counts.get(category, 0) + 1
        self.category_counts[category] = next_count
        placeholder = f"<REDACTED_{category}_{_ordinal_token(next_count)}>"

        by_category[normalized] = placeholder
        self.placeholder_to_entity[placeholder] = value
        self.placeholder_to_category[placeholder] = category
        return placeholder

    def apply(self, text: str) -> str:
        if not text:
            return ""

        redacted = text

        def _replace_url(match: re.Match[str]) -> str:
            raw = match.group(0)
            core = raw.rstrip(_TRAILING_URL_PUNCTUATION)
            if not core:
                return raw
            suffix = raw[len(core) :]
            return f"{self.placeholder_for('URL', core)}{suffix}"

        redacted = _URL_PATTERN.sub(_replace_url, redacted)
        redacted = _EMAIL_PATTERN.sub(lambda m: self.placeholder_for("EMAIL", m.group(0)), redacted)
        redacted = _PHONE_PATTERN.sub(lambda m: self.placeholder_for("PHONE", m.group(0)), redacted)
        redacted = _ACCOUNT_KEYWORD_PATTERN.sub(
            lambda m: self.placeholder_for("ACCOUNT", m.group(0)),
            redacted,
        )
        redacted = _ACCOUNT_LONG_NUMBER_PATTERN.sub(
            lambda m: self.placeholder_for("ACCOUNT", m.group(0)),
            redacted,
        )

        return redacted

    def unredact(self, text: str) -> str:
        if not text:
            return ""
        restored = text
        for placeholder in sorted(self.placeholder_to_entity, key=len, reverse=True):
            restored = restored.replace(placeholder, self.placeholder_to_entity[placeholder])
        return restored

    def to_dict(self) -> dict[str, object]:
        return {
            "scope_id": self.scope_id,
            "entity_to_placeholder": self.entity_to_placeholder,
            "placeholder_to_entity": self.placeholder_to_entity,
            "placeholder_to_category": self.placeholder_to_category,
            "category_counts": self.category_counts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "DeterministicRedactionMap":
        entity_to_placeholder_raw = data.get("entity_to_placeholder")
        placeholder_to_entity_raw = data.get("placeholder_to_entity")
        placeholder_to_category_raw = data.get("placeholder_to_category")
        category_counts_raw = data.get("category_counts")

        entity_to_placeholder: dict[str, dict[str, str]] = {}
        if isinstance(entity_to_placeholder_raw, dict):
            for category, mapping in entity_to_placeholder_raw.items():
                if isinstance(mapping, dict):
                    entity_to_placeholder[str(category)] = {
                        str(k): str(v) for k, v in mapping.items()
                    }

        placeholder_to_entity: dict[str, str] = {}
        if isinstance(placeholder_to_entity_raw, dict):
            placeholder_to_entity = {str(k): str(v) for k, v in placeholder_to_entity_raw.items()}

        placeholder_to_category: dict[str, RedactionCategory] = {}
        if isinstance(placeholder_to_category_raw, dict):
            for k, v in placeholder_to_category_raw.items():
                category = str(v)
                if category == "EMAIL":
                    placeholder_to_category[str(k)] = "EMAIL"
                elif category == "PHONE":
                    placeholder_to_category[str(k)] = "PHONE"
                elif category == "URL":
                    placeholder_to_category[str(k)] = "URL"
                elif category == "ACCOUNT":
                    placeholder_to_category[str(k)] = "ACCOUNT"

        category_counts: dict[str, int] = {}
        if isinstance(category_counts_raw, dict):
            for k, v in category_counts_raw.items():
                try:
                    category_counts[str(k)] = int(v)
                except (TypeError, ValueError):
                    continue

        return cls(
            scope_id=data.get("scope_id") if isinstance(data.get("scope_id"), str) else None,
            entity_to_placeholder=entity_to_placeholder,
            placeholder_to_entity=placeholder_to_entity,
            placeholder_to_category=placeholder_to_category,
            category_counts=category_counts,
        )


def apply_deterministic_redaction(
    text: str,
    *,
    redaction_map: DeterministicRedactionMap | None = None,
    scope_id: str | None = None,
) -> tuple[str, DeterministicRedactionMap]:
    mapping = redaction_map or DeterministicRedactionMap(scope_id=scope_id)
    return mapping.apply(text), mapping


def _normalize_value(category: RedactionCategory, value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return ""

    if category == "EMAIL":
        return stripped.lower()
    if category == "PHONE":
        digits = re.sub(r"\D", "", stripped)
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        return digits
    if category == "URL":
        return stripped.lower().rstrip("/")
    if category == "ACCOUNT":
        return re.sub(r"[^A-Za-z0-9]", "", stripped).lower()
    return stripped


def _ordinal_token(n: int) -> str:
    if n <= 0:
        raise ValueError("Ordinal must be positive")

    chars: list[str] = []
    x = n
    while x > 0:
        x -= 1
        chars.append(chr(ord("A") + (x % 26)))
        x //= 26
    return "".join(reversed(chars))
