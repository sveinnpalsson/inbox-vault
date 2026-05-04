from __future__ import annotations

import importlib
import inspect
import re
from functools import lru_cache
from typing import Any, Callable


class OPFUnavailableError(RuntimeError):
    pass


EntityPair = tuple[str, str]
EntityDetector = Callable[[str], list[EntityPair]]

_LABEL_KEYS = ("entity_group", "entity", "label", "type", "tag")
_TEXT_KEYS = ("word", "text", "value", "entity")
_SUPPORTED_LABELS = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "EMAIL": "EMAIL",
    "MAIL": "EMAIL",
    "PHONE": "PHONE",
    "TEL": "PHONE",
    "TELEPHONE": "PHONE",
    "MOBILE": "PHONE",
    "URL": "URL",
    "URI": "URL",
    "WEB": "URL",
    "ACCOUNT": "ACCOUNT",
    "IBAN": "ACCOUNT",
    "ROUTING": "ACCOUNT",
    "CARD": "ACCOUNT",
    "SSN": "ACCOUNT",
    "PASSPORT": "ACCOUNT",
    "TAXID": "ACCOUNT",
    "TIN": "ACCOUNT",
    "ID": "ACCOUNT",
    "ADDRESS": "ADDRESS",
    "ADDR": "ADDRESS",
    "LOC": "ADDRESS",
    "LOCATION": "ADDRESS",
    "GPE": "ADDRESS",
}


def resolve_opf_detector(model_name: str | None = None) -> EntityDetector:
    errors: list[str] = []

    try:
        return _native_opf_detector(model_name)
    except OPFUnavailableError as exc:
        errors.append(str(exc))

    try:
        return _transformers_opf_detector(model_name)
    except OPFUnavailableError as exc:
        errors.append(str(exc))

    detail = "; ".join(part for part in errors if part) or "OPF backend is unavailable"
    raise OPFUnavailableError(detail)


@lru_cache(maxsize=4)
def _native_opf_detector(model_name: str | None) -> EntityDetector:
    try:
        module = importlib.import_module("opf")
    except Exception as exc:
        raise OPFUnavailableError(f"opf package import failed: {exc}") from exc

    factory = getattr(module, "load_detector", None)
    if callable(factory):
        detector = _call_with_optional_model(factory, model_name)
        return _coerce_detector(detector)

    detect = getattr(module, "detect", None)
    if callable(detect):
        return _coerce_detector(detect, model_name=model_name)

    raise OPFUnavailableError(
        "opf package is installed but does not expose load_detector() or detect()"
    )


@lru_cache(maxsize=4)
def _transformers_opf_detector(model_name: str | None) -> EntityDetector:
    if not model_name:
        raise OPFUnavailableError(
            "missing redaction.model for backend=opf and no native opf package detected"
        )

    try:
        from transformers import pipeline  # type: ignore
    except Exception as exc:
        raise OPFUnavailableError(f"transformers import failed: {exc}") from exc

    pipe = pipeline(
        "token-classification",
        model=model_name,
        aggregation_strategy="simple",
    )

    def _detect(text: str) -> list[EntityPair]:
        return _normalize_detector_output(pipe(text), source_text=text)

    return _detect


def _call_with_optional_model(factory: Callable[..., Any], model_name: str | None) -> Any:
    if model_name is None:
        try:
            return factory()
        except TypeError:
            return factory(None)

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        if "model_name" in signature.parameters:
            return factory(model_name=model_name)
        if "model" in signature.parameters:
            return factory(model=model_name)
    try:
        return factory(model_name)
    except TypeError:
        return factory()


def _coerce_detector(detector: Any, *, model_name: str | None = None) -> EntityDetector:
    def _detect(text: str) -> list[EntityPair]:
        if callable(detector):
            result = _invoke_detector(detector, text=text, model_name=model_name)
        elif hasattr(detector, "detect") and callable(detector.detect):
            result = _invoke_detector(detector.detect, text=text, model_name=model_name)
        elif hasattr(detector, "predict") and callable(detector.predict):
            result = _invoke_detector(detector.predict, text=text, model_name=model_name)
        else:
            raise OPFUnavailableError("OPF detector does not expose a callable detect/predict API")
        return _normalize_detector_output(result, source_text=text)

    return _detect


def _invoke_detector(func: Callable[..., Any], *, text: str, model_name: str | None) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        if "text" in signature.parameters:
            kwargs: dict[str, Any] = {"text": text}
            if model_name and "model_name" in signature.parameters:
                kwargs["model_name"] = model_name
            elif model_name and "model" in signature.parameters:
                kwargs["model"] = model_name
            return func(**kwargs)
        if model_name and len(signature.parameters) >= 2:
            return func(text, model_name)
    return func(text)


def _normalize_detector_output(result: Any, *, source_text: str) -> list[EntityPair]:
    if not isinstance(result, list):
        return []

    entities: list[EntityPair] = []
    for item in result:
        label, value = _extract_entity(item, source_text=source_text)
        if label and value:
            entities.append((label, value))
    return entities


def _extract_entity(item: Any, *, source_text: str) -> EntityPair:
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        label = _normalize_label(item[0])
        value = str(item[1]).strip()
        return label, value

    if not isinstance(item, dict):
        return "", ""

    raw_label = ""
    for key in _LABEL_KEYS:
        value = item.get(key)
        if value is not None:
            raw_label = str(value)
            break
    label = _normalize_label(raw_label)

    text_value = ""
    for key in _TEXT_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            text_value = value
            break
    if not text_value:
        start = item.get("start")
        end = item.get("end")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(source_text):
            text_value = source_text[start:end]

    return label, _normalize_wordpiece_text(text_value)


def _normalize_label(raw_label: str) -> str:
    cleaned = re.sub(r"^(?:B|I|L|U|S|E)-", "", (raw_label or "").strip().upper())
    return _SUPPORTED_LABELS.get(cleaned, "")


def _normalize_wordpiece_text(value: str) -> str:
    text = (value or "").strip()
    text = text.replace(" ##", "").replace("##", "")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()
