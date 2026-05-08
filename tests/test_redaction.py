from __future__ import annotations

from inbox_vault.config import (
    AccountConfig,
    AppConfig,
    DBConfig,
    EmbeddingConfig,
    LLMConfig,
    RedactionConfig,
    RetrievalConfig,
)
from inbox_vault.opf import OPFUnavailableError
from inbox_vault.redaction import (
    is_redaction_value_allowed,
    model_redact_text,
    redact_text,
    regex_redact_text,
    resolve_redaction_backend,
)


def test_regex_redaction_masks_common_tokens():
    text = "Email bob@example.com call 212-555-1234 url https://example.com account acct-99887766"
    out = regex_redact_text(text)
    assert "[REDACTED_EMAIL]" in out
    assert "[REDACTED_PHONE]" in out
    assert "[REDACTED_URL]" in out
    assert "[REDACTED_ACCOUNT]" in out


def test_model_mode_falls_back_to_regex_on_error(monkeypatch):
    cfg = LLMConfig(
        enabled=True, endpoint="http://localhost:8080", model="local", timeout_seconds=1.0
    )

    def fail_chunk(*_args, **_kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr("inbox_vault.redaction._model_redact_chunk", fail_chunk)

    out = redact_text("Reach me at bob@example.com", mode="model", llm_cfg=cfg)
    assert "[REDACTED_EMAIL]" in out


def test_hybrid_mode_applies_regex_after_model(monkeypatch):
    cfg = LLMConfig(
        enabled=True, endpoint="http://localhost:8080", model="local", timeout_seconds=1.0
    )

    monkeypatch.setattr(
        "inbox_vault.redaction._model_redact_chunk",
        lambda chunk, **_kwargs: f"MODEL::{chunk}",
    )

    out = redact_text("URL https://internal.local", mode="hybrid", llm_cfg=cfg)
    assert "MODEL::" in out
    assert "[REDACTED_URL]" in out


def test_model_redaction_uses_chunking(monkeypatch):
    cfg = LLMConfig(
        enabled=True, endpoint="http://localhost:8080", model="local", timeout_seconds=1.0
    )
    calls: list[tuple[str, int, int]] = []

    def fake_chunk(chunk: str, **kwargs):
        calls.append((chunk, kwargs["chunk_index"], kwargs["chunk_total"]))
        return f"[{len(chunk)}]"

    monkeypatch.setattr("inbox_vault.redaction._model_redact_chunk", fake_chunk)

    out = model_redact_text("abcdefghij", llm_cfg=cfg, profile="std", instruction="", chunk_chars=4)
    assert calls == [("abcd", 1, 3), ("efgh", 2, 3), ("ij", 3, 3)]
    assert out == "[4][4][2]"


def test_redaction_value_validator_rejects_common_false_positives():
    assert is_redaction_value_allowed("ACCOUNT", "24") is False
    assert is_redaction_value_allowed("ADDRESS", "CA") is False
    assert is_redaction_value_allowed("PERSON", "LAST NAME") is False
    assert is_redaction_value_allowed("PERSON", "name") is False


def test_redaction_value_validator_rejects_custom_and_accepts_valid_entities():
    assert is_redaction_value_allowed("CUSTOM", "Project Delta") is False
    assert is_redaction_value_allowed("CUSTOM", "amy_doe") is True
    assert is_redaction_value_allowed("CUSTOM", "neo-43CU") is True
    assert is_redaction_value_allowed("CUSTOM", "Agent1") is False
    assert is_redaction_value_allowed("EMAIL", "alice@example.com") is True
    assert is_redaction_value_allowed("PHONE", "+1 (617) 555-1212") is True
    assert is_redaction_value_allowed("PERSON", "Alice Johnson") is True
    assert is_redaction_value_allowed("PERSON", "Tempobono", source_text='Last Name: "Tempobono"') is True
    assert is_redaction_value_allowed("ADDRESS", "123 Main Street") is True


def test_resolve_redaction_backend_builds_opf_detector(monkeypatch):
    def detector(text):
        return [("EMAIL", text)]

    monkeypatch.setattr("inbox_vault.redaction.resolve_opf_detector", lambda *_args, **_kwargs: detector)
    cfg = AppConfig(
        accounts=[
            AccountConfig(
                name="main",
                email="acct@example.com",
                credentials_file="credentials.json",
                token_file="token.json",
            )
        ],
        llm=LLMConfig(enabled=False, endpoint="http://localhost:8080", model="local-model"),
        db=DBConfig(),
        embeddings=EmbeddingConfig(),
        redaction=RedactionConfig(mode="hybrid", backend="opf", model="opf-local"),
        retrieval=RetrievalConfig(),
    )

    resolved = resolve_redaction_backend(cfg)
    assert resolved.backend == "opf"
    assert resolved.llm_cfg is None
    assert resolved.candidate_detector is not None
    detected = resolved.candidate_detector("alice@example.com", source="test")
    assert [(item.key_name, item.value, item.source) for item in detected] == [
        ("EMAIL", "alice@example.com", "test")
    ]


def test_resolve_opf_detector_supports_official_opf_api(monkeypatch):
    from inbox_vault import opf as opf_module

    opf_module._native_opf_detector.cache_clear()
    init_kwargs: dict[str, object] = {}

    class FakeSpan:
        def __init__(self, label: str, start: int, end: int, text: str):
            self.label = label
            self.start = start
            self.end = end
            self.text = text

    class FakeResult:
        def __init__(self):
            self.detected_spans = (
                FakeSpan("private_email", 14, 30, "jane@example.com"),
                FakeSpan("private_date", 34, 45, "May 3, 2026"),
            )

    class FakeOPF:
        def __init__(self, *, model=None, device="cuda", output_mode="typed", output_text_only=False):
            init_kwargs.update(
                {
                    "model": model,
                    "device": device,
                    "output_mode": output_mode,
                    "output_text_only": output_text_only,
                }
            )

        def redact(self, text: str):
            assert "jane@example.com" in text
            return FakeResult()

    class FakeModule:
        OPF = FakeOPF

    monkeypatch.setattr(opf_module.importlib, "import_module", lambda name: FakeModule)

    detector = opf_module.resolve_opf_detector("openai-privacy-filter")
    detected = detector("Email Jane at jane@example.com on May 3, 2026")

    assert init_kwargs == {
        "model": None,
        "device": "cpu",
        "output_mode": "typed",
        "output_text_only": False,
    }
    assert detected == [("EMAIL", "jane@example.com"), ("DATE", "May 3, 2026")]
    opf_module._native_opf_detector.cache_clear()



def test_resolve_opf_detector_handles_grouped_values_alias_and_device(monkeypatch):
    from inbox_vault import opf as opf_module

    opf_module._native_opf_detector.cache_clear()
    init_kwargs: dict[str, object] = {}

    class FakeOPF:
        def __init__(self, *, model=None, device="cpu", output_mode="typed", output_text_only=False):
            init_kwargs.update(
                {
                    "model": model,
                    "device": device,
                    "output_mode": output_mode,
                    "output_text_only": output_text_only,
                }
            )

        def redact(self, text: str):
            return {
                "redactions": [
                    {"key_name": "B_PRIVATE_EMAIL", "values": ["jane@example.com", "ops@example.com"]},
                    {"label": "private_date", "values": "May 7, 2026"},
                ]
            }

    class FakeModule:
        OPF = FakeOPF

    monkeypatch.setenv("INBOX_VAULT_OPF_DEVICE", "cuda")
    monkeypatch.setattr(opf_module.importlib, "import_module", lambda name: FakeModule)

    detector = opf_module.resolve_opf_detector("opf")
    detected = detector("Email jane@example.com and ops@example.com on May 7, 2026")

    assert init_kwargs == {
        "model": None,
        "device": "cuda",
        "output_mode": "typed",
        "output_text_only": False,
    }
    assert detected == [
        ("EMAIL", "jane@example.com"),
        ("EMAIL", "ops@example.com"),
        ("DATE", "May 7, 2026"),
    ]
    opf_module._native_opf_detector.cache_clear()


def test_resolve_opf_detector_supports_module_level_redact(monkeypatch):
    from inbox_vault import opf as opf_module

    opf_module._native_opf_detector.cache_clear()

    class Span:
        entity_group = "PRIVATE_PHONE"
        start = 5
        end = 17

    class FakeModule:
        @staticmethod
        def redact(text: str, model=None):
            assert model == "custom-checkpoint"
            return {"detected_spans": (Span(),)}

    monkeypatch.setattr(opf_module.importlib, "import_module", lambda name: FakeModule)

    detector = opf_module.resolve_opf_detector("custom-checkpoint")
    assert detector("Call 212-555-1212 today") == [("PHONE", "212-555-1212")]
    opf_module._native_opf_detector.cache_clear()

def test_resolve_redaction_backend_reports_opf_unavailable(monkeypatch):
    monkeypatch.setattr(
        "inbox_vault.redaction.resolve_opf_detector",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OPFUnavailableError("opf missing")),
    )
    cfg = AppConfig(
        accounts=[
            AccountConfig(
                name="main",
                email="acct@example.com",
                credentials_file="credentials.json",
                token_file="token.json",
            )
        ],
        llm=LLMConfig(enabled=False, endpoint="http://localhost:8080", model="local-model"),
        db=DBConfig(),
        embeddings=EmbeddingConfig(),
        redaction=RedactionConfig(mode="hybrid", backend="opf"),
        retrieval=RetrievalConfig(),
    )

    resolved = resolve_redaction_backend(cfg)
    assert resolved.backend == "opf"
    assert resolved.llm_cfg is None
    assert resolved.candidate_detector is None
    assert resolved.unavailable_reason == "opf missing"


def test_resolve_redaction_backend_keeps_local_endpoint_optional():
    cfg = AppConfig(
        accounts=[
            AccountConfig(
                name="main",
                email="acct@example.com",
                credentials_file="credentials.json",
                token_file="token.json",
            )
        ],
        llm=LLMConfig(enabled=True, endpoint="http://localhost:8080", model="local-model"),
        db=DBConfig(),
        embeddings=EmbeddingConfig(),
        redaction=RedactionConfig(
            mode="hybrid",
            backend="local",
            endpoint="http://localhost:9090",
            model="local-redactor",
            timeout_seconds=5.0,
        ),
        retrieval=RetrievalConfig(),
    )

    resolved = resolve_redaction_backend(cfg)
    assert resolved.backend == "local"
    assert resolved.llm_cfg is not None
    assert resolved.llm_cfg.endpoint == "http://localhost:9090"
    assert resolved.llm_cfg.model == "local-redactor"
    assert resolved.llm_cfg.timeout_seconds == 5.0
