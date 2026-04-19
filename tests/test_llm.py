from __future__ import annotations

import requests

from inbox_vault.config import EmbeddingConfig, LLMConfig
from inbox_vault.llm import chat_json, chat_text, embedding_vector, embedding_vectors


class _Resp:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = str(self._payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}", response=self)

    def json(self):
        return self._payload


def test_embedding_retries_then_success(monkeypatch):
    cfg = EmbeddingConfig(
        endpoint="http://embed.local",
        model="embed",
        timeout_seconds=1.0,
        max_retries=3,
        backoff_base_seconds=0.0,
        backoff_max_seconds=0.0,
        fallback="none",
    )

    calls = {"count": 0}

    def fake_post(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            return _Resp(503)
        return _Resp(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]})

    monkeypatch.setattr("inbox_vault.llm.requests.post", fake_post)

    out = embedding_vector(cfg, "hello")
    assert out == [0.1, 0.2, 0.3]
    assert calls["count"] == 3


def test_embedding_hash_fallback(monkeypatch):
    cfg = EmbeddingConfig(
        endpoint="http://embed.local",
        model="embed",
        timeout_seconds=1.0,
        max_retries=1,
        backoff_base_seconds=0.0,
        backoff_max_seconds=0.0,
        fallback="hash",
        fallback_dim=32,
    )

    monkeypatch.setattr("inbox_vault.llm.requests.post", lambda *_a, **_k: _Resp(503))

    out = embedding_vector(cfg, "fallback me")
    assert len(out) == 32
    assert any(v != 0 for v in out)


def test_embedding_vectors_preserve_input_order(monkeypatch):
    cfg = EmbeddingConfig(
        endpoint="http://embed.local",
        model="embed",
        timeout_seconds=1.0,
        max_retries=1,
        backoff_base_seconds=0.0,
        backoff_max_seconds=0.0,
        fallback="none",
    )

    def fake_post(*_args, **kwargs):
        assert kwargs["json"]["input"] == ["alpha", "beta"]
        return _Resp(
            200,
            {
                "data": [
                    {"embedding": [1.0, 0.0]},
                    {"embedding": [0.0, 1.0]},
                ]
            },
        )

    monkeypatch.setattr("inbox_vault.llm.requests.post", fake_post)

    out = embedding_vectors(cfg, ["alpha", "beta"])
    assert out == [[1.0, 0.0], [0.0, 1.0]]


def test_embedding_vectors_split_retryable_size_errors(monkeypatch):
    cfg = EmbeddingConfig(
        endpoint="http://embed.local",
        model="embed",
        timeout_seconds=1.0,
        max_retries=1,
        backoff_base_seconds=0.0,
        backoff_max_seconds=0.0,
        fallback="none",
    )

    calls: list[list[str]] = []

    def fake_post(*_args, **kwargs):
        texts = list(kwargs["json"]["input"])
        calls.append(texts)
        if len(texts) > 1:
            return _Resp(400, {"error": {"message": "input is too large"}})
        return _Resp(200, {"data": [{"embedding": [float(len(texts[0])), 1.0]}]})

    monkeypatch.setattr("inbox_vault.llm.requests.post", fake_post)

    out = embedding_vectors(cfg, ["alpha", "beta"])
    assert out == [[5.0, 1.0], [4.0, 1.0]]
    assert calls == [["alpha", "beta"], ["alpha"], ["beta"]]


def test_chat_json_uses_reasoning_content_when_content_empty(monkeypatch):
    cfg = LLMConfig(endpoint="http://llm.local", model="test-model", timeout_seconds=1.0)

    def fake_post(*_args, **_kwargs):
        return _Resp(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": 'preface {"category":"work","importance":7,"action":"review","summary":"ok"}',
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("inbox_vault.llm.requests.post", fake_post)

    out = chat_json(cfg, [{"role": "user", "content": "json please"}])
    assert out == {"category": "work", "importance": 7, "action": "review", "summary": "ok"}


def test_chat_text_supports_content_array(monkeypatch):
    cfg = LLMConfig(endpoint="http://llm.local", model="test-model", timeout_seconds=1.0)

    def fake_post(*_args, **kwargs):
        assert kwargs["json"]["chat_template_kwargs"] == {"enable_thinking": False}
        return _Resp(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "hello "},
                                {"type": "text", "text": "world"},
                            ]
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("inbox_vault.llm.requests.post", fake_post)

    out = chat_text(cfg, [{"role": "user", "content": "say hi"}])
    assert out == "hello world"


def test_chat_text_retries_without_response_format_then_chat_template_kwargs(monkeypatch):
    cfg = LLMConfig(endpoint="http://llm.local", model="test-model", timeout_seconds=1.0)
    payloads: list[dict] = []

    def fake_post(*_args, **kwargs):
        payloads.append(dict(kwargs["json"]))
        if len(payloads) < 3:
            return _Resp(400)
        return _Resp(200, {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("inbox_vault.llm.requests.post", fake_post)

    out = chat_text(
        cfg,
        [{"role": "user", "content": "json please"}],
        response_format={"type": "json_object"},
    )

    assert out == "ok"
    assert payloads[0]["response_format"] == {"type": "json_object"}
    assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" not in payloads[1]
    assert payloads[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" not in payloads[2]
    assert "chat_template_kwargs" not in payloads[2]
