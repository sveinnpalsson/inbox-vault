from __future__ import annotations

import json

import pytest
from google.auth.exceptions import RefreshError

from inbox_vault import gmail_client


class _FakeCreds:
    def __init__(self, *, should_fail_invalid_scope: bool = False):
        self.expired = True
        self.refresh_token = "rtok"
        self.valid = False
        self._should_fail_invalid_scope = should_fail_invalid_scope

    def refresh(self, _request):
        if self._should_fail_invalid_scope:
            raise RefreshError("invalid_scope")
        self.expired = False
        self.valid = True

    def to_json(self):
        return "{}"


def test_get_service_refresh_falls_back_to_token_scopes(tmp_path, monkeypatch: pytest.MonkeyPatch):
    token_file = tmp_path / "token.json"
    token_file.write_text(
        json.dumps(
            {
                "client_id": "cid",
                "client_secret": "sec",
                "refresh_token": "rtok",
                "type": "authorized_user",
                "scopes": [
                    "https://www.googleapis.com/auth/gmail.modify",
                    "openid",
                ],
            }
        ),
        encoding="utf-8",
    )

    first = _FakeCreds(should_fail_invalid_scope=True)
    second = _FakeCreds(should_fail_invalid_scope=False)
    calls: list[tuple[str, ...] | None] = []

    def _from_file(_path, scopes=None):
        calls.append(tuple(scopes) if scopes else None)
        if scopes == gmail_client.SCOPES:
            return first
        assert scopes == ["https://www.googleapis.com/auth/gmail.modify", "openid"]
        return second

    monkeypatch.setattr(gmail_client.Credentials, "from_authorized_user_file", _from_file)

    def _should_not_open_flow(*_args, **_kwargs):
        raise AssertionError("oauth local flow should not run on refresh fallback")

    monkeypatch.setattr(
        gmail_client.InstalledAppFlow,
        "from_client_secrets_file",
        _should_not_open_flow,
    )

    built_with = {}

    def _fake_build(_api, _version, credentials, **kwargs):
        built_with["credentials"] = credentials
        built_with["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(gmail_client, "build", _fake_build)
    monkeypatch.setattr(gmail_client, "AuthorizedHttp", None)

    gmail_client.get_service("credentials.json", str(token_file))

    assert calls == [
        tuple(gmail_client.SCOPES),
        ("https://www.googleapis.com/auth/gmail.modify", "openid"),
    ]
    assert built_with["credentials"] is second


def test_get_service_refresh_non_scope_error_raises(tmp_path, monkeypatch: pytest.MonkeyPatch):
    token_file = tmp_path / "token.json"
    token_file.write_text("{}", encoding="utf-8")

    class _Creds(_FakeCreds):
        def refresh(self, _request):
            raise RefreshError("invalid_grant")

    monkeypatch.setattr(
        gmail_client.Credentials,
        "from_authorized_user_file",
        lambda *_args, **_kwargs: _Creds(),
    )
    monkeypatch.setattr(gmail_client, "build", lambda *_args, **_kwargs: object())

    with pytest.raises(RefreshError):
        gmail_client.get_service("credentials.json", str(token_file))


def test_get_service_loads_gog_refresh_token_shape(tmp_path, monkeypatch: pytest.MonkeyPatch):
    token_file = tmp_path / "token-gog.json"
    token_file.write_text(
        json.dumps(
            {
                "email": "acct@example.com",
                "client": "gog",
                "services": ["gmail"],
                "scopes": ["https://www.googleapis.com/auth/gmail.modify"],
                "refresh_token": "rtok",
            }
        ),
        encoding="utf-8",
    )

    credentials_file = tmp_path / "credentials.json"
    credentials_file.write_text(
        json.dumps(
            {
                "installed": {
                    "client_id": "cid-from-creds",
                    "client_secret": "sec-from-creds",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            }
        ),
        encoding="utf-8",
    )

    class _ValidCreds:
        expired = False
        valid = True
        refresh_token = "rtok"

    captured: dict[str, object] = {}

    def _from_file(*_args, **_kwargs):
        raise ValueError(
            "Authorized user info was not in the expected format, missing fields client_secret, client_id."
        )

    def _from_info(info, scopes=None):
        captured["info"] = info
        captured["scopes"] = scopes
        return _ValidCreds()

    monkeypatch.setattr(gmail_client.Credentials, "from_authorized_user_file", _from_file)
    monkeypatch.setattr(gmail_client.Credentials, "from_authorized_user_info", _from_info)

    def _should_not_open_flow(*_args, **_kwargs):
        raise AssertionError("oauth local flow should not run when gog token can be normalized")

    monkeypatch.setattr(
        gmail_client.InstalledAppFlow, "from_client_secrets_file", _should_not_open_flow
    )

    built_with = {}

    def _fake_build(_api, _version, credentials, **kwargs):
        built_with["credentials"] = credentials
        built_with["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(gmail_client, "build", _fake_build)
    monkeypatch.setattr(gmail_client, "AuthorizedHttp", None)

    gmail_client.get_service(str(credentials_file), str(token_file))

    assert captured["scopes"] == gmail_client.SCOPES
    assert captured["info"] == {
        "type": "authorized_user",
        "client_id": "cid-from-creds",
        "client_secret": "sec-from-creds",
        "refresh_token": "rtok",
        "token_uri": "https://oauth2.googleapis.com/token",
        "scopes": ["https://www.googleapis.com/auth/gmail.modify"],
    }
    assert isinstance(built_with["credentials"], _ValidCreds)


def test_safe_execute_retries_timeout_exceptions(monkeypatch: pytest.MonkeyPatch):
    calls = {"count": 0}

    class _Call:
        def execute(self):
            calls["count"] += 1
            if calls["count"] < 3:
                raise TimeoutError("timed out")
            return {"ok": True}

    monkeypatch.setattr(gmail_client.time, "sleep", lambda *_args, **_kwargs: None)

    out = gmail_client.safe_execute(lambda: _Call(), retries=3, backoff=0.01)
    assert out == {"ok": True}
    assert calls["count"] == 3


def test_get_service_uses_configured_http_timeout(tmp_path, monkeypatch: pytest.MonkeyPatch):
    token_file = tmp_path / "token.json"
    token_file.write_text("{}", encoding="utf-8")

    class _Creds:
        expired = False
        valid = True
        refresh_token = "rtok"

        def to_json(self):
            return "{}"

    monkeypatch.setattr(
        gmail_client.Credentials,
        "from_authorized_user_file",
        lambda *_args, **_kwargs: _Creds(),
    )

    captured: dict[str, object] = {}

    class _FakeHttp:
        def __init__(self, timeout):
            captured["timeout"] = timeout

    monkeypatch.setattr(gmail_client.httplib2, "Http", _FakeHttp)

    def _fake_authorized_http(creds, http):
        captured["auth_creds"] = creds
        captured["auth_http"] = http
        return "authorized-http"

    monkeypatch.setattr(gmail_client, "AuthorizedHttp", _fake_authorized_http)

    def _fake_build(_api, _version, **kwargs):
        captured["build_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(gmail_client, "build", _fake_build)

    gmail_client.get_service("credentials.json", str(token_file), timeout_seconds=17)

    assert captured["timeout"] == 17.0
    assert captured["build_kwargs"] == {"http": "authorized-http"}
    assert "credentials" not in captured["build_kwargs"]
