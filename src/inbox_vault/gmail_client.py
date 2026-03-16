from __future__ import annotations

import base64
import http.client
import json
import os
import socket
import ssl
import sys
import time
import webbrowser
from datetime import datetime
from email.header import decode_header, make_header
from email.utils import getaddresses, parsedate_to_datetime
from typing import Any, Iterable

import httplib2
from bs4 import BeautifulSoup
from google.auth.exceptions import RefreshError, TransportError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

try:
    from google_auth_httplib2 import AuthorizedHttp
except Exception:  # pragma: no cover
    AuthorizedHttp = None  # type: ignore[assignment]
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]


def safe_execute(callable_execute, retries: int = 3, backoff: float = 1.0):
    for attempt in range(1, retries + 1):
        try:
            return callable_execute().execute()
        except (
            TransportError,
            ssl.SSLEOFError,
            http.client.RemoteDisconnected,
            socket.timeout,
            TimeoutError,
        ):
            if attempt == retries:
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))
        except HttpError as e:
            status = getattr(e, "status_code", None) or getattr(e.resp, "status", 0)
            if (status in {408, 429} or 500 <= status < 600) and attempt < retries:
                time.sleep(backoff * (2 ** (attempt - 1)))
                continue
            raise


def _token_scopes_from_file(token_file: str) -> list[str]:
    try:
        with open(token_file, encoding="utf-8") as f:
            token_data = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return []

    scopes = token_data.get("scopes")
    if isinstance(scopes, list):
        return [s for s in scopes if isinstance(s, str) and s.strip()]

    # Some token exports use a single space-delimited "scope" string.
    scope = token_data.get("scope")
    if isinstance(scope, str):
        return [s for s in scope.split() if s.strip()]

    return []


def _read_json_file(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_authorized_user_info(
    token_data: dict[str, Any], credentials_file: str
) -> dict[str, Any] | None:
    refresh_token = token_data.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        return None

    client_id = token_data.get("client_id")
    client_secret = token_data.get("client_secret")
    oauth_client: dict[str, Any] = {}

    if not (
        isinstance(client_id, str)
        and client_id.strip()
        and isinstance(client_secret, str)
        and client_secret.strip()
    ):
        credentials_data = _read_json_file(credentials_file) or {}
        if isinstance(credentials_data.get("installed"), dict):
            oauth_client = credentials_data["installed"]
        elif isinstance(credentials_data.get("web"), dict):
            oauth_client = credentials_data["web"]

        client_id = oauth_client.get("client_id")
        client_secret = oauth_client.get("client_secret")

        if not (
            isinstance(client_id, str)
            and client_id.strip()
            and isinstance(client_secret, str)
            and client_secret.strip()
        ):
            return None

    normalized: dict[str, Any] = {
        "type": "authorized_user",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "token_uri": token_data.get("token_uri")
        or oauth_client.get("token_uri")
        or "https://oauth2.googleapis.com/token",
    }

    scopes = token_data.get("scopes")
    if isinstance(scopes, list):
        normalized["scopes"] = [s for s in scopes if isinstance(s, str) and s.strip()]

    return normalized


def _load_credentials_from_token_file(
    credentials_file: str, token_file: str, scopes: list[str]
) -> Credentials:
    try:
        return Credentials.from_authorized_user_file(token_file, scopes)
    except FileNotFoundError:
        raise
    except ValueError as err:
        token_data = _read_json_file(token_file)
        if not token_data:
            raise err

        normalized_info = _normalize_authorized_user_info(token_data, credentials_file)
        if not normalized_info:
            raise err

        return Credentials.from_authorized_user_info(normalized_info, scopes)


def _is_invalid_scope(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "invalid_scope" in msg or "invalid scope" in msg


def get_service(credentials_file: str, token_file: str, *, timeout_seconds: float = 60.0):
    creds = None
    try:
        creds = _load_credentials_from_token_file(credentials_file, token_file, SCOPES)
    except FileNotFoundError:
        creds = None

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError as e:
            if not _is_invalid_scope(e):
                raise

            token_scopes = _token_scopes_from_file(token_file)
            if not token_scopes:
                raise

            # Compatibility fallback for exported gog-style tokens where forcing
            # local SCOPES during refresh can trigger invalid_scope.
            fallback_creds = _load_credentials_from_token_file(
                credentials_file, token_file, token_scopes
            )
            fallback_creds.refresh(Request())
            creds = fallback_creds

        with open(token_file, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    if not creds or not creds.valid:
        if not os.path.isfile(credentials_file):
            raise FileNotFoundError(
                f"Gmail OAuth credentials file not found: {credentials_file}\n\n"
                "To set up credentials:\n"
                "  1. Go to https://console.cloud.google.com/apis/credentials\n"
                "  2. Create an OAuth 2.0 Client ID (type: Desktop app)\n"
                "  3. Download the JSON and save it to the path above\n"
                "  4. Enable the Gmail API at https://console.cloud.google.com/apis/library/gmail.googleapis.com\n\n"
                "See the README 'Getting started' section for a full walkthrough."
            )
        flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
        try:
            webbrowser.get()
            open_browser = True
        except webbrowser.Error:
            open_browser = False
            print(
                "No browser detected (headless / WSL / SSH).\n"
                "A URL will be printed below -- open it in any browser to authorize.\n"
                "On WSL2, use your Windows browser; localhost is shared.\n",
                file=sys.stderr,
            )
        creds = flow.run_local_server(
            port=0, access_type="offline", prompt="consent", open_browser=open_browser
        )
        with open(token_file, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    timeout = max(1.0, float(timeout_seconds))
    if AuthorizedHttp is not None:
        http_client = AuthorizedHttp(creds, http=httplib2.Http(timeout=timeout))
        # googleapiclient.discovery.build() does not allow both `http` and
        # `credentials`; AuthorizedHttp already carries credentials.
        return build("gmail", "v1", http=http_client)

    return build("gmail", "v1", credentials=creds)


def list_message_ids_paged(
    service, query: str, page_size: int = 200, max_messages: int | None = None
) -> list[str]:
    ids: list[str] = []
    token = None
    while True:
        resp = safe_execute(
            lambda: service.users()
            .messages()
            .list(userId="me", q=query, maxResults=page_size, pageToken=token)
        )
        for item in resp.get("messages", []):
            ids.append(item["id"])
            if max_messages and len(ids) >= max_messages:
                return ids
        token = resp.get("nextPageToken")
        if not token:
            return ids


def fetch_full_message_payload(service, msg_id: str) -> dict[str, Any] | None:
    try:
        return safe_execute(
            lambda: service.users().messages().get(userId="me", id=msg_id, format="full")
        )
    except HttpError as e:
        if getattr(e.resp, "status", None) == 404:
            return None
        raise


def decode_name(name: str) -> str:
    try:
        return str(make_header(decode_header(name))).strip()
    except Exception:
        return (name or "").strip()


def parse_address_header(value: str) -> list[tuple[str, str]]:
    pairs = []
    for name, email in getaddresses([value or ""]):
        if email:
            pairs.append((decode_name(name), email.strip().lower()))
    return pairs


def _walk_parts(parts: Iterable[dict], collected: dict[str, str]):
    for part in parts:
        mime = part.get("mimeType", "")
        data = (part.get("body") or {}).get("data")
        if mime in ("text/plain", "text/html") and data:
            try:
                # Gmail base64url data may be missing padding.
                padding = "=" * (-len(data) % 4)
                decoded = base64.urlsafe_b64decode(data + padding)
                text = decoded.decode("utf-8", errors="ignore")
                key = "plain" if mime == "text/plain" else "html"
                collected[key] += text + "\n"
            except Exception:
                # Ignore malformed MIME parts and continue parsing siblings.
                pass
        nested = part.get("parts") or []
        if nested:
            _walk_parts(nested, collected)


def payload_to_record(raw: dict[str, Any], account_email: str) -> dict[str, Any]:
    headers = {h["name"]: h["value"] for h in raw.get("payload", {}).get("headers", [])}
    from_list = parse_address_header(headers.get("From", ""))
    to_list = parse_address_header(headers.get("To", ""))

    from_addr = from_list[0][1] if from_list else ""
    to_addr = to_list[0][1] if to_list else ""

    internal_ms = raw.get("internalDate")
    internal_ts = int(internal_ms) if internal_ms else None

    date_iso = None
    try:
        date_hdr = headers.get("Date")
        if date_hdr:
            date_iso = parsedate_to_datetime(date_hdr).date().isoformat()
    except Exception:
        if internal_ts:
            date_iso = datetime.utcfromtimestamp(internal_ts / 1000).date().isoformat()

    payload = raw.get("payload", {})
    parts = payload.get("parts") or [payload]
    collected = {"plain": "", "html": ""}
    _walk_parts(parts, collected)

    if collected["plain"].strip():
        body = collected["plain"]
    elif collected["html"].strip():
        body = BeautifulSoup(collected["html"], "html.parser").get_text(separator="\n")
    else:
        body = raw.get("snippet", "")

    snippet = (body[:220] + "…") if len(body) > 220 else body
    snippet = snippet.replace("\n", " ").strip()

    return {
        "msg_id": raw.get("id"),
        "account_email": account_email,
        "thread_id": raw.get("threadId"),
        "date_iso": date_iso,
        "internal_ts": internal_ts,
        "from_addr": from_addr,
        "to_addr": to_addr,
        "subject": headers.get("Subject", "(no subject)"),
        "snippet": snippet,
        "body_text": body,
        "labels": raw.get("labelIds", []),
        "history_id": int(raw.get("historyId")) if raw.get("historyId") else None,
    }


def get_profile(service) -> dict[str, Any]:
    return safe_execute(lambda: service.users().getProfile(userId="me"))


def get_profile_history_id(service) -> int:
    return int(get_profile(service)["historyId"])


def get_authenticated_email(service) -> str:
    return get_profile(service).get("emailAddress", "")


def list_incremental_added_ids(service, start_history_id: int) -> tuple[set[str], int]:
    ids: set[str] = set()
    token = None
    latest = start_history_id

    while True:
        resp = safe_execute(
            lambda: service.users()
            .history()
            .list(
                userId="me",
                startHistoryId=start_history_id,
                historyTypes=["messageAdded"],
                pageToken=token,
            )
        )
        latest = int(resp.get("historyId", latest))
        for record in resp.get("history", []) or []:
            for added in record.get("messagesAdded", []) or []:
                msg = added.get("message", {})
                labels = set(msg.get("labelIds", []) or [])
                if "INBOX" in labels or "SENT" in labels:
                    mid = msg.get("id")
                    if mid:
                        ids.add(mid)
        token = resp.get("nextPageToken")
        if not token:
            break

    return ids, latest
