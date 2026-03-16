from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any


def _b64url(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")


def gmail_message_payload(
    msg_id: str,
    *,
    thread_id: str | None = None,
    history_id: int = 1,
    from_addr: str = "sender@example.com",
    to_addr: str = "recipient@example.com",
    subject: str = "Hello",
    body_text: str = "This is a synthetic message body.",
    labels: list[str] | None = None,
    internal_date_ms: int = 1_700_000_000_000,
) -> dict[str, Any]:
    """Create a synthetic Gmail message payload in the shape returned by Gmail API."""
    return {
        "id": msg_id,
        "threadId": thread_id or f"thread-{msg_id}",
        "historyId": str(history_id),
        "internalDate": str(internal_date_ms),
        "labelIds": labels or ["INBOX"],
        "snippet": body_text[:100],
        "payload": {
            "headers": [
                {"name": "From", "value": from_addr},
                {"name": "To", "value": to_addr},
                {"name": "Subject", "value": subject},
                {"name": "Date", "value": "Mon, 04 Dec 2023 10:15:00 +0000"},
            ],
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": _b64url(body_text)},
                }
            ],
        },
    }


def gmail_history_page(
    *,
    added_ids: list[str],
    history_id: int,
    include_non_mailbox: bool = False,
) -> dict[str, Any]:
    """Create synthetic Gmail history.list page payload."""
    history = []
    for mid in added_ids:
        history.append(
            {
                "messagesAdded": [
                    {
                        "message": {
                            "id": mid,
                            "labelIds": ["INBOX"],
                        }
                    }
                ]
            }
        )
    if include_non_mailbox:
        history.append(
            {
                "messagesAdded": [
                    {
                        "message": {
                            "id": "ignored-archive",
                            "labelIds": ["CATEGORY_UPDATES"],
                        }
                    }
                ]
            }
        )
    return {"history": history, "historyId": str(history_id)}


@dataclass(slots=True)
class FakeAccount:
    name: str = "acct"
    email: str = "acct@example.com"
    credentials_file: str = "fake-credentials.json"
    token_file: str = "fake-token.json"
