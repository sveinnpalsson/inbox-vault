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
    attachments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a synthetic Gmail message payload in the shape returned by Gmail API."""
    parts: list[dict[str, Any]] = [
        {
            "partId": "1",
            "mimeType": "text/plain",
            "body": {"data": _b64url(body_text)},
        }
    ]

    for idx, attachment in enumerate(attachments or [], start=2):
        headers = []
        disposition = str(attachment.get("content_disposition", "attachment")).strip()
        if disposition:
            headers.append({"name": "Content-Disposition", "value": disposition})
        content_id = str(attachment.get("content_id") or "").strip()
        if content_id:
            headers.append({"name": "Content-ID", "value": content_id})

        part = {
            "partId": str(attachment.get("part_id") or idx),
            "mimeType": str(attachment.get("mime_type") or "application/octet-stream"),
            "filename": str(attachment.get("filename") or ""),
            "headers": headers,
            "body": {
                "size": int(attachment.get("size_bytes", 0)),
                "attachmentId": str(
                    attachment.get("attachment_id") or f"att-{msg_id}-{idx}"
                ),
            },
        }
        inline_data_text = attachment.get("inline_data_text")
        if isinstance(inline_data_text, str):
            part["body"]["data"] = _b64url(inline_data_text)
        parts.append(part)

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
            "parts": parts,
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
