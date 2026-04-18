from __future__ import annotations

import base64

from inbox_vault.gmail_client import payload_to_record


def _b64url(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")


def test_payload_to_record_collects_attachment_metadata_without_indexing_attachment_text():
    raw = {
        "id": "m-attach",
        "threadId": "thread-m-attach",
        "historyId": "42",
        "internalDate": "1701684900000",
        "labelIds": ["INBOX"],
        "snippet": "fallback snippet",
        "payload": {
            "headers": [
                {"name": "From", "value": "sender@example.com"},
                {"name": "To", "value": "acct@example.com"},
                {"name": "Subject", "value": "Attachment test"},
                {"name": "Date", "value": "Mon, 04 Dec 2023 10:15:00 +0000"},
            ],
            "parts": [
                {
                    "partId": "1",
                    "mimeType": "text/plain",
                    "body": {"data": _b64url("Real message body")},
                },
                {
                    "partId": "2",
                    "mimeType": "text/plain",
                    "filename": "notes.txt",
                    "headers": [
                        {
                            "name": "Content-Disposition",
                            "value": "attachment; filename=\"notes.txt\"",
                        }
                    ],
                    "body": {
                        "attachmentId": "att-note",
                        "size": 27,
                        "data": _b64url("Attachment text should not be indexed"),
                    },
                },
                {
                    "partId": "3",
                    "mimeType": "image/png",
                    "filename": "inline.png",
                    "headers": [
                        {
                            "name": "Content-Disposition",
                            "value": "inline; filename=\"inline.png\"",
                        },
                        {"name": "Content-ID", "value": "<cid-inline>"},
                    ],
                    "body": {
                        "attachmentId": "att-inline",
                        "size": 456,
                    },
                },
            ],
        },
    }

    rec = payload_to_record(raw, "acct@example.com")

    assert rec["body_text"].strip() == "Real message body"
    assert "Attachment text should not be indexed" not in rec["body_text"]
    assert rec["attachments"] == [
        {
            "part_id": "2",
            "gmail_attachment_id": "att-note",
            "mime_type": "text/plain",
            "filename": "notes.txt",
            "size_bytes": 27,
            "content_disposition": 'attachment; filename="notes.txt"',
            "content_id": "",
            "is_inline": False,
            "inventory_state": "metadata_only",
        },
        {
            "part_id": "3",
            "gmail_attachment_id": "att-inline",
            "mime_type": "image/png",
            "filename": "inline.png",
            "size_bytes": 456,
            "content_disposition": 'inline; filename="inline.png"',
            "content_id": "<cid-inline>",
            "is_inline": True,
            "inventory_state": "metadata_only",
        },
    ]
