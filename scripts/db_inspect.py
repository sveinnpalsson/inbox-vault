#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sqlcipher3 import dbapi2 as sqlite

DEFAULT_TABLES = [
    "messages",
    "raw_messages",
    "message_enrichment",
    "contact_stats",
    "contact_profiles",
    "message_vectors_v2",
    "message_chunk_vectors_v2",
    "vector_index_state_v2",
    "message_fts",
    "message_fts_redacted",
    "sync_cursors",
]


def _sqlcipher_quote(value: str) -> str:
    return value.replace("'", "''")


def _truncate(text: object, max_chars: int = 180) -> str:
    if text is None:
        return ""
    s = str(text).replace("\n", " ").strip()
    if len(s) <= max_chars:
        return s
    return s[: max(1, max_chars - 1)].rstrip() + "…"


def _table_exists(conn, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (table,),
    ).fetchone()
    return bool(row)


def _count(conn, table: str) -> int | None:
    if not _table_exists(conn, table):
        return None
    return int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


def _print_heading(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_table_counts(conn) -> None:
    _print_heading("Table counts")
    for table in DEFAULT_TABLES:
        c = _count(conn, table)
        if c is None:
            print(f"- {table}: <missing>")
        else:
            print(f"- {table}: {c}")


def _print_latest_messages(conn, *, limit: int) -> None:
    _print_heading(f"Latest messages (limit={limit})")
    if not _table_exists(conn, "messages"):
        print("messages table missing")
        return

    rows = conn.execute(
        """
        SELECT msg_id, date_iso, from_addr, to_addr, subject, snippet
        FROM messages
        ORDER BY COALESCE(internal_ts, 0) DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    if not rows:
        print("(no rows)")
        return

    for i, (msg_id, date_iso, from_addr, to_addr, subject, snippet) in enumerate(rows, start=1):
        print(
            f"{i}. msg_id={msg_id} date={_truncate(date_iso, 32)} "
            f"from={_truncate(from_addr, 40)} to={_truncate(to_addr, 40)}"
        )
        print(f"   subject={_truncate(subject, 120)}")
        print(f"   snippet={_truncate(snippet, 140)}")


def _print_enrichment(conn, *, limit: int) -> None:
    _print_heading("Enrichment overview")
    c = _count(conn, "message_enrichment")
    if c is None:
        print("message_enrichment table missing")
        return
    print(f"rows={c}")
    if c == 0:
        return

    rows = conn.execute(
        """
        SELECT msg_id, category, importance, action, summary, enriched_at
        FROM message_enrichment
        ORDER BY enriched_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    for i, (msg_id, category, importance, action, summary, enriched_at) in enumerate(rows, start=1):
        print(
            f"{i}. msg_id={msg_id} category={_truncate(category, 30)} "
            f"importance={importance} action={_truncate(action, 20)} "
            f"at={_truncate(enriched_at, 32)}"
        )
        print(f"   summary={_truncate(summary, 160)}")


def _profile_keys(profile_json: str) -> list[str]:
    try:
        data = json.loads(profile_json)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    return sorted(str(k) for k in data.keys())


def _print_profiles(conn, *, limit: int) -> None:
    _print_heading("Profile overview")
    c = _count(conn, "contact_profiles")
    if c is None:
        print("contact_profiles table missing")
        return
    print(f"rows={c}")
    if c == 0:
        return

    rows = conn.execute(
        """
        SELECT contact_email, profile_json, model, updated_at
        FROM contact_profiles
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    for i, (contact_email, profile_json, model, updated_at) in enumerate(rows, start=1):
        keys = _profile_keys(profile_json)
        print(
            f"{i}. contact={_truncate(contact_email, 60)} model={_truncate(model, 32)} "
            f"updated={_truncate(updated_at, 32)}"
        )
        print(f"   profile_keys={', '.join(keys) if keys else '<none/invalid-json>'}")


def _print_redaction_comparison(conn, *, limit: int) -> None:
    _print_heading("Vector/chunk redaction comparisons")

    if _table_exists(conn, "message_vectors_v2"):
        rows = conn.execute(
            """
            SELECT msg_id, index_level, source_text, source_text_redacted
            FROM message_vectors_v2
            WHERE COALESCE(source_text, '') != COALESCE(source_text_redacted, '')
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        print(f"message_vectors_v2 differing rows={len(rows)}")
        if not rows:
            print("- no differing message-level rows found")
        for i, (msg_id, index_level, raw, redacted) in enumerate(rows, start=1):
            print(f"  {i}. msg_id={msg_id} index_level={index_level}")
            print(f"     raw:      {_truncate(raw, 180)}")
            print(f"     redacted: {_truncate(redacted, 180)}")
    else:
        print("message_vectors_v2 table missing")

    if _table_exists(conn, "message_chunk_vectors_v2"):
        rows = conn.execute(
            """
            SELECT chunk_id, index_level, msg_id, chunk_text, chunk_text_redacted
            FROM message_chunk_vectors_v2
            WHERE COALESCE(chunk_text, '') != COALESCE(chunk_text_redacted, '')
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        print(f"message_chunk_vectors_v2 differing rows={len(rows)}")
        if not rows:
            print("- no differing chunk-level rows found")
        for i, (chunk_id, index_level, msg_id, raw, redacted) in enumerate(rows, start=1):
            print(f"  {i}. chunk_id={chunk_id} index_level={index_level} msg_id={msg_id}")
            print(f"     raw:      {_truncate(raw, 180)}")
            print(f"     redacted: {_truncate(redacted, 180)}")
    else:
        print("message_chunk_vectors_v2 table missing")


def _print_lexical_preview(conn, query: str, *, limit: int) -> None:
    _print_heading(f"Lexical preview for query={query!r}")
    if not _table_exists(conn, "message_fts"):
        print("message_fts table missing")
        return

    sql = """
        SELECT f.msg_id, m.subject, m.snippet, bm25(message_fts) AS bm25_score
        FROM message_fts f
        JOIN messages m ON m.msg_id = f.msg_id
        WHERE message_fts MATCH ?
        ORDER BY bm25_score
        LIMIT ?
        """

    def _run(q: str):
        return conn.execute(sql, (q, limit)).fetchall()

    try:
        rows = _run(query)
    except Exception:
        sanitized = query.replace('"', "")
        rows = _run(f'"{sanitized}"')

    if not rows:
        print("(no lexical matches)")
        return

    for i, (msg_id, subject, snippet, score) in enumerate(rows, start=1):
        print(f"{i}. msg_id={msg_id} bm25={score:.4f}")
        print(f"   subject={_truncate(subject, 120)}")
        print(f"   snippet={_truncate(snippet, 140)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="db_inspect.py",
        description=(
            "Inspect an encrypted inbox-vault SQLCipher DB with safe, truncated output."
        ),
    )
    parser.add_argument("--db", required=True, help="Path to SQLCipher DB file")
    parser.add_argument(
        "--key-env",
        default="INBOX_VAULT_DB_PASSWORD",
        help="Environment variable containing SQLCipher passphrase (default: INBOX_VAULT_DB_PASSWORD)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Sample row limit per section (default: 5)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional lexical query preview against message_fts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        print(f"ERROR: DB file not found: {db_path}", file=sys.stderr)
        return 2

    key = os.getenv(args.key_env, "")
    if not key:
        print(
            f"ERROR: environment variable {args.key_env} is empty/missing. "
            f"Set it to your SQLCipher DB passphrase.",
            file=sys.stderr,
        )
        return 2

    conn = sqlite.connect(str(db_path), timeout=30.0, check_same_thread=False)
    try:
        escaped_key = _sqlcipher_quote(key)
        conn.execute(f"PRAGMA key='{escaped_key}';")
        try:
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except sqlite.DatabaseError as exc:
            print(
                f"ERROR: unable to decrypt/open DB ({exc}). Check --db and --key-env.",
                file=sys.stderr,
            )
            return 2

        safe_limit = max(1, int(args.limit))
        print(f"Inspecting DB: {db_path}")
        print(f"Key env var: {args.key_env}")

        _print_table_counts(conn)
        _print_latest_messages(conn, limit=safe_limit)
        _print_enrichment(conn, limit=safe_limit)
        _print_profiles(conn, limit=safe_limit)
        _print_redaction_comparison(conn, limit=safe_limit)
        if args.query:
            _print_lexical_preview(conn, args.query, limit=safe_limit)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
