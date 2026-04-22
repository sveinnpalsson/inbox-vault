from __future__ import annotations

from inbox_vault import cli, db, enrich, vectors


def test_enrich_pending_filters_to_requested_msg_ids(tmp_path):
    conn = db.get_conn(tmp_path / "test.db", "pw")
    conn.execute(
        "INSERT INTO messages (msg_id, account_email, thread_id, internal_ts, subject, snippet, body_text, from_addr, to_addr, date_iso, labels_json, last_seen_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("m1", "acct@example.com", "t1", 1000, "hello", "hello", "body", "a@example.com", "b@example.com", "2026-01-01T00:00:00Z", "[]", "2026-01-01T00:00:00Z"),
    )
    conn.execute(
        "INSERT INTO messages (msg_id, account_email, thread_id, internal_ts, subject, snippet, body_text, from_addr, to_addr, date_iso, labels_json, last_seen_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("m2", "acct@example.com", "t2", 2000, "world", "world", "body", "a@example.com", "b@example.com", "2026-01-02T00:00:00Z", "[]", "2026-01-02T00:00:00Z"),
    )
    conn.commit()

    cfg = type("Cfg", (), {"llm": type("LLM", (), {"enabled": False})()})()

    updated = enrich.enrich_pending(conn, cfg, msg_ids=["m1"])
    assert updated == 0


def test_count_pending_vector_updates_filters_to_requested_msg_ids(tmp_path):
    conn = db.get_conn(tmp_path / "test.db", "pw")
    conn.execute(
        "INSERT INTO messages (msg_id, account_email, thread_id, internal_ts, subject, snippet, body_text, from_addr, to_addr, date_iso, labels_json, last_seen_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("m1", "acct@example.com", "t1", 1000, "hello", "hello", "body", "a@example.com", "b@example.com", "2026-01-01T00:00:00Z", "[\"INBOX\"]", "2026-01-01T00:00:00Z"),
    )
    conn.execute(
        "INSERT INTO messages (msg_id, account_email, thread_id, internal_ts, subject, snippet, body_text, from_addr, to_addr, date_iso, labels_json, last_seen_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("m2", "acct@example.com", "t2", 2000, "world", "world", "body", "a@example.com", "b@example.com", "2026-01-02T00:00:00Z", "[\"INBOX\"]", "2026-01-02T00:00:00Z"),
    )
    conn.commit()

    indexing_cfg = type(
        "IndexCfg",
        (),
        {
            "include_labels": ["INBOX"],
            "exclude_labels": [],
            "strip_zero_width": True,
            "collapse_whitespace": True,
            "max_index_chars": 20000,
        },
    )()
    cfg = type("Cfg", (), {"indexing": indexing_cfg})()

    pending = vectors.count_pending_vector_updates(conn, cfg, msg_ids=["m2"])
    assert pending == 1


def test_cli_update_with_no_new_ingest_skips_backlog_followups(monkeypatch, tmp_path, capsys):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
[[accounts]]
email = \"acct@example.com\"
credentials_file = \"fake-creds.json\"
token_file = \"fake-token.json\"

[database]
path = \"test.db\"
password_env = \"TEST_DB_PASSWORD\"
""".strip()
    )
    (tmp_path / "fake-creds.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    class _Conn:
        def close(self):
            return None

    monkeypatch.setattr(cli, "get_conn", lambda *_args, **_kwargs: _Conn())
    monkeypatch.setattr(cli, "resolve_password", lambda *_args, **_kwargs: "pw")
    monkeypatch.setattr(cli, "load_config", lambda *_args, **_kwargs: type(
        "Cfg",
        (),
        {
            "db": type("DB", (), {"path": "test.db", "password_env": "TEST_DB_PASSWORD"})(),
            "database": type("DB", (), {"path": "test.db", "password_env": "TEST_DB_PASSWORD"})(),
            "indexing": type("Indexing", (), {"auto_index_limit": 300})(),
            "ingest_triage": type("Triage", (), {"enabled": False, "mode": "observe"})(),
            "llm": type("LLM", (), {"enabled": True, "endpoint": "http://llm"})(),
            "embeddings": type("Emb", (), {"endpoint": "http://embeddings"})(),
        },
    )())
    monkeypatch.setattr(cli, "update", lambda *_args, **_kwargs: {"ingested": 0, "new_ids": 0, "ingested_msg_ids": []})
    monkeypatch.setattr(cli, "enrich_pending", lambda *_args, **_kwargs: 999)
    monkeypatch.setattr(cli, "_endpoint_reachable", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(cli, "_validate_accounts", lambda *_args, **_kwargs: None)

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("indexing backlog should not run when update ingested no new messages")

    monkeypatch.setattr(cli, "_run_index_vectors_for_ingest", _unexpected)

    cli.main(["--config", str(cfg_path), "update"])
    out = __import__("json").loads(capsys.readouterr().out)
    assert out["enrich"]["updated"] == 0
    assert out["enrich"]["scope"] == "new_ingest_only"
    assert out["index_vectors"]["indexed"] == 0
    assert out["index_vectors"]["scope"] == "new_ingest_only"
