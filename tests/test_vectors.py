from __future__ import annotations

from inbox_vault.db import DBLockRetryExhausted, upsert_message, vector_level_counts
from inbox_vault.redaction import redact_text
from inbox_vault.vectors import (
    INDEX_LEVEL_FULL,
    INDEX_LEVEL_REDACTED,
    count_pending_vector_updates,
    index_vectors,
    search_vectors,
)


def _insert_msg(
    conn,
    *,
    msg_id: str,
    account: str,
    labels: list[str],
    subject: str,
    body: str,
    internal_ts: int = 1,
    date_iso: str = "2026-01-01T00:00:00Z",
):
    upsert_message(
        conn,
        {
            "msg_id": msg_id,
            "account_email": account,
            "thread_id": f"thread-{msg_id}",
            "date_iso": date_iso,
            "internal_ts": internal_ts,
            "from_addr": "sender@example.com",
            "to_addr": "recipient@example.com",
            "subject": subject,
            "snippet": body[:40],
            "body_text": body,
            "labels": labels,
            "history_id": 1,
        },
    )


def test_index_and_search_with_scope_and_clearance(conn, app_cfg, monkeypatch):
    _insert_msg(
        conn,
        msg_id="m-inbox",
        account="acct@example.com",
        labels=["INBOX"],
        subject="Project alpha",
        body="Contact me at alice@example.com or +1 (555) 222-1111 and visit https://internal.local",
    )
    _insert_msg(
        conn,
        msg_id="m-sent",
        account="acct@example.com",
        labels=["SENT"],
        subject="Travel plans",
        body="This is unrelated to project alpha",
    )
    _insert_msg(
        conn,
        msg_id="m-other-account",
        account="other@example.com",
        labels=["INBOX"],
        subject="Project alpha external",
        body="Other account message",
    )
    conn.commit()

    def fake_embed(_cfg, text: str) -> list[float]:
        text_l = text.lower()
        if "project alpha" in text_l:
            return [1.0, 0.0]
        if "project" in text_l:
            return [0.9, 0.1]
        if "travel" in text_l:
            return [0.1, 0.9]
        return [0.0, 1.0]

    monkeypatch.setattr("inbox_vault.vectors.embedding_vector", fake_embed)

    stats = index_vectors(conn, app_cfg)
    assert stats["indexed"] == 3
    assert stats["failed"] == 0

    # Scope filter should include only acct@example.com + INBOX
    redacted = search_vectors(
        conn,
        app_cfg,
        "project",
        account_email="acct@example.com",
        label="INBOX",
        clearance="redacted",
    )
    assert len(redacted) == 1
    first = redacted[0]
    assert first.msg_id == "m-inbox"
    assert ("[REDACTED_EMAIL]" in first.content) or ("<REDACTED_EMAIL_" in first.content)
    assert ("[REDACTED_PHONE]" in first.content) or ("<REDACTED_PHONE_" in first.content)
    assert ("[REDACTED_URL]" in first.content) or ("<REDACTED_URL_" in first.content)

    full = search_vectors(
        conn,
        app_cfg,
        "project",
        account_email="acct@example.com",
        label="INBOX",
        clearance="full",
    )
    assert len(full) == 1
    assert "alice@example.com" in full[0].content


def test_default_index_builds_redacted_level_only(conn, app_cfg, monkeypatch):
    _insert_msg(
        conn,
        msg_id="m-redacted-default",
        account="acct@example.com",
        labels=["INBOX"],
        subject="Project alpha",
        body="Reach alice@example.com about the project",
    )
    conn.commit()

    monkeypatch.setattr("inbox_vault.vectors.embedding_vector", lambda *_a, **_k: [1.0, 0.0])

    stats = index_vectors(conn, app_cfg)
    assert stats["index_level"] == INDEX_LEVEL_REDACTED

    counts = vector_level_counts(conn)
    assert counts[INDEX_LEVEL_REDACTED]["messages"] == 1
    assert INDEX_LEVEL_FULL not in counts


def test_full_clearance_can_fallback_to_redacted_rank_with_diagnostics(conn, app_cfg, monkeypatch):
    _insert_msg(
        conn,
        msg_id="m-fallback",
        account="acct@example.com",
        labels=["INBOX"],
        subject="Project alpha",
        body="Contact alice@example.com about invoice 998877.",
    )
    conn.commit()

    monkeypatch.setattr("inbox_vault.vectors.embedding_vector", lambda *_a, **_k: [1.0, 0.0])
    index_vectors(conn, app_cfg, index_level=INDEX_LEVEL_REDACTED)

    rows, diagnostics = search_vectors(
        conn,
        app_cfg,
        "project alpha",
        clearance="full",
        include_diagnostics=True,
    )
    assert len(rows) == 1
    assert "alice@example.com" in rows[0].content
    assert diagnostics.used_level == INDEX_LEVEL_REDACTED
    assert diagnostics.fallback_from_level == INDEX_LEVEL_FULL
    assert diagnostics.full_level_available is False


def test_explicit_full_search_level_errors_when_unavailable(conn, app_cfg, monkeypatch):
    _insert_msg(
        conn,
        msg_id="m-no-full",
        account="acct@example.com",
        labels=["INBOX"],
        subject="Project alpha",
        body="Contact alice@example.com",
    )
    conn.commit()

    monkeypatch.setattr("inbox_vault.vectors.embedding_vector", lambda *_a, **_k: [1.0, 0.0])
    index_vectors(conn, app_cfg, index_level=INDEX_LEVEL_REDACTED)

    import pytest

    with pytest.raises(ValueError, match="full search level is unavailable"):
        search_vectors(
            conn,
            app_cfg,
            "project alpha",
            clearance="full",
            search_level=INDEX_LEVEL_FULL,
        )


def test_search_applies_date_range_filters_for_dense_and_lexical(conn, app_cfg, monkeypatch):
    _insert_msg(
        conn,
        msg_id="m-old-window",
        account="acct@example.com",
        labels=["INBOX"],
        subject="Project weekly update",
        body="Old project details",
        internal_ts=1772323200000,
        date_iso="2026-03-01T00:00:00+00:00",
    )
    _insert_msg(
        conn,
        msg_id="m-new-window",
        account="acct@example.com",
        labels=["INBOX"],
        subject="Project weekly update",
        body="New project details",
        internal_ts=1773014400000,
        date_iso="2026-03-09T00:00:00+00:00",
    )
    conn.commit()

    monkeypatch.setattr(
        "inbox_vault.vectors.embedding_vector", lambda *_a, **_k: [1.0, 0.0]
    )
    stats = index_vectors(conn, app_cfg)
    assert stats["indexed"] == 2

    dense_rows = search_vectors(
        conn,
        app_cfg,
        "project",
        strategy="dense",
        from_ts_ms=1772841600000,
        to_ts_ms=1773273600000,
        top_k=5,
    )
    assert [row.msg_id for row in dense_rows] == ["m-new-window"]

    lexical_rows = search_vectors(
        conn,
        app_cfg,
        "project",
        strategy="lexical",
        from_ts_ms=1772841600000,
        to_ts_ms=1773273600000,
        top_k=5,
    )
    assert [row.msg_id for row in lexical_rows] == ["m-new-window"]


def test_index_vectors_surfaces_lock_diagnostics(conn, app_cfg, monkeypatch):
    _insert_msg(
        conn,
        msg_id="m-lock-diag",
        account="acct@example.com",
        labels=["INBOX"],
        subject="Lock diag",
        body="project details",
    )
    conn.commit()

    monkeypatch.setattr(
        "inbox_vault.vectors.embedding_vector", lambda *_a, **_k: [1.0, 0.0]
    )
    stats = index_vectors(conn, app_cfg)
    assert "lock_retries" in stats
    assert "lock_errors" in stats
    assert stats["lock_errors"] == 0


def test_index_vectors_skips_unchanged_on_rerun_and_pending_counter_matches(
    conn, app_cfg, monkeypatch
):
    _insert_msg(
        conn,
        msg_id="m-rerun",
        account="acct@example.com",
        labels=["INBOX"],
        subject="Rerun test",
        body="Project details for rerun idempotency",
    )
    conn.commit()

    monkeypatch.setattr(
        "inbox_vault.vectors.embedding_vector", lambda *_a, **_k: [1.0, 0.0]
    )

    pending_before = count_pending_vector_updates(conn, app_cfg, index_level=INDEX_LEVEL_REDACTED)
    assert pending_before == 1

    first = index_vectors(conn, app_cfg)
    assert first["indexed"] == 1

    pending_after_first = count_pending_vector_updates(
        conn, app_cfg, index_level=INDEX_LEVEL_REDACTED
    )
    assert pending_after_first == 0

    second = index_vectors(conn, app_cfg)
    assert second["indexed"] == 0
    assert second["unchanged"] == 1

    pending_after_second = count_pending_vector_updates(
        conn, app_cfg, index_level=INDEX_LEVEL_REDACTED
    )
    assert pending_after_second == 0


def test_index_vectors_pending_only_limits_to_pending_rows(conn, app_cfg, monkeypatch):
    _insert_msg(
        conn,
        msg_id="m-pend-1",
        account="acct@example.com",
        labels=["INBOX"],
        subject="Pending one",
        body="Project details one",
    )
    _insert_msg(
        conn,
        msg_id="m-pend-2",
        account="acct@example.com",
        labels=["INBOX"],
        subject="Pending two",
        body="Project details two",
    )
    _insert_msg(
        conn,
        msg_id="m-pend-3",
        account="acct@example.com",
        labels=["INBOX"],
        subject="Pending three",
        body="Project details three",
    )
    conn.commit()

    monkeypatch.setattr(
        "inbox_vault.vectors.embedding_vector", lambda *_a, **_k: [1.0, 0.0]
    )

    first = index_vectors(conn, app_cfg, pending_only=True, limit=2)
    assert first["indexed"] == 2
    assert first["unchanged"] == 0
    assert count_pending_vector_updates(conn, app_cfg, index_level=INDEX_LEVEL_REDACTED) == 1

    second = index_vectors(conn, app_cfg, pending_only=True, limit=2)
    assert second["indexed"] == 1
    assert second["unchanged"] == 0
    assert count_pending_vector_updates(conn, app_cfg, index_level=INDEX_LEVEL_REDACTED) == 0


def test_index_vectors_lock_exhaustion_marks_failed(conn, app_cfg, monkeypatch):
    _insert_msg(
        conn,
        msg_id="m-lock-fail",
        account="acct@example.com",
        labels=["INBOX"],
        subject="Lock fail",
        body="project details",
    )
    conn.commit()

    monkeypatch.setattr(
        "inbox_vault.vectors.embedding_vector", lambda *_a, **_k: [1.0, 0.0]
    )
    monkeypatch.setattr(
        "inbox_vault.vectors.upsert_message_vector",
        lambda *_a, **_k: (_ for _ in ()).throw(DBLockRetryExhausted("forced")),
    )

    stats = index_vectors(conn, app_cfg)
    assert stats["failed"] == 1
    assert stats["lock_errors"] == 1
    assert stats["indexed"] == 0


def test_index_vectors_commits_incrementally_by_default(conn, app_cfg, monkeypatch):
    _insert_msg(
        conn,
        msg_id="m-commit-1",
        account="acct@example.com",
        labels=["INBOX"],
        subject="Commit one",
        body="project details one",
    )
    _insert_msg(
        conn,
        msg_id="m-commit-2",
        account="acct@example.com",
        labels=["INBOX"],
        subject="Commit two",
        body="project details two",
    )
    conn.commit()

    monkeypatch.setattr(
        "inbox_vault.vectors.embedding_vector", lambda *_a, **_k: [1.0, 0.0]
    )

    from inbox_vault import vectors as vectors_mod

    original_upsert = vectors_mod.upsert_message_vector
    seen_msg_ids: list[str] = []

    def interrupt_on_second(*args, **kwargs):
        msg_id = kwargs["msg_id"]
        seen_msg_ids.append(msg_id)
        if msg_id == "m-commit-2":
            raise KeyboardInterrupt("simulated interruption")
        return original_upsert(*args, **kwargs)

    monkeypatch.setattr("inbox_vault.vectors.upsert_message_vector", interrupt_on_second)

    try:
        index_vectors(conn, app_cfg)
    except KeyboardInterrupt:
        pass

    assert "m-commit-1" in seen_msg_ids
    assert "m-commit-2" in seen_msg_ids

    committed_ids = {
        row[0]
        for row in conn.execute(
            "SELECT msg_id FROM message_vectors WHERE index_level = ? AND msg_id LIKE 'm-commit-%'",
            (INDEX_LEVEL_REDACTED,),
        ).fetchall()
    }
    assert committed_ids == {"m-commit-1"}


def test_regex_redactor_fallback_patterns():
    text = "Email bob@example.com call 212-555-1234 url https://example.com account acct-99887766"
    redacted = redact_text(text)
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "[REDACTED_URL]" in redacted
    assert "[REDACTED_ACCOUNT]" in redacted


def test_hybrid_rrf_combines_dense_and_lexical(conn, app_cfg, monkeypatch):
    _insert_msg(
        conn,
        msg_id="dense-top",
        account="acct@example.com",
        labels=["INBOX"],
        subject="Roadmap planning",
        body="Alpha project roadmap and milestones",
    )
    _insert_msg(
        conn,
        msg_id="lexical-top",
        account="acct@example.com",
        labels=["INBOX"],
        subject="Budget approval",
        body="Contains the unique phrase zebra-approval-token",
    )
    conn.commit()

    def fake_embed(_cfg, text: str) -> list[float]:
        text_l = text.lower()
        if "query" in text_l:
            return [1.0, 0.0]
        if "roadmap" in text_l:
            return [0.95, 0.05]
        if "zebra-approval-token" in text_l:
            return [0.1, 0.9]
        return [0.0, 1.0]

    monkeypatch.setattr("inbox_vault.vectors.embedding_vector", fake_embed)

    stats = index_vectors(conn, app_cfg)
    assert stats["indexed"] == 2

    rows = search_vectors(
        conn,
        app_cfg,
        "query zebra-approval-token",
        strategy="hybrid",
        top_k=2,
    )
    ids = [row.msg_id for row in rows]
    assert "dense-top" in ids
    assert "lexical-top" in ids


def test_reranker_boundary_with_mock(conn, app_cfg, monkeypatch):
    _insert_msg(
        conn,
        msg_id="m-a",
        account="acct@example.com",
        labels=["INBOX"],
        subject="A",
        body="topic alpha",
    )
    _insert_msg(
        conn,
        msg_id="m-b",
        account="acct@example.com",
        labels=["INBOX"],
        subject="B",
        body="topic beta",
    )
    conn.commit()

    monkeypatch.setattr(
        "inbox_vault.vectors.embedding_vector", lambda *_a, **_k: [1.0, 0.0]
    )
    index_vectors(conn, app_cfg)

    app_cfg.rerank.enabled = True
    app_cfg.rerank.top_n = 2

    class FakeCrossEncoder:
        def predict(self, pairs):
            # reverse the first-stage order deterministically
            assert len(pairs) == 2
            return [0.1, 0.9]

    monkeypatch.setattr(
        "inbox_vault.vectors._load_cross_encoder",
        lambda _model_name: FakeCrossEncoder(),
    )

    rows = search_vectors(conn, app_cfg, "topic", strategy="dense", top_k=2)
    assert rows[0].msg_id == "m-b"


def test_index_vectors_model_mode_uses_redaction_overrides(conn, app_cfg, monkeypatch):
    _insert_msg(
        conn,
        msg_id="m-model",
        account="acct@example.com",
        labels=["INBOX"],
        subject="Project",
        body="Please email alice@example.com",
    )
    conn.commit()

    monkeypatch.setattr(
        "inbox_vault.vectors.embedding_vector", lambda *_args, **_kwargs: [1.0, 0.0]
    )

    captured: dict[str, str] = {}

    class _Result:
        source_text_redacted = "MODEL_REDACTED::source"
        chunk_text_redacted = ["MODEL_REDACTED::chunk"]
        inserted_entries = []
        persisted_entries = []

    def fake_redact_pipeline(source_text: str, **kwargs):
        captured["mode"] = kwargs["mode"]
        captured["profile"] = kwargs["profile"]
        captured["instruction"] = kwargs["instruction"]
        return _Result()

    monkeypatch.setattr(
        "inbox_vault.vectors.redact_with_persistent_map", fake_redact_pipeline
    )

    stats = index_vectors(
        conn,
        app_cfg,
        redaction_mode="model",
        redaction_profile="secret",
        redaction_instruction="Mask names",
    )
    assert stats["indexed"] == 1
    assert stats["chunks_indexed"] >= 1
    assert captured == {"mode": "model", "profile": "secret", "instruction": "Mask names"}

    rows = search_vectors(conn, app_cfg, "project", clearance="redacted", strategy="dense")
    assert rows[0].content.startswith("MODEL_REDACTED::")


def test_index_vectors_uses_single_combined_redaction_pass_per_message(conn, app_cfg, monkeypatch):
    app_cfg.retrieval.chunk_chars = 12
    app_cfg.retrieval.chunk_overlap_chars = 4
    _insert_msg(
        conn,
        msg_id="m-combined",
        account="acct@example.com",
        labels=["INBOX"],
        subject="alpha",
        body=("one two three four five six seven eight nine ten " * 5).strip(),
    )
    conn.commit()

    monkeypatch.setattr(
        "inbox_vault.vectors.embedding_vector", lambda *_a, **_k: [1.0, 0.0]
    )

    calls = {"count": 0, "chunk_count": 0}

    class _Result:
        source_text_redacted = "REDACTED"

        def __init__(self, chunk_count: int):
            self.chunk_text_redacted = ["REDACTED"] * chunk_count
            self.inserted_entries = []
            self.persisted_entries = []

    def fake_pipeline(_source_text: str, **kwargs):
        calls["count"] += 1
        calls["chunk_count"] = len(kwargs["chunks"])
        return _Result(len(kwargs["chunks"]))

    monkeypatch.setattr("inbox_vault.vectors.redact_with_persistent_map", fake_pipeline)

    stats = index_vectors(conn, app_cfg)
    assert stats["indexed"] == 1
    assert calls["count"] == 1
    assert calls["chunk_count"] >= 2


def test_chunk_indexing_with_overlap_and_message_aggregation(conn, app_cfg, monkeypatch):
    app_cfg.retrieval.chunk_chars = 12
    app_cfg.retrieval.chunk_overlap_chars = 4

    _insert_msg(
        conn,
        msg_id="m-chunks",
        account="acct@example.com",
        labels=["INBOX"],
        subject="alpha",
        body=("one two three four five six seven eight nine ten " * 20).strip(),
    )
    conn.commit()

    def fake_embed(_cfg, text: str) -> list[float]:
        lowered = text.lower()
        if lowered.startswith("subject:"):
            return [0.1, 0.2]
        if "six seven" in lowered:
            return [1.0, 0.0]
        if "seven eight" in lowered:
            return [0.95, 0.05]
        if "query" in lowered:
            return [1.0, 0.0]
        return [0.0, 1.0]

    monkeypatch.setattr("inbox_vault.vectors.embedding_vector", fake_embed)

    stats = index_vectors(conn, app_cfg)
    assert stats["indexed"] == 1
    assert stats["chunks_indexed"] >= 3

    rows = conn.execute(
        """
        SELECT chunk_index, chunk_start, chunk_end, chunk_type
        FROM message_chunk_vectors
        WHERE msg_id = 'm-chunks' AND index_level = ?
        ORDER BY chunk_index
        """,
        (INDEX_LEVEL_REDACTED,),
    ).fetchall()
    assert rows[0][3] == "subject"
    # body chunks should overlap: each following start is less than previous end
    for (_, prev_start, prev_end, _), (_, next_start, _next_end, _type) in zip(rows[1:], rows[2:]):
        assert next_start < prev_end
        assert next_start >= prev_start

    out = search_vectors(conn, app_cfg, "query", strategy="dense", top_k=1)
    assert len(out) == 1
    assert out[0].msg_id == "m-chunks"


def test_index_vectors_filters_labels_before_embedding(conn, app_cfg, monkeypatch):
    _insert_msg(
        conn,
        msg_id="m-inbox-ok",
        account="acct@example.com",
        labels=["INBOX"],
        subject="Ship update",
        body="Project shipping details",
    )
    _insert_msg(
        conn,
        msg_id="m-spam",
        account="acct@example.com",
        labels=["SPAM"],
        subject="Win now",
        body="Promotional body",
    )
    _insert_msg(
        conn,
        msg_id="m-promotions",
        account="acct@example.com",
        labels=["CATEGORY_PROMOTIONS"],
        subject="Coupon",
        body="Buy now",
    )
    conn.commit()

    embed_calls = {"count": 0}

    def fake_embed(_cfg, _text):
        embed_calls["count"] += 1
        return [1.0, 0.0]

    monkeypatch.setattr("inbox_vault.vectors.embedding_vector", fake_embed)

    stats = index_vectors(conn, app_cfg)
    assert stats["scanned"] == 3
    assert stats["indexed"] == 1
    assert stats["skipped_filtered"] == 2
    # one message vector + at least one chunk for the only indexed message
    assert embed_calls["count"] >= 2


def test_index_vectors_normalizes_and_trims_text(conn, app_cfg, monkeypatch):
    app_cfg.indexing.max_index_chars = 64

    noisy_body = "One\u200b two\u200d   three\n\n\t" + ("very long boilerplate tail " * 20)
    _insert_msg(
        conn,
        msg_id="m-normalize",
        account="acct@example.com",
        labels=["INBOX"],
        subject="  Subject\u200b line   ",
        body=noisy_body,
    )
    conn.commit()

    monkeypatch.setattr(
        "inbox_vault.vectors.embedding_vector", lambda *_a, **_k: [1.0, 0.0]
    )

    stats = index_vectors(conn, app_cfg)
    assert stats["indexed"] == 1

    source_text = conn.execute(
        "SELECT source_text FROM message_vectors WHERE msg_id = ? AND index_level = ?",
        ("m-normalize", INDEX_LEVEL_REDACTED),
    ).fetchone()[0]
    assert "\u200b" not in source_text
    assert "\u200d" not in source_text
    assert "  " not in source_text

    body_segment = source_text.split("Body: ", 1)[1]
    assert len(body_segment) <= app_cfg.indexing.max_index_chars
