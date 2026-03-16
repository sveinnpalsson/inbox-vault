from __future__ import annotations

import json
from pathlib import Path

from inbox_vault.db import upsert_message
from inbox_vault.evals import bootstrap_eval_template, run_retrieval_eval


def test_run_retrieval_eval_metrics_and_slices(conn, app_cfg, monkeypatch, tmp_path: Path):
    eval_file = tmp_path / "eval.json"
    eval_file.write_text(
        json.dumps(
            [
                {"query": "q1", "relevant_msg_ids": ["m1"], "label": "INBOX"},
                {"query": "q2", "relevant_msg_ids": ["m3"], "label": "INBOX"},
                {"query": "q3", "relevant_msg_ids": ["m4"], "scope": "sent-only"},
            ]
        )
    )

    def fake_search_vectors(_conn, _cfg, query: str, **_kwargs):
        if query == "q1":
            return [type("R", (), {"msg_id": "m1"})(), type("R", (), {"msg_id": "m2"})()]
        if query == "q2":
            return [type("R", (), {"msg_id": "m2"})(), type("R", (), {"msg_id": "m3"})()]
        return [type("R", (), {"msg_id": "x"})(), type("R", (), {"msg_id": "m4"})()]

    monkeypatch.setattr("inbox_vault.evals.search_vectors", fake_search_vectors)

    out = run_retrieval_eval(
        conn,
        app_cfg,
        eval_file=str(eval_file),
        strategy="hybrid",
        top_k=2,
    )
    assert out["cases"] == 3
    assert out["hits"] == 3
    assert out["recall_at_k"] == 1.0
    assert out["mrr_at_k"] == 0.666667
    assert out["ndcg_at_k"] == 0.753953
    assert out["slices"]["label:INBOX"]["cases"] == 2
    assert out["slices"]["label:INBOX"]["mrr_at_k"] == 0.75
    assert out["slices"]["scope:sent-only"]["cases"] == 1


def test_bootstrap_eval_template_safe_metadata_only(conn, tmp_path: Path):
    upsert_message(
        conn,
        {
            "msg_id": "m-bootstrap",
            "account_email": "acct@example.com",
            "thread_id": "t1",
            "date_iso": "2026-01-01T00:00:00Z",
            "internal_ts": 1,
            "from_addr": "sender@example.com",
            "to_addr": "recipient@example.com",
            "subject": "Private Subject",
            "snippet": "Sensitive snippet",
            "body_text": "Sensitive body should never appear",
            "labels": ["INBOX"],
            "history_id": 10,
        },
    )
    conn.commit()

    out_file = tmp_path / "eval.local.json"
    out = bootstrap_eval_template(conn, output_file=str(out_file), limit=10)

    assert out["written"] == 1
    payload = json.loads(out_file.read_text())
    assert payload[0]["relevant_msg_ids"] == ["m-bootstrap"]
    assert payload[0]["account_email"] == "acct@example.com"
    assert payload[0]["label"] == "INBOX"
    assert "snippet" not in json.dumps(payload).lower()
    assert "body" not in json.dumps(payload).lower()
