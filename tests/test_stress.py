from __future__ import annotations

from pathlib import Path

from inbox_vault.stress import run_isolated_stress


def test_run_isolated_stress_smoke(tmp_path: Path, app_cfg, monkeypatch):
    app_cfg.db.path = str(tmp_path / "base.db")

    monkeypatch.setattr(
        "inbox_vault.stress.backfill", lambda *_a, **_k: {"ingested": 3, "failed": 0}
    )
    monkeypatch.setattr("inbox_vault.stress.enrich_pending", lambda *_a, **_k: 3)
    monkeypatch.setattr("inbox_vault.stress.build_profiles", lambda *_a, **_k: 2)
    monkeypatch.setattr(
        "inbox_vault.stress.index_vectors",
        lambda *_a, **_k: {"indexed": 3, "failed": 0, "lancedb_status": "disabled"},
    )

    class _Result:
        msg_id = "m1"

    monkeypatch.setattr("inbox_vault.stress.search_vectors", lambda *_a, **_k: [_Result()])

    report = run_isolated_stress(
        app_cfg,
        "pw",
        isolated_root=str(tmp_path / "isolated"),
        max_messages=20,
        search_query="project",
    )

    assert report["ok"] is True
    assert Path(report["report_path"]).exists()
    names = [step["name"] for step in report["steps"]]
    assert names[:6] == [
        "validate",
        "backfill",
        "enrich",
        "build_profiles",
        "index_vectors",
        "semantic_search_check",
    ]
    assert report["steps"][-1]["status"] == "skipped"
