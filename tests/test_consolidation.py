from __future__ import annotations

import json
from pathlib import Path

from inbox_vault.consolidation import run_consolidation


def test_run_consolidation_happy_path(tmp_path: Path, app_cfg, monkeypatch):
    target_root = tmp_path / "base-run"

    monkeypatch.setattr(
        "inbox_vault.consolidation.backfill",
        lambda *_a, **_k: {"accounts": 1, "ingested": 2, "failed": 0, "skipped_existing": 0},
    )

    def fake_enrich(*_a, diagnostics=None, **_k):
        if diagnostics is not None:
            diagnostics.update(
                {"attempted": 2, "succeeded": 2, "http_failed": 0, "parse_failed": 0}
            )
        return 2

    monkeypatch.setattr("inbox_vault.consolidation.enrich_pending", fake_enrich)

    def fake_profiles(*_a, diagnostics=None, **_k):
        if diagnostics is not None:
            diagnostics.update(
                {"attempted": 1, "succeeded": 1, "http_failed": 0, "parse_failed": 0}
            )
        return 1

    monkeypatch.setattr("inbox_vault.consolidation.build_profiles", fake_profiles)
    monkeypatch.setattr(
        "inbox_vault.consolidation.index_vectors",
        lambda *_a, **_k: {
            "scanned": 2,
            "indexed": 2,
            "unchanged": 0,
            "failed": 0,
            "chunks_indexed": 3,
            "lancedb_indexed": 0,
            "lancedb_failed": 0,
            "lancedb_status": "disabled",
            "lock_retries": 0,
            "lock_errors": 0,
        },
    )

    out = run_consolidation(app_cfg, "pw", target_root=str(target_root), max_messages=7)

    assert out["ok"] is True
    assert out["max_messages"] == 7
    assert Path(out["report_path"]).exists()
    assert Path(out["artifacts"]["target_db_path"]).exists()
    assert Path(out["artifacts"]["inspect_summary_path"]).exists()

    report = json.loads(Path(out["report_path"]).read_text())
    step_names = [step["name"] for step in report["steps"]]
    assert step_names == [
        "backfill",
        "enrich",
        "build_profiles",
        "index_vectors",
        "inspect_summary_report",
    ]


def test_run_consolidation_single_writer_guard(tmp_path: Path, app_cfg):
    target_root = tmp_path / "base-run"
    target_root.mkdir(parents=True)
    (target_root / ".index-writer.lock").write_text('{"pid":123}')

    out = run_consolidation(app_cfg, "pw", target_root=str(target_root), max_messages=5)
    assert out["ok"] is False
    assert out["steps"][0]["name"] == "single_writer_guard"
    assert out["steps"][0]["status"] == "failed"
    assert Path(out["report_path"]).exists()
