from __future__ import annotations

import json
from pathlib import Path

import pytest

from inbox_vault import cli


@pytest.fixture(autouse=True)
def _fake_endpoints_reachable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli, "_endpoint_reachable", lambda *_args, **_kw: True)


def _write_config(path: Path):
    path.write_text(
        """
[[accounts]]
email = "acct@example.com"
credentials_file = "fake-creds.json"
token_file = "fake-token.json"

[database]
path = "test.db"
password_env = "TEST_DB_PASSWORD"
""".strip()
    )
    (path.parent / "fake-creds.json").write_text("{}")


def _search_diag(
    *,
    requested: str = "auto",
    used: str = "redacted",
    fallback: str | None = None,
    full_available: bool = False,
):
    return type(
        "Diag",
        (),
        {
            "requested_level": requested,
            "used_level": used,
            "fallback_from_level": fallback,
            "full_level_available": full_available,
        },
    )()


def test_validate_command_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    cli.main(["--config", str(cfg), "validate"])
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True


@pytest.mark.parametrize(
    ("argv", "patches", "expected"),
    [
        (
            ["backfill", "--enrich", "--build-profiles", "--profiles-use-llm"],
            {
                "backfill": lambda *_args, **_kwargs: {"ingested": 2},
                "enrich_pending": lambda *_args, **_kwargs: 2,
                "build_profiles": lambda *_args, **_kwargs: 1,
            },
            {
                "ingest": {"ingested": 2},
                "enrich": {"updated": 2, "diagnostics": {}},
                "profiles": {"updated": 1, "diagnostics": {}},
            },
        ),
        (
            ["update"],
            {
                "update": lambda *_args, **_kwargs: {"ingested": 1, "new_ids": 1},
                "enrich_pending": lambda *_args, **_kwargs: 1,
                "count_pending_vector_updates": lambda *_args, **_kwargs: 0,
                "index_vectors": lambda *_args, **_kwargs: {
                    "scanned": 0,
                    "indexed": 0,
                    "unchanged": 0,
                    "failed": 0,
                },
            },
                {
                    "ingest": {"ingested": 1, "new_ids": 1},
                    "enrich": {"updated": 1, "diagnostics": {}},
                    "index_vectors": {
                        "scanned": 0,
                        "indexed": 0,
                        "unchanged": 0,
                        "failed": 0,
                        "pending_before": 0,
                        "pending_after": 0,
                        "skip_applied_light": False,
                    },
                },
            ),
        (
            ["repair", "--no-enrich", "--no-index-vectors"],
            {
                "repair": lambda *_args, **_kwargs: {"backfill_ingested": 1, "interrupted": False},
            },
            {"ingest": {"backfill_ingested": 1, "interrupted": False}},
        ),
        (
            ["enrich", "--limit", "5"],
            {"enrich_pending": lambda *_args, **_kwargs: 5},
            {"updated": 5, "diagnostics": {}},
        ),
        (
            ["build-profiles", "--limit", "3"],
            {"build_profiles": lambda *_args, **_kwargs: 3},
            {"updated": 3, "diagnostics": {}},
        ),
        (
            ["index-vectors", "--account-email", "acct@example.com"],
            {
                "index_vectors": lambda *_args, **_kwargs: {
                    "scanned": 2,
                    "indexed": 2,
                    "unchanged": 0,
                    "failed": 0,
                }
            },
            {
                "scanned": 2,
                "indexed": 2,
                "unchanged": 0,
                "failed": 0,
                "pending_before": 0,
                "pending_after": 0,
            },
        ),
        (
            ["search", "project status", "--top-k", "1", "--strategy", "hybrid"],
            {
                "search_vectors": lambda *_args, **_kwargs: (
                    [
                        type(
                            "R",
                            (),
                            {
                                "score": 0.99,
                                "msg_id": "m1",
                                "thread_id": "t1",
                                "account_email": "acct@example.com",
                                "labels": ["INBOX"],
                                "content": "Subject: Project",
                            },
                        )(),
                    ],
                    _search_diag(),
                )
            },
            {
                "query": "project status",
                "count": 1,
                "clearance": "redacted",
                "diagnostics": {
                    "search_level_requested": "auto",
                    "search_level_used": "redacted",
                    "search_level_fallback": None,
                    "full_level_available": False,
                },
                "results": [
                    {
                        "score": 0.99,
                        "msg_id": "m1",
                        "thread_id": "t1",
                        "account_email": "acct@example.com",
                        "from_addr": None,
                        "to_addr": None,
                        "labels": ["INBOX"],
                        "content": "Subject: Project",
                    }
                ],
            },
        ),
        (
            ["eval-retrieval", "--eval-file", "docs/eval.local.json"],
            {
                "run_retrieval_eval": lambda *_args, **_kwargs: {"cases": 2, "recall_at_k": 0.5},
            },
            {"cases": 2, "recall_at_k": 0.5},
        ),
        (
            ["stress-run", "--max-messages", "20", "--search-query", "project"],
            {
                "run_isolated_stress": lambda *_args, **_kwargs: {
                    "ok": True,
                    "report_path": "/tmp/report.json",
                },
            },
            {"ok": True, "report_path": "/tmp/report.json"},
        ),
    ],
)
def test_core_commands_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    argv: list[str],
    patches: dict,
    expected: dict,
):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    for name, fn in patches.items():
        monkeypatch.setattr(cli, name, fn)

    cli.main(["--config", str(cfg), *argv])
    out = json.loads(capsys.readouterr().out)
    assert out == expected


def test_repair_backfill_limit_passthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    captured: dict[str, object] = {}

    def fake_repair(*_args, **kwargs):
        captured.update(kwargs)
        return {"backfill_ingested": 0, "interrupted": False}

    monkeypatch.setattr(cli, "repair", fake_repair)
    monkeypatch.setattr(cli, "count_pending_vector_updates", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        cli,
        "index_vectors",
        lambda *_args, **_kwargs: {"scanned": 0, "indexed": 0, "unchanged": 0, "failed": 0},
    )

    cli.main(
        ["--config", str(cfg), "repair", "--backfill-limit", "10", "--commit-every", "50"]
    )

    out = json.loads(capsys.readouterr().out)
    assert out["ingest"] == {"backfill_ingested": 0, "interrupted": False}
    assert captured["backfill_limit"] == 10
    assert captured["commit_every_messages"] == 50


def test_repair_rejects_negative_backfill_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    with pytest.raises(ValueError, match="backfill-limit"):
        cli.main(["--config", str(cfg), "repair", "--backfill-limit", "-1"])


def test_update_no_longer_accepts_idle_backfill_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    with pytest.raises(SystemExit):
        cli.main(["--config", str(cfg), "update", "--idle-backfill-limit", "5"])


def test_update_index_vectors_flag_runs_single_index_pass_and_emits_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    captured: dict[str, object] = {"calls": 0}

    monkeypatch.setattr(cli, "update", lambda *_args, **_kwargs: {"ingested": 2, "new_ids": 2})

    monkeypatch.setattr(cli, "enrich_pending", lambda *_args, **_kwargs: 0)

    pending_counts = iter([5, 1])
    monkeypatch.setattr(
        cli, "count_pending_vector_updates", lambda *_args, **_kwargs: next(pending_counts)
    )

    def fake_index_vectors(*_args, **kwargs):
        captured["calls"] = int(captured["calls"]) + 1
        captured.update(kwargs)
        return {"scanned": 4, "indexed": 4, "unchanged": 0, "failed": 0}

    monkeypatch.setattr(cli, "index_vectors", fake_index_vectors)

    cli.main(
        [
            "--config",
            str(cfg),
            "update",
            "--index-vectors",
            "--index-pending-only",
            "--index-limit",
            "25",
        ]
    )

    out = json.loads(capsys.readouterr().out)
    assert out["ingest"] == {"ingested": 2, "new_ids": 2}
    assert out["enrich"] == {"updated": 0, "diagnostics": {}}
    assert out["index_vectors"] == {
        "scanned": 4,
        "indexed": 4,
        "unchanged": 0,
        "failed": 0,
        "pending_before": 5,
        "pending_after": 1,
        "skip_applied_light": False,
    }
    assert captured["calls"] == 1
    assert captured["pending_only"] is True
    assert captured["limit"] == 25
    assert captured["skip_applied_light"] is False


def test_update_indexes_by_default_and_can_be_disabled_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    cfg.write_text(
        cfg.read_text()
        + "\n\n[indexing]\nauto_index_after_ingest = false\nauto_index_pending_only = false\nauto_index_limit = 7\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    monkeypatch.setattr(cli, "update", lambda *_args, **_kwargs: {"ingested": 0})

    call_counter = {"count": 0}

    monkeypatch.setattr(cli, "enrich_pending", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(cli, "count_pending_vector_updates", lambda *_args, **_kwargs: 0)

    captured: dict[str, object] = {}

    def fake_index_vectors(*_args, **kwargs):
        call_counter["count"] += 1
        captured.update(kwargs)
        return {"scanned": 0, "indexed": 0, "unchanged": 0, "failed": 0}

    monkeypatch.setattr(cli, "index_vectors", fake_index_vectors)

    cli.main(["--config", str(cfg), "update"])
    out_auto = json.loads(capsys.readouterr().out)
    assert "index_vectors" in out_auto
    assert call_counter["count"] == 1
    assert captured["pending_only"] is True
    assert captured["limit"] == 7
    assert captured["skip_applied_light"] is False

    cli.main(["--config", str(cfg), "update", "--no-index-vectors", "--no-enrich"])
    out_disabled = json.loads(capsys.readouterr().out)
    assert out_disabled == {"ingest": {"ingested": 0}}
    assert call_counter["count"] == 1


def test_repair_defaults_run_enrich_and_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    cfg.write_text(cfg.read_text() + "\n\n[indexing]\nauto_index_pending_only = false\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    monkeypatch.setattr(
        cli,
        "repair",
        lambda *_args, **_kwargs: {"backfill_ingested": 1, "interrupted": False},
    )

    enrich_calls = {"count": 0}
    enrich_kwargs: dict[str, object] = {}

    def fake_enrich(*_args, **kwargs):
        enrich_calls["count"] += 1
        enrich_kwargs.update(kwargs)
        return 3

    monkeypatch.setattr(cli, "enrich_pending", fake_enrich)

    monkeypatch.setattr(cli, "count_pending_vector_updates", lambda *_args, **_kwargs: 2)
    captured: dict[str, object] = {}

    def fake_index_vectors(*_args, **kwargs):
        captured.update(kwargs)
        return {"scanned": 2, "indexed": 2, "unchanged": 0, "failed": 0}

    monkeypatch.setattr(cli, "index_vectors", fake_index_vectors)

    cli.main(["--config", str(cfg), "repair", "--backfill-limit", "5"])

    out = json.loads(capsys.readouterr().out)
    assert out["ingest"]["backfill_ingested"] == 1
    assert out["enrich"]["updated"] == 3
    assert out["enrich"]["repair_scope"] == "pending+heuristic-fallback"
    assert out["index_vectors"]["indexed"] == 2
    assert enrich_calls["count"] == 1
    assert enrich_kwargs["include_degraded"] is True
    assert captured["pending_only"] is True
    assert captured["skip_applied_light"] is False


def test_repair_interrupted_skips_post_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    monkeypatch.setattr(
        cli,
        "repair",
        lambda *_args, **_kwargs: {"backfill_ingested": 1, "interrupted": True},
    )

    monkeypatch.setattr(cli, "enrich_pending", lambda *_args, **_kwargs: pytest.fail("no enrich"))
    monkeypatch.setattr(cli, "index_vectors", lambda *_args, **_kwargs: pytest.fail("no index"))

    cli.main(["--config", str(cfg), "repair", "--backfill-limit", "5"])

    out = json.loads(capsys.readouterr().out)
    assert out["ingest"]["interrupted"] is True
    assert "warning" in out


def test_backfill_rejects_invalid_index_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    with pytest.raises(ValueError, match="index-limit"):
        cli.main(["--config", str(cfg), "backfill", "--index-vectors", "--index-limit", "0"])


def test_search_strategy_passthrough(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    captured: dict[str, object] = {}

    def fake_search_vectors(*_args, **kwargs):
        captured.update(kwargs)
        return [], _search_diag()

    monkeypatch.setattr(cli, "search_vectors", fake_search_vectors)

    cli.main(["--config", str(cfg), "search", "query text", "--strategy", "lexical"])

    out = json.loads(capsys.readouterr().out)
    assert out["count"] == 0
    assert captured["strategy"] == "lexical"
    assert captured["include_diagnostics"] is True


def test_search_level_passthrough(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    captured: dict[str, object] = {}

    def fake_search_vectors(*_args, **kwargs):
        captured.update(kwargs)
        return [], _search_diag(requested="full", used="full", full_available=True)

    monkeypatch.setattr(cli, "search_vectors", fake_search_vectors)

    cli.main(
        [
            "--config",
            str(cfg),
            "search",
            "query text",
            "--clearance",
            "full",
            "--search-level",
            "full",
        ]
    )

    out = json.loads(capsys.readouterr().out)
    assert out["diagnostics"]["search_level_used"] == "full"
    assert captured["search_level"] == "full"


def test_search_date_range_passthrough(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    captured: dict[str, object] = {}

    def fake_search_vectors(*_args, **kwargs):
        captured.update(kwargs)
        return [], _search_diag()

    monkeypatch.setattr(cli, "search_vectors", fake_search_vectors)

    cli.main(
        [
            "--config",
            str(cfg),
            "search",
            "query text",
            "--from-date",
            "2026-03-01",
            "--to-date",
            "2026-03-03",
        ]
    )

    out = json.loads(capsys.readouterr().out)
    assert out["count"] == 0
    assert captured["from_ts_ms"] == 1772323200000
    # date-only --to-date is exclusive next-day midnight UTC
    assert captured["to_ts_ms"] == 1772582400000


def test_search_date_range_rejects_invalid_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    with pytest.raises(ValueError, match="from-date"):
        cli.main(
            [
                "--config",
                str(cfg),
                "search",
                "query text",
                "--from-date",
                "2026-03-03",
                "--to-date",
                "2026-03-02",
            ]
        )


def test_build_profiles_rebuild_clears_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    captured: dict[str, object] = {}

    def fake_clear_contact_profiles(*_args, **_kwargs):
        return 7

    def fake_build_profiles(*_args, **kwargs):
        captured.update(kwargs)
        return 3

    monkeypatch.setattr(cli, "clear_contact_profiles", fake_clear_contact_profiles)
    monkeypatch.setattr(cli, "build_profiles", fake_build_profiles)

    cli.main(["--config", str(cfg), "build-profiles", "--use-llm", "--limit", "3", "--rebuild"])

    out = json.loads(capsys.readouterr().out)
    assert out == {
        "updated": 3,
        "diagnostics": {"cleared_before_rebuild": 7},
    }
    assert captured["use_llm"] is True
    assert captured["limit"] == 3
    assert "diagnostics" in captured


def test_consolidate_run_passthrough(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    captured: dict[str, object] = {}

    def fake_consolidate(*_args, **kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "report_path": "/tmp/consolidation-report.json",
            "artifacts": {"target_db_path": "/tmp/run/inbox_vault.db"},
        }

    monkeypatch.setattr(cli, "run_consolidation", fake_consolidate)

    cli.main(
        [
            "--config",
            str(cfg),
            "consolidate-run",
            "--target-root",
            "/tmp/run",
            "--max-messages",
            "11",
            "--redaction-mode",
            "hybrid",
        ]
    )

    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["report_path"] == "/tmp/consolidation-report.json"
    assert captured["target_root"] == "/tmp/run"
    assert captured["max_messages"] == 11
    assert captured["redaction_mode"] == "hybrid"


def test_eval_bootstrap_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    monkeypatch.setattr(
        cli,
        "bootstrap_eval_template",
        lambda *_args, **_kwargs: {"written": 2, "output_file": "docs/eval.local.json"},
    )

    cli.main(["--config", str(cfg), "eval-bootstrap", "--output-file", "docs/eval.local.json"])

    out = json.loads(capsys.readouterr().out)
    assert out["written"] == 2


def test_backfill_auto_index_uses_config_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    cfg.write_text(
        cfg.read_text()
        + "\n\n[indexing]\nauto_index_after_ingest = true\nauto_index_pending_only = false\nauto_index_limit = 11\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    monkeypatch.setattr(cli, "backfill", lambda *_args, **_kwargs: {"ingested": 3})
    monkeypatch.setattr(cli, "count_pending_vector_updates", lambda *_args, **_kwargs: 0)

    captured: dict[str, object] = {}

    def fake_index_vectors(*_args, **kwargs):
        captured.update(kwargs)
        return {"scanned": 3, "indexed": 2, "unchanged": 1, "failed": 0}

    monkeypatch.setattr(cli, "index_vectors", fake_index_vectors)

    cli.main(["--config", str(cfg), "backfill"])

    out = json.loads(capsys.readouterr().out)
    assert out["ingest"] == {"ingested": 3}
    assert out["index_vectors"]["indexed"] == 2
    assert captured["pending_only"] is False
    assert captured["limit"] == 11
    assert captured["skip_applied_light"] is False


def test_update_auto_index_skips_applied_light_when_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    cfg.write_text(
        cfg.read_text()
        + "\n\n[indexing]\nauto_index_after_ingest = true\nauto_index_pending_only = true\n"
        + "\n[ingest_triage]\nenabled = true\nmode = \"enforce\"\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    monkeypatch.setattr(cli, "update", lambda *_args, **_kwargs: {"ingested": 1, "new_ids": 1})
    monkeypatch.setattr(cli, "enrich_pending", lambda *_args, **_kwargs: 0)

    seen_pending_kwargs: list[dict[str, object]] = []

    def fake_pending(*_args, **kwargs):
        seen_pending_kwargs.append(dict(kwargs))
        return 0

    monkeypatch.setattr(cli, "count_pending_vector_updates", fake_pending)

    captured: dict[str, object] = {}

    def fake_index_vectors(*_args, **kwargs):
        captured.update(kwargs)
        return {
            "scanned": 0,
            "indexed": 0,
            "unchanged": 0,
            "failed": 0,
            "skipped_triage_light": 1,
        }

    monkeypatch.setattr(cli, "index_vectors", fake_index_vectors)

    cli.main(["--config", str(cfg), "update"])
    out = json.loads(capsys.readouterr().out)

    assert out["index_vectors"]["skip_applied_light"] is True
    assert captured["skip_applied_light"] is True
    assert all(kwargs["skip_applied_light"] is True for kwargs in seen_pending_kwargs)


def test_backfill_max_messages_passthrough_and_help_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    captured: dict[str, object] = {}

    def fake_backfill(*_args, **kwargs):
        captured.update(kwargs)
        return {"ingested": 0}

    monkeypatch.setattr(cli, "backfill", fake_backfill)

    cli.main(["--config", str(cfg), "backfill", "--max-messages", "7"])
    out = json.loads(capsys.readouterr().out)
    assert out == {"ingest": {"ingested": 0}}
    assert captured["max_messages"] == 7

    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["backfill", "--help"])
    help_out = capsys.readouterr().out
    assert "Global backfill cap" in help_out
    assert "configured accounts" in help_out


def test_index_vectors_redaction_flags_passthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")

    captured: dict[str, object] = {}

    def fake_index_vectors(*_args, **kwargs):
        captured.update(kwargs)
        return {"scanned": 1, "indexed": 1, "unchanged": 0, "failed": 0}

    monkeypatch.setattr(cli, "index_vectors", fake_index_vectors)

    cli.main(
        [
            "--config",
            str(cfg),
            "index-vectors",
            "--pending-only",
            "--index-level",
            "full",
            "--redaction-mode",
            "model",
            "--redaction-profile",
            "confidential",
            "--redaction-instruction",
            "Mask names and project codenames",
            "--include-label",
            "INBOX",
            "--include-label",
            "SENT,IMPORTANT",
            "--exclude-label",
            "SPAM",
            "--max-index-chars",
            "4096",
        ]
    )

    out = json.loads(capsys.readouterr().out)
    assert out["indexed"] == 1
    assert captured["pending_only"] is True
    assert captured["index_level"] == "full"
    assert captured["redaction_mode"] == "model"
    assert captured["redaction_profile"] == "confidential"
    assert captured["redaction_instruction"] == "Mask names and project codenames"
    assert captured["include_labels"] == ["INBOX", "SENT", "IMPORTANT"]
    assert captured["exclude_labels"] == ["SPAM"]
    assert captured["max_index_chars"] == 4096


def _seed_retrieval_data(tmp_path: Path):
    conn = cli.get_conn(str(tmp_path / "test.db"), "pw")
    try:
        conn.execute(
            """
            INSERT INTO messages (
              msg_id, account_email, thread_id, date_iso, internal_ts, from_addr, to_addr,
              subject, snippet, body_text, labels_json, history_id, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "m-new",
                "acct@example.com",
                "t1",
                "2026-03-10T10:00:00+00:00",
                1773136800000,
                "boss@example.com",
                "acct@example.com",
                "Urgent quarterly planning update",
                "Need final numbers before EOD and board review.",
                "This body contains alice@example.com, +1 (555) 222-1111, and https://internal.local details.",
                json.dumps(["INBOX", "IMPORTANT"]),
                1,
                "2026-03-10T10:01:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO messages (
              msg_id, account_email, thread_id, date_iso, internal_ts, from_addr, to_addr,
              subject, snippet, body_text, labels_json, history_id, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "m-old",
                "acct@example.com",
                "t2",
                "2026-03-01T10:00:00+00:00",
                1772359200000,
                "friend@example.com",
                "acct@example.com",
                "Dinner plans",
                "Want to meet this weekend?",
                "Let's grab dinner at 7.",
                json.dumps(["INBOX"]),
                1,
                "2026-03-01T10:01:00+00:00",
            ),
        )
        conn.execute(
            "INSERT INTO message_vectors (msg_id, index_level, account_email, thread_id, labels_json, source_text, source_text_redacted, embedding_json, embedding_dim, embedding_model, content_hash, redaction_policy_version, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "m-new",
                "redacted",
                "acct@example.com",
                "t1",
                json.dumps(["INBOX"]),
                "source",
                "redacted",
                json.dumps([0.1, 0.2]),
                2,
                "test-model",
                "hash-1",
                "2026-03-22-precision-1",
                "2026-03-10T10:02:00+00:00",
            ),
        )
        conn.execute(
            "INSERT INTO message_chunk_vectors (chunk_id, index_level, msg_id, account_email, thread_id, labels_json, chunk_index, chunk_type, chunk_start, chunk_end, chunk_text, chunk_text_redacted, embedding_json, embedding_dim, embedding_model, content_hash, redaction_policy_version, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "c1",
                "redacted",
                "m-new",
                "acct@example.com",
                "t1",
                json.dumps(["INBOX"]),
                0,
                "body",
                0,
                10,
                "chunk",
                "chunk",
                json.dumps([0.1, 0.2]),
                2,
                "test-model",
                "chunk-hash-1",
                "2026-03-22-precision-1",
                "2026-03-10T10:02:00+00:00",
            ),
        )
        conn.execute(
            "INSERT INTO vector_index_state (msg_id, index_level, content_hash, redaction_policy_version, updated_at) VALUES (?, ?, ?, ?, ?)",
            (
                "m-new",
                "redacted",
                "hash-1",
                "2026-03-22-precision-1",
                "2026-03-10T10:02:00+00:00",
            ),
        )
        conn.execute(
            "INSERT INTO message_enrichment (msg_id, category, importance, action, summary, model, enriched_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("m-new", "work", 5, "respond", "summary", "test-model", "2026-03-10T10:03:00+00:00"),
        )
        conn.execute(
            "INSERT INTO contact_stats (contact_email, display_name, first_seen, last_seen, message_count) VALUES (?, ?, ?, ?, ?)",
            (
                "boss@example.com",
                "Boss",
                "2026-03-01T00:00:00+00:00",
                "2026-03-10T00:00:00+00:00",
                9,
            ),
        )
        conn.execute(
            "INSERT INTO contact_profiles (contact_email, profile_json, model, updated_at) VALUES (?, ?, ?, ?)",
            (
                "boss@example.com",
                json.dumps(
                    {
                        "role": "manager",
                        "org": "Example Corp",
                        "notes": "Approves quarterly budgets",
                    }
                ),
                "heuristic",
                "2026-03-10T10:04:00+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_status_command_reports_counts_and_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    cfg.write_text(
        cfg.read_text()
        + """

[llm]
enabled = true
endpoint = "http://llm.test:11434"

[embeddings]
endpoint = "http://embedding.test:11434"
""",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")
    _seed_retrieval_data(tmp_path)
    monkeypatch.setattr(cli, "_endpoint_reachable", lambda url: "embedding" not in url)

    db_path = tmp_path / "test.db"
    from inbox_vault.db import get_conn

    conn = get_conn(str(db_path), "pw")
    try:
        conn.execute(
            """
            INSERT INTO message_enrichment (msg_id, category, importance, action, summary, model, enriched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(msg_id) DO UPDATE SET
              category=excluded.category,
              importance=excluded.importance,
              action=excluded.action,
              summary=excluded.summary,
              model=excluded.model,
              enriched_at=excluded.enriched_at
            """,
            (
                "m-old",
                "billing",
                4,
                "review",
                "heuristic summary",
                "heuristic-fallback",
                "2026-04-06T20:00:00+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(
        cli,
        "count_pending_vector_updates",
        lambda *_args, **kwargs: 4 if kwargs.get("index_level") == "redacted" else 0,
    )

    cli.main(["--config", str(cfg), "status", "--json"])
    out = json.loads(capsys.readouterr().out)

    assert out["counts"] == {
        "messages": 2,
        "message_vectors": 1,
        "message_chunk_vectors": 1,
        "enrichments": 2,
        "profiles": 1,
        "active_redaction_entries": 0,
        "rejected_redaction_entries": 0,
    }
    assert out["redaction_policy_version"] == "2026-03-22-precision-2"
    assert out["endpoint_health"] == {
        "llm": {
            "enabled": True,
            "endpoint": "http://llm.test:11434",
            "reachable": True,
        },
        "embeddings": {
            "endpoint": "http://embedding.test:11434",
            "reachable": False,
        },
    }
    assert out["enrichment_status"] == {
        "pending": 0,
        "heuristic_fallback": 1,
        "repairable": 1,
        "degraded": True,
    }
    assert out["available_index_levels"] == ["redacted"]
    assert out["full_search_available"] is False
    assert out["vector_level_counts"] == {"redacted": {"messages": 1, "chunks": 1}}
    assert out["pending_vectors"] == {"redacted": 4, "full": None}
    assert out["policy_drift_vectors"] == {"redacted": 1}
    assert out["ingest_triage"] == {
        "enabled": False,
        "mode": "observe",
        "summary": {
            "messages": 0,
            "streams": 0,
            "proposed_tiers": {},
            "applied_tiers": {},
            "applied_light_messages": 0,
        },
    }
    assert out["action_needed"] is True
    assert out["latest_message"]["msg_id"] == "m-new"
    assert out["latest_message"]["freshness_seconds"] is not None


def test_latest_command_uses_safe_truncated_previews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")
    _seed_retrieval_data(tmp_path)

    cli.main(
        [
            "--config",
            str(cfg),
            "latest",
            "--limit",
            "1",
            "--max-subject-chars",
            "10",
            "--max-snippet-chars",
            "14",
            "--max-body-chars",
            "80",
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)

    assert out["count"] == 1
    assert out["messages"][0]["msg_id"] == "m-new"
    assert out["messages"][0]["subject"].endswith("...")
    assert out["messages"][0]["snippet"].endswith("...")
    assert out["messages"][0]["body_preview"].endswith("...")
    assert "alice@example.com" not in out["messages"][0]["body_preview"]
    assert "[REDACTED_EMAIL]" in out["messages"][0]["body_preview"]


def test_latest_supports_date_range_and_clearance_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")
    _seed_retrieval_data(tmp_path)

    cli.main(
        [
            "--config",
            str(cfg),
            "latest",
            "--limit",
            "5",
            "--from-date",
            "2026-03-05",
            "--to-date",
            "2026-03-10",
            "--clearance",
            "full",
            "--max-body-chars",
            "200",
            "--json",
        ]
    )
    full_out = json.loads(capsys.readouterr().out)
    assert full_out["count"] == 1
    assert full_out["messages"][0]["msg_id"] == "m-new"
    assert "alice@example.com" in full_out["messages"][0]["body_preview"]

    cli.main(
        [
            "--config",
            str(cfg),
            "latest",
            "--limit",
            "5",
            "--from-date",
            "2026-03-05",
            "--to-date",
            "2026-03-10",
            "--clearance",
            "redacted",
            "--max-body-chars",
            "200",
            "--json",
        ]
    )
    redacted_out = json.loads(capsys.readouterr().out)
    assert redacted_out["count"] == 1
    assert redacted_out["messages"][0]["msg_id"] == "m-new"
    assert "alice@example.com" not in redacted_out["messages"][0]["body_preview"]
    assert "[REDACTED_EMAIL]" in redacted_out["messages"][0]["body_preview"]


def test_message_command_fetches_by_msg_id_with_clearance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")
    _seed_retrieval_data(tmp_path)

    cli.main(
        [
            "--config",
            str(cfg),
            "message",
            "m-new",
            "--clearance",
            "full",
            "--max-body-chars",
            "200",
            "--json",
        ]
    )
    full_out = json.loads(capsys.readouterr().out)
    assert full_out["found"] is True
    assert full_out["msg_id"] == "m-new"
    assert full_out["message"]["msg_id"] == "m-new"
    assert "alice@example.com" in full_out["message"]["body_preview"]

    cli.main(
        [
            "--config",
            str(cfg),
            "message",
            "m-new",
            "--clearance",
            "redacted",
            "--max-body-chars",
            "200",
            "--json",
        ]
    )
    redacted_out = json.loads(capsys.readouterr().out)
    assert redacted_out["found"] is True
    assert redacted_out["message"]["msg_id"] == "m-new"
    assert "alice@example.com" not in redacted_out["message"]["body_preview"]
    assert "[REDACTED_EMAIL]" in redacted_out["message"]["body_preview"]


def test_message_command_returns_not_found_for_unknown_msg_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")
    _seed_retrieval_data(tmp_path)

    cli.main(["--config", str(cfg), "message", "does-not-exist", "--json"])
    out = json.loads(capsys.readouterr().out)

    assert out == {
        "found": False,
        "msg_id": "does-not-exist",
        "count": 0,
        "message": None,
    }


def test_search_includes_sender_fields_with_clearance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")
    _seed_retrieval_data(tmp_path)

    def fake_search_vectors(*_args, **_kwargs):
        return (
            [
                type(
                    "R",
                    (),
                    {
                        "score": 0.99,
                        "msg_id": "m-new",
                        "thread_id": "t1",
                        "account_email": "acct@example.com",
                        "labels": ["INBOX"],
                        "content": "Subject: Project updates",
                    },
                )(),
            ],
            _search_diag(requested="auto", used="redacted", fallback="full", full_available=False),
        )

    monkeypatch.setattr(cli, "search_vectors", fake_search_vectors)

    cli.main(["--config", str(cfg), "search", "project", "--clearance", "full"])
    full_out = json.loads(capsys.readouterr().out)
    assert full_out["results"][0]["from_addr"] == "boss@example.com"
    assert full_out["results"][0]["to_addr"] == "acct@example.com"

    cli.main(["--config", str(cfg), "search", "project", "--clearance", "redacted"])
    redacted_out = json.loads(capsys.readouterr().out)
    assert "boss@example.com" not in redacted_out["results"][0]["from_addr"]
    assert "acct@example.com" not in redacted_out["results"][0]["to_addr"]
    assert "[REDACTED_EMAIL]" in redacted_out["results"][0]["from_addr"]
    assert "[REDACTED_EMAIL]" in redacted_out["results"][0]["to_addr"]


def test_help_includes_date_gated_and_msg_id_examples(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])

    help_text = capsys.readouterr().out
    assert "latest --from-date 2026-03-01 --to-date 2026-03-08 --clearance redacted" in help_text
    assert "search \"invoice follow-up\" --from-date 2026-03-01 --to-date 2026-03-08" in help_text
    assert "message 190b5f6b1f8a7c2d --clearance full" in help_text
    assert "Fetch one message by msg_id with safe preview output" in help_text


def test_profile_search_command_matches_email_and_profile_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    cfg = tmp_path / "config.toml"
    _write_config(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DB_PASSWORD", "pw")
    _seed_retrieval_data(tmp_path)

    cli.main(["--config", str(cfg), "profile-search", "budget", "--json"])
    out = json.loads(capsys.readouterr().out)

    assert out["count"] == 1
    assert out["results"][0]["contact_email"] == "boss@example.com"
    assert out["results"][0]["has_profile"] is True
    assert "quarterly budgets" in out["results"][0]["profile_preview"].lower()
