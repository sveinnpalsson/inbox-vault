from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from .config import AccountConfig, AppConfig
from .db import get_conn
from .enrich import enrich_pending
from .evals import run_retrieval_eval
from .ingest import backfill
from .profiles import build_profiles
from .vectors import index_vectors, search_vectors


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_validate(conn) -> dict:
    checks = {
        "messages_table": conn.execute(
            "SELECT name FROM sqlite_master WHERE name='messages'"
        ).fetchone()
        is not None,
        "raw_messages_table": conn.execute(
            "SELECT name FROM sqlite_master WHERE name='raw_messages'"
        ).fetchone()
        is not None,
        "sync_cursors_table": conn.execute(
            "SELECT name FROM sqlite_master WHERE name='sync_cursors'"
        ).fetchone()
        is not None,
        "message_vectors_table": conn.execute(
            "SELECT name FROM sqlite_master WHERE name='message_vectors'"
        ).fetchone()
        is not None,
        "message_chunk_vectors_table": conn.execute(
            "SELECT name FROM sqlite_master WHERE name='message_chunk_vectors'"
        ).fetchone()
        is not None,
        "message_fts_table": conn.execute(
            "SELECT name FROM sqlite_master WHERE name='message_fts'"
        ).fetchone()
        is not None,
    }
    checks["ok"] = all(checks.values())
    return checks


def _safe_step(name: str, fn):
    started = perf_counter()
    try:
        out = fn()
        duration_ms = int((perf_counter() - started) * 1000)
        return {
            "name": name,
            "status": "ok",
            "duration_ms": duration_ms,
            "metrics": out if isinstance(out, dict) else {"value": out},
        }
    except Exception as exc:
        duration_ms = int((perf_counter() - started) * 1000)
        return {
            "name": name,
            "status": "failed",
            "duration_ms": duration_ms,
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def _clone_isolated_cfg(base_cfg: AppConfig, isolated_dir: Path, *, copy_tokens: bool) -> AppConfig:
    isolated_dir.mkdir(parents=True, exist_ok=True)

    cloned_accounts: list[AccountConfig] = []
    oauth_dir = isolated_dir / "oauth"
    oauth_dir.mkdir(parents=True, exist_ok=True)

    for account in base_cfg.accounts:
        token_dst = oauth_dir / f"{account.email.replace('@', '_at_')}.token.json"
        src_token = Path(account.token_file)
        if copy_tokens and src_token.exists() and not token_dst.exists():
            token_dst.write_text(src_token.read_text())

        cloned_accounts.append(
            replace(
                account,
                token_file=str(token_dst),
            )
        )

    retrieval_cfg = replace(
        base_cfg.retrieval,
        lancedb_path=str(isolated_dir / "lancedb"),
    )

    return replace(
        base_cfg,
        accounts=cloned_accounts,
        db=replace(base_cfg.db, path=str(isolated_dir / "inbox_vault.db")),
        retrieval=retrieval_cfg,
    )


def _db_counts(conn) -> dict[str, int]:
    return {
        "messages": int(conn.execute("SELECT count(*) FROM messages").fetchone()[0]),
        "raw_messages": int(conn.execute("SELECT count(*) FROM raw_messages").fetchone()[0]),
        "enrichment_rows": int(
            conn.execute("SELECT count(*) FROM message_enrichment").fetchone()[0]
        ),
        "profiles": int(conn.execute("SELECT count(*) FROM contact_profiles").fetchone()[0]),
        "message_vectors": int(conn.execute("SELECT count(*) FROM message_vectors").fetchone()[0]),
        "chunk_vectors": int(
            conn.execute("SELECT count(*) FROM message_chunk_vectors").fetchone()[0]
        ),
    }


def run_isolated_stress(
    base_cfg: AppConfig,
    db_password: str,
    *,
    isolated_root: str,
    max_messages: int = 20,
    enrich_limit: int = 500,
    profiles_limit: int = 200,
    profiles_use_llm: bool = False,
    redaction_mode: str | None = None,
    redaction_profile: str | None = None,
    redaction_instruction: str | None = None,
    search_query: str = "project",
    search_top_k: int = 5,
    search_account_email: str | None = None,
    search_label: str | None = None,
    strategy: str = "hybrid",
    eval_file: str | None = None,
    copy_tokens: bool = True,
    report_path: str | None = None,
) -> dict:
    root = Path(isolated_root).expanduser().resolve()
    run_dir = root / _utc_stamp()
    cfg = _clone_isolated_cfg(base_cfg, run_dir, copy_tokens=copy_tokens)

    conn = get_conn(cfg.db.path, db_password)
    steps: list[dict] = []
    try:
        steps.append(_safe_step("validate", lambda: _run_validate(conn)))
        if steps[-1]["status"] == "ok" and not steps[-1]["metrics"].get("ok", False):
            steps[-1]["status"] = "failed"
            steps[-1]["error"] = "validate returned ok=false"

        if all(step["status"] == "ok" for step in steps):
            steps.append(
                _safe_step(
                    "backfill",
                    lambda: backfill(conn, cfg, max_messages=max_messages),
                )
            )

        if all(step["status"] == "ok" for step in steps):

            def _enrich_step() -> dict:
                diagnostics: dict[str, int] = {}
                updated = enrich_pending(conn, cfg, limit=enrich_limit, diagnostics=diagnostics)
                return {"updated": updated, "diagnostics": diagnostics}

            steps.append(_safe_step("enrich", _enrich_step))

        if all(step["status"] == "ok" for step in steps):

            def _profiles_step() -> dict:
                diagnostics: dict[str, int] = {}
                updated = build_profiles(
                    conn,
                    cfg,
                    use_llm=profiles_use_llm,
                    limit=profiles_limit,
                    diagnostics=diagnostics,
                )
                return {"updated": updated, "diagnostics": diagnostics}

            steps.append(_safe_step("build_profiles", _profiles_step))

        if all(step["status"] == "ok" for step in steps):
            steps.append(
                _safe_step(
                    "index_vectors",
                    lambda: index_vectors(
                        conn,
                        cfg,
                        force=True,
                        redaction_mode=redaction_mode,
                        redaction_profile=redaction_profile,
                        redaction_instruction=redaction_instruction,
                    ),
                )
            )

        if all(step["status"] == "ok" for step in steps):

            def _search_check() -> dict:
                results = search_vectors(
                    conn,
                    cfg,
                    search_query,
                    account_email=search_account_email,
                    label=search_label,
                    top_k=search_top_k,
                    clearance="redacted",
                    strategy=strategy,
                )
                return {
                    "query": search_query,
                    "top_k": search_top_k,
                    "result_count": len(results),
                    "top_msg_ids": [item.msg_id for item in results[: min(5, len(results))]],
                }

            steps.append(_safe_step("semantic_search_check", _search_check))

        if eval_file:
            if all(step["status"] == "ok" for step in steps):
                steps.append(
                    _safe_step(
                        "eval_retrieval",
                        lambda: run_retrieval_eval(
                            conn,
                            cfg,
                            eval_file=eval_file,
                            strategy=strategy,
                            top_k=max(search_top_k, 1),
                            clearance="redacted",
                        ),
                    )
                )
            else:
                steps.append(
                    {"name": "eval_retrieval", "status": "skipped", "reason": "prior_step_failed"}
                )
        else:
            steps.append({"name": "eval_retrieval", "status": "skipped", "reason": "no_eval_file"})

        ok = all(step["status"] in {"ok", "skipped"} for step in steps)
        report = {
            "ok": ok,
            "isolated_run_dir": str(run_dir),
            "db_path": str(Path(cfg.db.path).resolve()),
            "lancedb_path": str(Path(cfg.retrieval.lancedb_path).resolve()),
            "redaction_mode": redaction_mode or cfg.redaction.mode,
            "strategy": strategy,
            "steps": steps,
            "counts": _db_counts(conn),
        }

        output_path = (
            Path(report_path).expanduser().resolve()
            if report_path
            else (run_dir / "stress-report.json")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        report["report_path"] = str(output_path)
        return report
    finally:
        conn.close()
