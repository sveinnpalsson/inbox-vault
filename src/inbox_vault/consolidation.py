from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from .config import AccountConfig, AppConfig
from .db import get_conn
from .enrich import enrich_pending
from .ingest import backfill
from .profiles import build_profiles
from .vectors import index_vectors


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str | None, max_chars: int = 100) -> str:
    value = (text or "").replace("\n", " ").strip()
    if len(value) <= max_chars:
        return value
    return value[: max(1, max_chars - 1)].rstrip() + "…"


@contextmanager
def _single_writer_guard(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        lock_owner = "unknown"
        try:
            lock_owner = lock_path.read_text().strip() or "unknown"
        except Exception:
            pass
        raise RuntimeError(
            f"Consolidation lock already held: {lock_path} (owner={lock_owner}). "
            "Avoid concurrent index writers for this target; wait for current run to finish."
        ) from exc

    try:
        payload = {"pid": os.getpid(), "created_at": _utc_now_iso()}
        os.write(fd, json.dumps(payload).encode("utf-8"))
        os.close(fd)
        yield
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass


def _clone_target_cfg(base_cfg: AppConfig, target_dir: Path, *, copy_tokens: bool) -> AppConfig:
    target_dir.mkdir(parents=True, exist_ok=True)

    oauth_dir = target_dir / "oauth"
    oauth_dir.mkdir(parents=True, exist_ok=True)

    cloned_accounts: list[AccountConfig] = []
    for account in base_cfg.accounts:
        token_dst = oauth_dir / f"{account.email.replace('@', '_at_')}.token.json"
        src_token = Path(account.token_file)
        if copy_tokens and src_token.exists() and not token_dst.exists():
            token_dst.write_text(src_token.read_text())

        cloned_accounts.append(replace(account, token_file=str(token_dst)))

    return replace(
        base_cfg,
        accounts=cloned_accounts,
        db=replace(base_cfg.db, path=str(target_dir / "inbox_vault.db")),
    )


def _safe_step(name: str, fn):
    started = perf_counter()
    try:
        out = fn()
        return {
            "name": name,
            "status": "ok",
            "duration_ms": int((perf_counter() - started) * 1000),
            "metrics": out if isinstance(out, dict) else {"value": out},
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "failed",
            "duration_ms": int((perf_counter() - started) * 1000),
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def _build_inspect_summary(conn, *, sample_limit: int = 10) -> dict:
    counts = {
        "messages": int(conn.execute("SELECT count(*) FROM messages").fetchone()[0]),
        "raw_messages": int(conn.execute("SELECT count(*) FROM raw_messages").fetchone()[0]),
        "enrichment_rows": int(
            conn.execute("SELECT count(*) FROM message_enrichment").fetchone()[0]
        ),
        "profiles": int(conn.execute("SELECT count(*) FROM contact_profiles").fetchone()[0]),
        "message_vectors_v2": int(
            conn.execute("SELECT count(*) FROM message_vectors_v2").fetchone()[0]
        ),
        "chunk_vectors_v2": int(
            conn.execute("SELECT count(*) FROM message_chunk_vectors_v2").fetchone()[0]
        ),
    }

    rows = conn.execute(
        """
        SELECT msg_id, account_email, thread_id, date_iso, labels_json, subject
        FROM messages
        ORDER BY COALESCE(internal_ts, 0) DESC
        LIMIT ?
        """,
        (max(1, int(sample_limit)),),
    ).fetchall()

    samples = [
        {
            "msg_id": row[0],
            "account_email": row[1],
            "thread_id": row[2],
            "date_iso": row[3],
            "labels_json": row[4],
            "subject_preview": _truncate(row[5], 100),
        }
        for row in rows
    ]

    return {
        "generated_at": _utc_now_iso(),
        "counts": counts,
        "recent_message_samples": samples,
    }


def run_consolidation(
    base_cfg: AppConfig,
    db_password: str,
    *,
    target_root: str,
    max_messages: int = 20,
    enrich_limit: int = 500,
    profiles_limit: int = 200,
    profiles_use_llm: bool = False,
    redaction_mode: str | None = None,
    redaction_profile: str | None = None,
    redaction_instruction: str | None = None,
    copy_tokens: bool = True,
    report_path: str | None = None,
) -> dict:
    target_dir = Path(target_root).expanduser().resolve()
    cfg = _clone_target_cfg(base_cfg, target_dir, copy_tokens=copy_tokens)

    artifacts = {
        "target_root": str(target_dir),
        "source_db_path": str(Path(base_cfg.db.path).expanduser().resolve()),
        "target_db_path": str(Path(cfg.db.path).resolve()),
    }

    lock_path = target_dir / ".index-writer.lock"
    conn = get_conn(cfg.db.path, db_password)
    steps: list[dict] = []
    try:
        try:
            with _single_writer_guard(lock_path):
                steps.append(
                    _safe_step("backfill", lambda: backfill(conn, cfg, max_messages=max_messages))
                )

                if all(step["status"] == "ok" for step in steps):

                    def _enrich() -> dict:
                        diagnostics: dict[str, int] = {}
                        updated = enrich_pending(
                            conn, cfg, limit=enrich_limit, diagnostics=diagnostics
                        )
                        return {"updated": updated, "diagnostics": diagnostics}

                    steps.append(_safe_step("enrich", _enrich))

                if all(step["status"] == "ok" for step in steps):

                    def _profiles() -> dict:
                        diagnostics: dict[str, int] = {}
                        updated = build_profiles(
                            conn,
                            cfg,
                            use_llm=profiles_use_llm,
                            limit=profiles_limit,
                            diagnostics=diagnostics,
                        )
                        return {"updated": updated, "diagnostics": diagnostics}

                    steps.append(_safe_step("build_profiles", _profiles))

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

                inspect_summary: dict[str, object] | None = None
                if all(step["status"] == "ok" for step in steps):
                    inspect_summary = _build_inspect_summary(conn)
                    summary_path = target_dir / "inspect-summary.json"
                    summary_path.write_text(json.dumps(inspect_summary, indent=2))
                    artifacts["inspect_summary_path"] = str(summary_path)
                    steps.append(
                        {
                            "name": "inspect_summary_report",
                            "status": "ok",
                            "duration_ms": 0,
                            "metrics": {
                                "summary_path": str(summary_path),
                                "counts": inspect_summary["counts"],
                            },
                        }
                    )
                else:
                    steps.append(
                        {
                            "name": "inspect_summary_report",
                            "status": "skipped",
                            "reason": "prior_step_failed",
                        }
                    )
        except Exception as exc:
            steps.append(
                {
                    "name": "single_writer_guard",
                    "status": "failed",
                    "duration_ms": 0,
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
    finally:
        conn.close()

    ok = all(step["status"] in {"ok", "skipped"} for step in steps)
    report = {
        "ok": ok,
        "pipeline": "backfill->enrich->build-profiles->index-vectors->inspect",
        "max_messages": max(1, int(max_messages)),
        "redaction_mode": redaction_mode or cfg.redaction.mode,
        "artifacts": artifacts,
        "steps": steps,
    }

    output_path = (
        Path(report_path).expanduser().resolve()
        if report_path
        else (target_dir / "consolidation-report.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    report["report_path"] = str(output_path)
    return report
