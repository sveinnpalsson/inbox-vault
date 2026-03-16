from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import AppConfig
from .db import (
    DBLockRetryExhausted,
    delete_message_chunk_vectors,
    fetch_chunk_vectors_for_search,
    fetch_messages_by_ids,
    fetch_redaction_entries,
    fetch_vectors_for_search,
    get_vector_row,
    lexical_search_rows,
    upsert_message_chunk_vector,
    upsert_message_vector,
    upsert_redaction_entries,
    vector_index_source_rows,
)
from .llm import embedding_vector
from .redaction import PersistentRedactionMap, redact_text, redact_with_persistent_map

LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class SearchResult:
    score: float
    msg_id: str
    thread_id: str | None
    account_email: str
    labels: list[str]
    content: str


@dataclass(slots=True)
class _Candidate:
    msg_id: str
    thread_id: str | None
    account_email: str
    labels: list[str]
    source_text: str
    source_text_redacted: str
    score: float
    chunk_hits: int = 1
    first_chunk_rank: int = 1


@dataclass(slots=True)
class _Chunk:
    chunk_id: str
    chunk_index: int
    chunk_type: str
    chunk_start: int
    chunk_end: int
    text: str


@dataclass(slots=True)
class _ChunkCandidate:
    chunk_id: str
    msg_id: str
    thread_id: str | None
    account_email: str
    labels: list[str]
    chunk_index: int
    chunk_text: str
    chunk_text_redacted: str
    score: float


def _compose_source_text(subject: str | None, snippet: str | None, body: str | None) -> str:
    return "\n".join(
        [
            f"Subject: {(subject or '').strip()}",
            f"Snippet: {(snippet or '').strip()}",
            f"Body: {(body or '').strip()}",
        ]
    ).strip()


_ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\u2060\uFEFF]")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_indexing(
    text: str | None,
    *,
    strip_zero_width: bool,
    collapse_whitespace: bool,
    max_chars: int,
) -> str:
    out = text or ""
    if strip_zero_width:
        out = _ZERO_WIDTH_RE.sub("", out)
    if collapse_whitespace:
        out = _WHITESPACE_RE.sub(" ", out)
    out = out.strip()
    if max_chars > 0 and len(out) > max_chars:
        out = out[:max_chars].rstrip()
    return out


def _normalized_label_set(labels: list[str]) -> set[str]:
    return {str(item).strip().upper() for item in labels if str(item).strip()}


def _should_filter_message(
    labels: list[str],
    *,
    include_labels: set[str],
    exclude_labels: set[str],
) -> bool:
    normalized = _normalized_label_set(labels)
    if include_labels and not (normalized & include_labels):
        return True
    if exclude_labels and (normalized & exclude_labels):
        return True
    return False


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_id(msg_id: str, chunk_index: int) -> str:
    return f"{msg_id}::c{chunk_index:05d}"


def _chunk_text(body: str, *, chunk_chars: int, overlap_chars: int) -> list[tuple[int, int, str]]:
    text = body.strip()
    if not text:
        return []

    safe_size = max(200, int(chunk_chars))
    safe_overlap = max(0, min(int(overlap_chars), safe_size - 1))
    step = max(1, safe_size - safe_overlap)

    out: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + safe_size)
        chunk = text[start:end].strip()
        if chunk:
            out.append((start, end, chunk))
        if end >= len(text):
            break
        start += step
    return out


def _build_chunks(
    msg_id: str,
    *,
    subject: str | None,
    body_text: str | None,
    chunk_chars: int,
    overlap_chars: int,
) -> list[_Chunk]:
    chunks: list[_Chunk] = []
    next_index = 0

    subj = (subject or "").strip()
    if subj:
        chunks.append(
            _Chunk(
                chunk_id=_chunk_id(msg_id, next_index),
                chunk_index=next_index,
                chunk_type="subject",
                chunk_start=0,
                chunk_end=len(subj),
                text=f"Subject: {subj}",
            )
        )
        next_index += 1

    body = (body_text or "").strip()
    for start, end, chunk in _chunk_text(
        body,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
    ):
        chunks.append(
            _Chunk(
                chunk_id=_chunk_id(msg_id, next_index),
                chunk_index=next_index,
                chunk_type="body",
                chunk_start=start,
                chunk_end=end,
                text=f"Body: {chunk}",
            )
        )
        next_index += 1

    if not chunks:
        chunks.append(
            _Chunk(
                chunk_id=_chunk_id(msg_id, 0),
                chunk_index=0,
                chunk_type="body",
                chunk_start=0,
                chunk_end=0,
                text="Body:",
            )
        )

    return chunks


def _lancedb_available() -> tuple[bool, str]:
    try:
        import lancedb  # noqa: F401

        return True, "enabled"
    except Exception as exc:
        return False, f"unavailable:{exc.__class__.__name__}"


def _upsert_lancedb_chunk_record(cfg: AppConfig, record: dict[str, Any]) -> bool:
    available, _status = _lancedb_available()
    if not available:
        return False

    import lancedb

    db_path = Path(cfg.retrieval.lancedb_path)
    db_path.mkdir(parents=True, exist_ok=True)

    db = lancedb.connect(str(db_path))
    table_name = cfg.retrieval.lancedb_table

    try:
        table = db.open_table(table_name)
    except Exception:
        db.create_table(table_name, data=[record], mode="overwrite")
        return True

    try:
        escaped_chunk_id = str(record["chunk_id"]).replace("'", "''")
        table.delete(f"chunk_id = '{escaped_chunk_id}'")
    except Exception:
        LOG.debug("Unable to delete existing LanceDB row for chunk_id=%s", record["chunk_id"])
    table.add([record])
    return True


def _search_lancedb_chunks(
    cfg: AppConfig,
    query_embedding: list[float],
    *,
    account_email: str | None,
    label: str | None,
    limit: int,
) -> list[_ChunkCandidate]:
    available, _status = _lancedb_available()
    if not available:
        return []

    import lancedb

    db = lancedb.connect(cfg.retrieval.lancedb_path)
    try:
        table = db.open_table(cfg.retrieval.lancedb_table)
    except Exception:
        return []

    try:
        rows = table.search(query_embedding).limit(max(1, int(limit))).to_list()
    except Exception:
        LOG.exception("LanceDB search failed; falling back to sqlite vector scan")
        return []

    wanted_label = (label or "").strip().upper()
    out: list[_ChunkCandidate] = []
    for row in rows:
        msg_id = str(row.get("msg_id", ""))
        if not msg_id:
            continue

        acct = str(row.get("account_email", ""))
        if account_email and acct != account_email:
            continue

        labels = [str(v).upper() for v in row.get("labels", [])]
        if wanted_label and wanted_label not in labels:
            continue

        distance = row.get("_distance")
        try:
            score = 1.0 - float(distance)
        except (TypeError, ValueError):
            score = 0.0

        chunk_id = str(row.get("chunk_id") or f"{msg_id}::legacy")
        chunk_index = int(row.get("chunk_index") or 0)
        out.append(
            _ChunkCandidate(
                chunk_id=chunk_id,
                msg_id=msg_id,
                thread_id=str(row.get("thread_id")) if row.get("thread_id") else None,
                account_email=acct,
                labels=labels,
                chunk_index=chunk_index,
                chunk_text=str(row.get("chunk_text", "")),
                chunk_text_redacted=str(row.get("chunk_text_redacted", "")),
                score=score,
            )
        )

    out.sort(key=lambda r: (-r.score, r.msg_id, r.chunk_index, r.chunk_id))
    return out


def _pending_message_ids_for_index(
    conn,
    cfg: AppConfig,
    *,
    account_email: str | None = None,
    include_labels: list[str] | None = None,
    exclude_labels: list[str] | None = None,
    max_index_chars: int | None = None,
    limit: int | None = None,
) -> list[str]:
    rows = vector_index_source_rows(conn, account_email=account_email, limit=None)
    effective_include_labels = _normalized_label_set(
        include_labels if include_labels is not None else cfg.indexing.include_labels
    )
    effective_exclude_labels = _normalized_label_set(
        exclude_labels if exclude_labels is not None else cfg.indexing.exclude_labels
    )
    effective_max_chars = (
        max(1, int(max_index_chars))
        if max_index_chars is not None
        else cfg.indexing.max_index_chars
    )
    safe_limit = max(1, int(limit)) if limit is not None else None

    pending_msg_ids: list[str] = []
    for msg_id, _acct, _thread_id, subject, snippet, body_text, labels_json in rows:
        labels = json.loads(labels_json or "[]")
        if _should_filter_message(
            labels,
            include_labels=effective_include_labels,
            exclude_labels=effective_exclude_labels,
        ):
            continue

        normalized_subject = _normalize_for_indexing(
            subject,
            strip_zero_width=cfg.indexing.strip_zero_width,
            collapse_whitespace=cfg.indexing.collapse_whitespace,
            max_chars=effective_max_chars,
        )
        normalized_snippet = _normalize_for_indexing(
            snippet,
            strip_zero_width=cfg.indexing.strip_zero_width,
            collapse_whitespace=cfg.indexing.collapse_whitespace,
            max_chars=effective_max_chars,
        )
        normalized_body = _normalize_for_indexing(
            body_text,
            strip_zero_width=cfg.indexing.strip_zero_width,
            collapse_whitespace=cfg.indexing.collapse_whitespace,
            max_chars=effective_max_chars,
        )
        source = _compose_source_text(normalized_subject, normalized_snippet, normalized_body)
        prior = get_vector_row(conn, msg_id)
        if not prior or prior[0] != _content_hash(source):
            pending_msg_ids.append(msg_id)
            if safe_limit is not None and len(pending_msg_ids) >= safe_limit:
                break

    return pending_msg_ids


def count_pending_vector_updates(
    conn,
    cfg: AppConfig,
    *,
    account_email: str | None = None,
    include_labels: list[str] | None = None,
    exclude_labels: list[str] | None = None,
    max_index_chars: int | None = None,
) -> int:
    return len(
        _pending_message_ids_for_index(
            conn,
            cfg,
            account_email=account_email,
            include_labels=include_labels,
            exclude_labels=exclude_labels,
            max_index_chars=max_index_chars,
        )
    )


def index_vectors(
    conn,
    cfg: AppConfig,
    *,
    account_email: str | None = None,
    limit: int | None = None,
    force: bool = False,
    pending_only: bool = False,
    redaction_mode: str | None = None,
    redaction_profile: str | None = None,
    redaction_instruction: str | None = None,
    include_labels: list[str] | None = None,
    exclude_labels: list[str] | None = None,
    max_index_chars: int | None = None,
    lock_max_retries: int = 6,
    lock_backoff_base_seconds: float = 0.05,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    commit_every_messages: int = 1,
) -> dict[str, int | str]:
    lancedb_enabled = False
    lancedb_status = "disabled"
    if cfg.retrieval.vector_backend == "lancedb":
        lancedb_enabled, lancedb_status = _lancedb_available()
        if not lancedb_enabled:
            LOG.warning(
                "LanceDB backend requested but unavailable (%s); using sqlite-only vectors",
                lancedb_status,
            )

    stats: dict[str, int | str] = {
        "scanned": 0,
        "indexed": 0,
        "unchanged": 0,
        "skipped_filtered": 0,
        "failed": 0,
        "chunks_indexed": 0,
        "lancedb_indexed": 0,
        "lancedb_failed": 0,
        "lancedb_status": lancedb_status,
        "lock_retries": 0,
        "lock_errors": 0,
        "redaction_entries_added": 0,
    }
    if pending_only and not force:
        pending_ids = _pending_message_ids_for_index(
            conn,
            cfg,
            account_email=account_email,
            include_labels=include_labels,
            exclude_labels=exclude_labels,
            max_index_chars=max_index_chars,
            limit=limit,
        )
        pending_id_set = set(pending_ids)
        source_rows = vector_index_source_rows(conn, account_email=account_email, limit=None)
        rows = [row for row in source_rows if row[0] in pending_id_set]
    else:
        rows = vector_index_source_rows(conn, account_email=account_email, limit=limit)
    total_rows = len(rows)
    safe_commit_every_messages = max(1, int(commit_every_messages))
    indexed_since_last_commit = 0

    mode = (redaction_mode or cfg.redaction.mode).strip().lower()
    profile = (
        redaction_profile if redaction_profile is not None else cfg.redaction.profile
    ).strip()
    instruction = (
        redaction_instruction if redaction_instruction is not None else cfg.redaction.instruction
    ).strip()

    effective_include_labels = _normalized_label_set(
        include_labels if include_labels is not None else cfg.indexing.include_labels
    )
    effective_exclude_labels = _normalized_label_set(
        exclude_labels if exclude_labels is not None else cfg.indexing.exclude_labels
    )
    effective_max_chars = (
        max(1, int(max_index_chars))
        if max_index_chars is not None
        else cfg.indexing.max_index_chars
    )

    redaction_maps_by_account: dict[str, PersistentRedactionMap] = {}

    for row_idx, (msg_id, acct, thread_id, subject, snippet, body_text, labels_json) in enumerate(
        rows, start=1
    ):
        message_started = time.perf_counter()
        stats["scanned"] += 1
        labels = json.loads(labels_json or "[]")
        if progress_callback:
            progress_callback(
                {
                    "event": "message_start",
                    "position": row_idx,
                    "total": total_rows,
                    "msg_id": msg_id,
                    "account_email": acct,
                    "scanned": int(stats["scanned"]),
                    "indexed": int(stats["indexed"]),
                    "chunks_indexed": int(stats["chunks_indexed"]),
                    "failed": int(stats["failed"]),
                    "substep": "normalization",
                }
            )
        if _should_filter_message(
            labels,
            include_labels=effective_include_labels,
            exclude_labels=effective_exclude_labels,
        ):
            stats["skipped_filtered"] += 1
            if progress_callback:
                progress_callback(
                    {
                        "event": "message_skipped_filtered",
                        "position": row_idx,
                        "total": total_rows,
                        "msg_id": msg_id,
                        "account_email": acct,
                        "elapsed_ms": int((time.perf_counter() - message_started) * 1000),
                        "indexed": int(stats["indexed"]),
                        "chunks_indexed": int(stats["chunks_indexed"]),
                        "failed": int(stats["failed"]),
                    }
                )
            continue

        normalized_subject = _normalize_for_indexing(
            subject,
            strip_zero_width=cfg.indexing.strip_zero_width,
            collapse_whitespace=cfg.indexing.collapse_whitespace,
            max_chars=effective_max_chars,
        )
        normalized_snippet = _normalize_for_indexing(
            snippet,
            strip_zero_width=cfg.indexing.strip_zero_width,
            collapse_whitespace=cfg.indexing.collapse_whitespace,
            max_chars=effective_max_chars,
        )
        normalized_body = _normalize_for_indexing(
            body_text,
            strip_zero_width=cfg.indexing.strip_zero_width,
            collapse_whitespace=cfg.indexing.collapse_whitespace,
            max_chars=effective_max_chars,
        )
        source_text = _compose_source_text(normalized_subject, normalized_snippet, normalized_body)
        fingerprint = _content_hash(source_text)
        prior = get_vector_row(conn, msg_id)
        if prior and prior[0] == fingerprint and not force:
            stats["unchanged"] += 1
            if progress_callback:
                progress_callback(
                    {
                        "event": "message_unchanged",
                        "position": row_idx,
                        "total": total_rows,
                        "msg_id": msg_id,
                        "account_email": acct,
                        "elapsed_ms": int((time.perf_counter() - message_started) * 1000),
                        "indexed": int(stats["indexed"]),
                        "chunks_indexed": int(stats["chunks_indexed"]),
                        "failed": int(stats["failed"]),
                    }
                )
            continue

        chunks = _build_chunks(
            msg_id,
            subject=normalized_subject,
            body_text=normalized_body,
            chunk_chars=cfg.retrieval.chunk_chars,
            overlap_chars=cfg.retrieval.chunk_overlap_chars,
        )
        if progress_callback:
            progress_callback(
                {
                    "event": "message_chunks_ready",
                    "position": row_idx,
                    "total": total_rows,
                    "msg_id": msg_id,
                    "account_email": acct,
                    "chunk_count": len(chunks),
                    "substep": "embed-message",
                }
            )

        redaction_map = redaction_maps_by_account.get(acct)
        if redaction_map is None:
            scope_rows = fetch_redaction_entries(conn, scope_type="account", scope_id=acct)
            redaction_map = PersistentRedactionMap.from_rows(scope_rows)
            redaction_maps_by_account[acct] = redaction_map

        try:
            message_embedding = embedding_vector(cfg.embeddings, source_text)
            if progress_callback:
                progress_callback(
                    {
                        "event": "message_substep",
                        "position": row_idx,
                        "total": total_rows,
                        "msg_id": msg_id,
                        "account_email": acct,
                        "substep": "redaction",
                        "chunk_count": len(chunks),
                    }
                )
            redaction_result = redact_with_persistent_map(
                source_text,
                chunks=[chunk.text for chunk in chunks],
                mode=mode,
                llm_cfg=cfg.llm,
                profile=profile,
                instruction=instruction,
                table=redaction_map,
            )
            source_text_redacted = redaction_result.source_text_redacted
        except Exception:
            LOG.exception("Vector/redaction generation failed for message_id=%s", msg_id)
            stats["failed"] += 1
            if progress_callback:
                progress_callback(
                    {
                        "event": "message_failed",
                        "position": row_idx,
                        "total": total_rows,
                        "msg_id": msg_id,
                        "account_email": acct,
                        "substep": "embed-message/redaction",
                        "elapsed_ms": int((time.perf_counter() - message_started) * 1000),
                        "indexed": int(stats["indexed"]),
                        "chunks_indexed": int(stats["chunks_indexed"]),
                        "failed": int(stats["failed"]),
                    }
                )
            continue

        if redaction_result.inserted_entries:
            try:
                stats["lock_retries"] += upsert_redaction_entries(
                    conn,
                    scope_type="account",
                    scope_id=acct,
                    entries=redaction_result.inserted_entries,
                    lock_max_retries=lock_max_retries,
                    lock_backoff_base_seconds=lock_backoff_base_seconds,
                )
                stats["redaction_entries_added"] += len(redaction_result.inserted_entries)
            except DBLockRetryExhausted as exc:
                LOG.error(
                    "Redaction table write lock retries exhausted for message_id=%s: %s",
                    msg_id,
                    exc,
                )
                stats["lock_errors"] += 1
                stats["failed"] += 1
                if progress_callback:
                    progress_callback(
                        {
                            "event": "message_failed",
                            "position": row_idx,
                            "total": total_rows,
                            "msg_id": msg_id,
                            "account_email": acct,
                            "substep": "upsert-redaction-entries",
                            "elapsed_ms": int((time.perf_counter() - message_started) * 1000),
                            "indexed": int(stats["indexed"]),
                            "chunks_indexed": int(stats["chunks_indexed"]),
                            "failed": int(stats["failed"]),
                        }
                    )
                continue

        try:
            stats["lock_retries"] += upsert_message_vector(
                conn,
                msg_id=msg_id,
                account_email=acct,
                thread_id=thread_id,
                labels=labels,
                source_text=source_text,
                source_text_redacted=source_text_redacted,
                embedding=message_embedding,
                embedding_model=cfg.embeddings.model,
                content_hash=fingerprint,
                lock_max_retries=lock_max_retries,
                lock_backoff_base_seconds=lock_backoff_base_seconds,
            )
        except DBLockRetryExhausted as exc:
            LOG.error("Vector write lock retries exhausted for message_id=%s: %s", msg_id, exc)
            stats["lock_errors"] += 1
            stats["failed"] += 1
            if progress_callback:
                progress_callback(
                    {
                        "event": "message_failed",
                        "position": row_idx,
                        "total": total_rows,
                        "msg_id": msg_id,
                        "account_email": acct,
                        "substep": "upsert-message-vector",
                        "elapsed_ms": int((time.perf_counter() - message_started) * 1000),
                        "indexed": int(stats["indexed"]),
                        "chunks_indexed": int(stats["chunks_indexed"]),
                        "failed": int(stats["failed"]),
                    }
                )
            continue
        try:
            stats["lock_retries"] += delete_message_chunk_vectors(
                conn,
                msg_id=msg_id,
                lock_max_retries=lock_max_retries,
                lock_backoff_base_seconds=lock_backoff_base_seconds,
            )
        except DBLockRetryExhausted as exc:
            LOG.error("Chunk delete lock retries exhausted for message_id=%s: %s", msg_id, exc)
            stats["lock_errors"] += 1
            stats["failed"] += 1
            if progress_callback:
                progress_callback(
                    {
                        "event": "message_failed",
                        "position": row_idx,
                        "total": total_rows,
                        "msg_id": msg_id,
                        "account_email": acct,
                        "substep": "delete-old-chunks",
                        "elapsed_ms": int((time.perf_counter() - message_started) * 1000),
                        "indexed": int(stats["indexed"]),
                        "chunks_indexed": int(stats["chunks_indexed"]),
                        "failed": int(stats["failed"]),
                    }
                )
            continue

        if progress_callback:
            progress_callback(
                {
                    "event": "message_substep",
                    "position": row_idx,
                    "total": total_rows,
                    "msg_id": msg_id,
                    "account_email": acct,
                    "substep": "embed-chunk/upsert",
                    "chunk_count": len(chunks),
                }
            )

        for chunk, chunk_redacted in zip(chunks, redaction_result.chunk_text_redacted):
            try:
                chunk_embedding = embedding_vector(cfg.embeddings, chunk.text)
            except Exception:
                LOG.exception(
                    "Chunk vector/redaction generation failed for message_id=%s chunk_id=%s",
                    msg_id,
                    chunk.chunk_id,
                )
                stats["failed"] += 1
                if progress_callback:
                    progress_callback(
                        {
                            "event": "message_failed",
                            "position": row_idx,
                            "total": total_rows,
                            "msg_id": msg_id,
                            "account_email": acct,
                            "substep": "embed-chunk",
                            "chunk_id": chunk.chunk_id,
                            "elapsed_ms": int((time.perf_counter() - message_started) * 1000),
                            "indexed": int(stats["indexed"]),
                            "chunks_indexed": int(stats["chunks_indexed"]),
                            "failed": int(stats["failed"]),
                        }
                    )
                continue

            chunk_fingerprint = _content_hash(
                f"{msg_id}|{chunk.chunk_type}|{chunk.chunk_start}|{chunk.chunk_end}|{chunk.text}"
            )
            try:
                stats["lock_retries"] += upsert_message_chunk_vector(
                    conn,
                    chunk_id=chunk.chunk_id,
                    msg_id=msg_id,
                    account_email=acct,
                    thread_id=thread_id,
                    labels=labels,
                    chunk_index=chunk.chunk_index,
                    chunk_type=chunk.chunk_type,
                    chunk_start=chunk.chunk_start,
                    chunk_end=chunk.chunk_end,
                    chunk_text=chunk.text,
                    chunk_text_redacted=chunk_redacted,
                    embedding=chunk_embedding,
                    embedding_model=cfg.embeddings.model,
                    content_hash=chunk_fingerprint,
                    lock_max_retries=lock_max_retries,
                    lock_backoff_base_seconds=lock_backoff_base_seconds,
                )
            except DBLockRetryExhausted as exc:
                LOG.error(
                    "Chunk write lock retries exhausted for message_id=%s chunk_id=%s: %s",
                    msg_id,
                    chunk.chunk_id,
                    exc,
                )
                stats["lock_errors"] += 1
                stats["failed"] += 1
                if progress_callback:
                    progress_callback(
                        {
                            "event": "message_failed",
                            "position": row_idx,
                            "total": total_rows,
                            "msg_id": msg_id,
                            "account_email": acct,
                            "substep": "upsert-chunk",
                            "chunk_id": chunk.chunk_id,
                            "elapsed_ms": int((time.perf_counter() - message_started) * 1000),
                            "indexed": int(stats["indexed"]),
                            "chunks_indexed": int(stats["chunks_indexed"]),
                            "failed": int(stats["failed"]),
                        }
                    )
                continue
            stats["chunks_indexed"] += 1

            if cfg.retrieval.vector_backend == "lancedb" and lancedb_enabled:
                lance_record = {
                    "chunk_id": chunk.chunk_id,
                    "msg_id": msg_id,
                    "account_email": acct,
                    "thread_id": thread_id,
                    "labels": labels,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "chunk_start": chunk.chunk_start,
                    "chunk_end": chunk.chunk_end,
                    "chunk_text": chunk.text,
                    "chunk_text_redacted": chunk_redacted,
                    "content_hash": chunk_fingerprint,
                    "embedding": chunk_embedding,
                }
                try:
                    if _upsert_lancedb_chunk_record(cfg, lance_record):
                        stats["lancedb_indexed"] += 1
                    else:
                        stats["lancedb_failed"] += 1
                except Exception:
                    LOG.exception(
                        "LanceDB upsert failed for message_id=%s chunk_id=%s",
                        msg_id,
                        chunk.chunk_id,
                    )
                    stats["lancedb_failed"] += 1

        stats["indexed"] += 1
        if progress_callback:
            elapsed_s = max(0.0001, time.perf_counter() - message_started)
            progress_callback(
                {
                    "event": "message_done",
                    "position": row_idx,
                    "total": total_rows,
                    "msg_id": msg_id,
                    "account_email": acct,
                    "chunk_count": len(chunks),
                    "elapsed_ms": int(elapsed_s * 1000),
                    "messages_per_min": round(60.0 / elapsed_s, 2),
                    "indexed": int(stats["indexed"]),
                    "chunks_indexed": int(stats["chunks_indexed"]),
                    "failed": int(stats["failed"]),
                }
            )

        indexed_since_last_commit += 1
        if indexed_since_last_commit >= safe_commit_every_messages:
            conn.commit()
            indexed_since_last_commit = 0

    conn.commit()
    return stats


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return -1.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)


def _dense_candidates_legacy_messages(
    conn,
    cfg: AppConfig,
    query: str,
    *,
    account_email: str | None,
    label: str | None,
    from_ts_ms: int | None,
    to_ts_ms: int | None,
) -> list[_Candidate]:
    qvec = embedding_vector(cfg.embeddings, query)

    rows = fetch_vectors_for_search(
        conn,
        account_email=account_email,
        label=label,
        from_ts_ms=from_ts_ms,
        to_ts_ms=to_ts_ms,
    )
    ranked: list[_Candidate] = []
    for (
        msg_id,
        acct,
        thread_id,
        labels_json,
        source_text,
        source_text_redacted,
        emb_json,
        _model,
    ) in rows:
        emb = [float(v) for v in json.loads(emb_json)]
        score = _cosine_similarity(qvec, emb)
        if score < 0:
            continue
        ranked.append(
            _Candidate(
                msg_id=msg_id,
                thread_id=thread_id,
                account_email=acct,
                labels=json.loads(labels_json or "[]"),
                source_text=source_text,
                source_text_redacted=source_text_redacted,
                score=score,
                chunk_hits=1,
                first_chunk_rank=1,
            )
        )
    ranked.sort(key=lambda r: (-r.score, r.msg_id))
    return ranked[: max(1, int(cfg.retrieval.dense_candidate_k))]


def _aggregate_chunk_candidates(
    conn,
    chunks: list[_ChunkCandidate],
    *,
    top_messages: int,
    from_ts_ms: int | None,
    to_ts_ms: int | None,
) -> list[_Candidate]:
    if not chunks:
        return []

    msg_rows = fetch_messages_by_ids(conn, [item.msg_id for item in chunks])
    buckets: dict[str, dict[str, Any]] = {}

    for rank, chunk in enumerate(chunks, start=1):
        bucket = buckets.get(chunk.msg_id)
        if bucket is None:
            bucket = {
                "best_score": chunk.score,
                "chunk_hits": 1,
                "first_rank": rank,
                "best_chunk_index": chunk.chunk_index,
            }
            buckets[chunk.msg_id] = bucket
        else:
            bucket["chunk_hits"] += 1
            better = chunk.score > bucket["best_score"]
            tied_and_earlier = (
                chunk.score == bucket["best_score"]
                and chunk.chunk_index < bucket["best_chunk_index"]
            )
            if better or tied_and_earlier:
                bucket["best_score"] = chunk.score
                bucket["best_chunk_index"] = chunk.chunk_index

    aggregated: list[_Candidate] = []
    for msg_id, bucket in buckets.items():
        row = msg_rows.get(msg_id)
        if not row:
            continue

        internal_ts = int(row[4]) if row[4] is not None else 0
        if from_ts_ms is not None and internal_ts < int(from_ts_ms):
            continue
        if to_ts_ms is not None and internal_ts >= int(to_ts_ms):
            continue

        labels = json.loads(row[3] or "[]")
        source_text = row[8] or _compose_source_text(row[5], row[6], row[7])
        source_text_redacted = row[9] or redact_text(source_text, mode="regex")
        aggregated.append(
            _Candidate(
                msg_id=msg_id,
                thread_id=row[2],
                account_email=row[1],
                labels=labels,
                source_text=source_text,
                source_text_redacted=source_text_redacted,
                score=float(bucket["best_score"]),
                chunk_hits=int(bucket["chunk_hits"]),
                first_chunk_rank=int(bucket["first_rank"]),
            )
        )

    # deterministic message-level aggregation:
    # 1) best chunk similarity desc
    # 2) number of matched chunks desc
    # 3) earliest winning chunk rank asc
    # 4) msg_id asc
    aggregated.sort(key=lambda r: (-r.score, -r.chunk_hits, r.first_chunk_rank, r.msg_id))
    return aggregated[: max(1, int(top_messages))]


def _dense_candidates(
    conn,
    cfg: AppConfig,
    query: str,
    *,
    account_email: str | None,
    label: str | None,
    from_ts_ms: int | None,
    to_ts_ms: int | None,
) -> list[_Candidate]:
    qvec = embedding_vector(cfg.embeddings, query)

    chunk_limit = max(1, int(cfg.retrieval.dense_candidate_k))

    if cfg.retrieval.vector_backend == "lancedb":
        available, status = _lancedb_available()
        if not available:
            LOG.info("LanceDB unavailable during search (%s); falling back to sqlite", status)
        else:
            chunk_rows = _search_lancedb_chunks(
                cfg,
                qvec,
                account_email=account_email,
                label=label,
                limit=chunk_limit,
            )
            if chunk_rows:
                return _aggregate_chunk_candidates(
                    conn,
                    chunk_rows,
                    top_messages=chunk_limit,
                    from_ts_ms=from_ts_ms,
                    to_ts_ms=to_ts_ms,
                )

    chunk_rows = fetch_chunk_vectors_for_search(
        conn,
        account_email=account_email,
        label=label,
        from_ts_ms=from_ts_ms,
        to_ts_ms=to_ts_ms,
    )
    ranked_chunks: list[_ChunkCandidate] = []
    for (
        chunk_id,
        msg_id,
        acct,
        thread_id,
        labels_json,
        chunk_index,
        _chunk_type,
        _chunk_start,
        _chunk_end,
        chunk_text,
        chunk_text_redacted,
        emb_json,
        _model,
    ) in chunk_rows:
        emb = [float(v) for v in json.loads(emb_json)]
        score = _cosine_similarity(qvec, emb)
        if score < 0:
            continue
        ranked_chunks.append(
            _ChunkCandidate(
                chunk_id=chunk_id,
                msg_id=msg_id,
                thread_id=thread_id,
                account_email=acct,
                labels=json.loads(labels_json or "[]"),
                chunk_index=int(chunk_index),
                chunk_text=chunk_text,
                chunk_text_redacted=chunk_text_redacted,
                score=score,
            )
        )

    ranked_chunks.sort(key=lambda r: (-r.score, r.msg_id, r.chunk_index, r.chunk_id))
    ranked_chunks = ranked_chunks[:chunk_limit]
    if ranked_chunks:
        return _aggregate_chunk_candidates(
            conn,
            ranked_chunks,
            top_messages=chunk_limit,
            from_ts_ms=from_ts_ms,
            to_ts_ms=to_ts_ms,
        )

    return _dense_candidates_legacy_messages(
        conn,
        cfg,
        query,
        account_email=account_email,
        label=label,
        from_ts_ms=from_ts_ms,
        to_ts_ms=to_ts_ms,
    )


def _lexical_candidates(
    conn,
    cfg: AppConfig,
    query: str,
    *,
    account_email: str | None,
    label: str | None,
    from_ts_ms: int | None,
    to_ts_ms: int | None,
) -> list[_Candidate]:
    if cfg.retrieval.lexical_backend != "fts5":
        return []

    try:
        rows = lexical_search_rows(
            conn,
            query=query,
            account_email=account_email,
            label=label,
            from_ts_ms=from_ts_ms,
            to_ts_ms=to_ts_ms,
            limit=cfg.retrieval.lexical_candidate_k,
        )
    except Exception:
        sanitized = query.replace('"', "")
        quoted = f'"{sanitized}"'
        rows = lexical_search_rows(
            conn,
            query=quoted,
            account_email=account_email,
            label=label,
            from_ts_ms=from_ts_ms,
            to_ts_ms=to_ts_ms,
            limit=cfg.retrieval.lexical_candidate_k,
        )

    out: list[_Candidate] = []
    for msg_id, acct, thread_id, labels_json, subject, snippet, body_text, bm25_score in rows:
        source_text = _compose_source_text(subject, snippet, body_text)
        out.append(
            _Candidate(
                msg_id=msg_id,
                thread_id=thread_id,
                account_email=acct,
                labels=json.loads(labels_json or "[]"),
                source_text=source_text,
                source_text_redacted=redact_text(source_text, mode="regex"),
                score=-float(bm25_score),
            )
        )
    return out


def _apply_rrf(
    dense: list[_Candidate],
    lexical: list[_Candidate],
    *,
    rrf_k: int,
) -> list[_Candidate]:
    pooled: dict[str, dict[str, Any]] = {}

    def add_ranked(items: list[_Candidate], channel: str):
        for rank, item in enumerate(items, start=1):
            bucket = pooled.setdefault(item.msg_id, {"item": item, "score": 0.0})
            bucket["score"] += 1.0 / (rrf_k + rank)
            if channel == "dense" and item.score > bucket["item"].score:
                bucket["item"] = item

    add_ranked(dense, "dense")
    add_ranked(lexical, "lexical")

    merged = [
        _Candidate(
            msg_id=value["item"].msg_id,
            thread_id=value["item"].thread_id,
            account_email=value["item"].account_email,
            labels=value["item"].labels,
            source_text=value["item"].source_text,
            source_text_redacted=value["item"].source_text_redacted,
            score=value["score"],
            chunk_hits=value["item"].chunk_hits,
            first_chunk_rank=value["item"].first_chunk_rank,
        )
        for value in pooled.values()
    ]
    merged.sort(key=lambda r: (-r.score, r.msg_id))
    return merged


def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _maybe_rerank(cfg: AppConfig, query: str, candidates: list[_Candidate]) -> list[_Candidate]:
    if not cfg.rerank.enabled or not candidates:
        return candidates

    top_n = max(1, min(cfg.rerank.top_n, len(candidates)))
    head = candidates[:top_n]
    tail = candidates[top_n:]

    try:
        model = _load_cross_encoder(cfg.rerank.model)
    except Exception:
        LOG.warning("sentence-transformers unavailable; skipping rerank")
        return candidates

    pairs = [[query, item.source_text] for item in head]
    try:
        scores = model.predict(pairs)
    except Exception:
        LOG.exception("Reranker predict failed; keeping hybrid rank")
        return candidates

    rescored = [
        _Candidate(
            msg_id=item.msg_id,
            thread_id=item.thread_id,
            account_email=item.account_email,
            labels=item.labels,
            source_text=item.source_text,
            source_text_redacted=item.source_text_redacted,
            score=float(score),
            chunk_hits=item.chunk_hits,
            first_chunk_rank=item.first_chunk_rank,
        )
        for item, score in zip(head, scores)
    ]
    rescored.sort(key=lambda r: (-r.score, r.msg_id))
    return rescored + tail


def search_vectors(
    conn,
    cfg: AppConfig,
    query: str,
    *,
    account_email: str | None = None,
    label: str | None = None,
    top_k: int = 5,
    clearance: str = "redacted",
    strategy: str | None = None,
    from_ts_ms: int | None = None,
    to_ts_ms: int | None = None,
) -> list[SearchResult]:
    chosen = (strategy or cfg.retrieval.search_strategy).strip().lower()
    if chosen not in {"dense", "lexical", "hybrid"}:
        chosen = "hybrid"

    dense = (
        _dense_candidates(
            conn,
            cfg,
            query,
            account_email=account_email,
            label=label,
            from_ts_ms=from_ts_ms,
            to_ts_ms=to_ts_ms,
        )
        if chosen in {"dense", "hybrid"}
        else []
    )
    lexical = (
        _lexical_candidates(
            conn,
            cfg,
            query,
            account_email=account_email,
            label=label,
            from_ts_ms=from_ts_ms,
            to_ts_ms=to_ts_ms,
        )
        if chosen in {"lexical", "hybrid"}
        else []
    )

    if chosen == "dense":
        ranked = dense
    elif chosen == "lexical":
        ranked = lexical
    else:
        ranked = _apply_rrf(dense, lexical, rrf_k=cfg.retrieval.rrf_k)

    ranked = _maybe_rerank(cfg, query, ranked)

    out: list[SearchResult] = []
    for item in ranked[: max(1, int(top_k))]:
        content = item.source_text_redacted if clearance == "redacted" else item.source_text
        out.append(
            SearchResult(
                score=item.score,
                msg_id=item.msg_id,
                thread_id=item.thread_id,
                account_email=item.account_email,
                labels=item.labels,
                content=content,
            )
        )
    return out
