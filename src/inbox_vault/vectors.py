from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from .config import AppConfig
from .db import (
    DBLockRetryExhausted,
    delete_message_chunk_vectors,
    fetch_applied_light_message_ids,
    fetch_chunk_vectors_for_search,
    fetch_messages_by_ids,
    fetch_redaction_entries,
    fetch_vectors_for_search,
    get_vector_state,
    lexical_search_rows,
    lexical_search_rows_redacted,
    prune_invalid_redaction_entries,
    upsert_message_chunk_vector,
    upsert_message_fts_redacted,
    upsert_message_vector,
    upsert_redaction_entries,
    upsert_vector_state,
    vector_index_source_rows,
    vector_level_counts,
)
from .llm import embedding_vector, embedding_vectors
from .redaction import (
    REDACTION_POLICY_VERSION,
    PersistentRedactionMap,
    redact_text,
    redact_with_persistent_map,
)

LOG = logging.getLogger(__name__)
INDEX_LEVEL_REDACTED = "redacted"
INDEX_LEVEL_FULL = "full"
INDEX_LEVEL_AUTO = "auto"
_INDEX_EMBED_MESSAGE_BATCH_SIZE = 16


@dataclass(slots=True)
class SearchResult:
    score: float
    msg_id: str
    thread_id: str | None
    account_email: str
    labels: list[str]
    content: str


@dataclass(slots=True)
class SearchDiagnostics:
    requested_level: str
    used_level: str
    fallback_from_level: str | None = None
    full_level_available: bool = False


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


@dataclass(slots=True)
class _PreparedIndexMessage:
    row_idx: int
    total_rows: int
    message_started: float
    msg_id: str
    account_email: str
    thread_id: str | None
    labels: list[str]
    source_text: str
    source_text_redacted: str
    embed_source_text: str
    fingerprint: str
    chunks: list[_Chunk]
    chunk_text_redacted: list[str]
    chunk_embedding_inputs: list[str]
    persisted_entries: list[dict[str, str]]


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


def _vector_content_hash(*, source_text: str, index_level: str) -> str:
    payload = {
        "index_level": index_level,
        "policy_version": REDACTION_POLICY_VERSION,
        "source_hash": _content_hash(source_text),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _vector_levels_available(conn) -> set[str]:
    counts = vector_level_counts(conn)
    return {level for level, bucket in counts.items() if int(bucket.get("messages", 0)) > 0}


def _resolve_effective_search_level(
    conn,
    *,
    clearance: str,
    search_level: str,
) -> SearchDiagnostics:
    requested = (search_level or INDEX_LEVEL_AUTO).strip().lower()
    if requested not in {INDEX_LEVEL_AUTO, INDEX_LEVEL_REDACTED, INDEX_LEVEL_FULL}:
        requested = INDEX_LEVEL_AUTO

    available = _vector_levels_available(conn)
    full_available = INDEX_LEVEL_FULL in available
    if requested == INDEX_LEVEL_FULL:
        if not full_available:
            raise ValueError(
                "full search level is unavailable; run `index-vectors --index-level full` first."
            )
        return SearchDiagnostics(
            requested_level=INDEX_LEVEL_FULL,
            used_level=INDEX_LEVEL_FULL,
            full_level_available=True,
        )
    if requested == INDEX_LEVEL_REDACTED:
        return SearchDiagnostics(
            requested_level=INDEX_LEVEL_REDACTED,
            used_level=INDEX_LEVEL_REDACTED,
            full_level_available=full_available,
        )

    desired = INDEX_LEVEL_FULL if clearance == "full" and full_available else INDEX_LEVEL_REDACTED
    fallback = INDEX_LEVEL_FULL if clearance == "full" and not full_available else None
    return SearchDiagnostics(
        requested_level=INDEX_LEVEL_AUTO,
        used_level=desired,
        fallback_from_level=fallback,
        full_level_available=full_available,
    )


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


def _pending_message_ids_for_index(
    conn,
    cfg: AppConfig,
    *,
    account_email: str | None = None,
    include_labels: list[str] | None = None,
    exclude_labels: list[str] | None = None,
    max_index_chars: int | None = None,
    limit: int | None = None,
    index_level: str = INDEX_LEVEL_REDACTED,
    skip_applied_light: bool = False,
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
    applied_light_ids = fetch_applied_light_message_ids(conn) if skip_applied_light else set()

    pending_msg_ids: list[str] = []
    for msg_id, _acct, _thread_id, subject, snippet, body_text, labels_json in rows:
        if msg_id in applied_light_ids:
            continue
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
        expected_hash = _vector_content_hash(source_text=source, index_level=index_level)
        prior = get_vector_state(conn, msg_id, index_level=index_level)
        if not prior or str(prior[0]) != expected_hash or str(prior[1] or "") != REDACTION_POLICY_VERSION:
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
    index_level: str = INDEX_LEVEL_REDACTED,
    skip_applied_light: bool = False,
) -> int:
    return len(
        _pending_message_ids_for_index(
            conn,
            cfg,
            account_email=account_email,
            include_labels=include_labels,
            exclude_labels=exclude_labels,
            max_index_chars=max_index_chars,
            index_level=index_level,
            skip_applied_light=skip_applied_light,
        )
    )


def _embed_prepared_messages(
    cfg: AppConfig,
    prepared_messages: list[_PreparedIndexMessage],
) -> tuple[dict[str, tuple[list[float], list[list[float]]]], set[str]]:
    if not prepared_messages:
        return {}, set()

    all_texts: list[str] = []
    layout: list[tuple[_PreparedIndexMessage, int, int]] = []
    for item in prepared_messages:
        all_texts.append(item.embed_source_text)
        all_texts.extend(item.chunk_embedding_inputs)
        layout.append((item, len(all_texts) - (1 + len(item.chunk_embedding_inputs)), 1 + len(item.chunk_embedding_inputs)))

    results: dict[str, tuple[list[float], list[list[float]]]] = {}
    failed: set[str] = set()

    try:
        vectors = embedding_vectors(cfg.embeddings, all_texts)
        for item, start, count in layout:
            item_vectors = vectors[start : start + count]
            results[item.msg_id] = (item_vectors[0], item_vectors[1:])
        return results, failed
    except Exception:
        LOG.exception(
            "Batched embedding generation failed for %s messages; retrying individually",
            len(prepared_messages),
        )

    for item in prepared_messages:
        try:
            item_vectors = embedding_vectors(
                cfg.embeddings,
                [item.embed_source_text, *item.chunk_embedding_inputs],
            )
            results[item.msg_id] = (item_vectors[0], item_vectors[1:])
        except Exception:
            LOG.exception("Vector generation failed for message_id=%s", item.msg_id)
            failed.add(item.msg_id)
    return results, failed


def _persist_prepared_message(
    conn,
    cfg: AppConfig,
    *,
    prepared: _PreparedIndexMessage,
    chosen_index_level: str,
    message_embedding: list[float],
    chunk_embeddings: list[list[float]],
    stats: dict[str, int | str],
    lock_max_retries: int,
    lock_backoff_base_seconds: float,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> bool:
    msg_id = prepared.msg_id
    acct = prepared.account_email

    if prepared.persisted_entries:
        try:
            stats["lock_retries"] += upsert_redaction_entries(
                conn,
                scope_type="account",
                scope_id=acct,
                entries=prepared.persisted_entries,
                lock_max_retries=lock_max_retries,
                lock_backoff_base_seconds=lock_backoff_base_seconds,
            )
            stats["redaction_entries_added"] += len(prepared.persisted_entries)
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
                        "position": prepared.row_idx,
                        "total": prepared.total_rows,
                        "msg_id": msg_id,
                        "account_email": acct,
                        "substep": "upsert-redaction-entries",
                        "elapsed_ms": int((time.perf_counter() - prepared.message_started) * 1000),
                        "indexed": int(stats["indexed"]),
                        "chunks_indexed": int(stats["chunks_indexed"]),
                        "failed": int(stats["failed"]),
                    }
                )
            return False

    try:
        stats["lock_retries"] += upsert_message_vector(
            conn,
            msg_id=msg_id,
            index_level=chosen_index_level,
            account_email=acct,
            thread_id=prepared.thread_id,
            labels=prepared.labels,
            source_text=prepared.source_text,
            source_text_redacted=prepared.source_text_redacted,
            embedding=message_embedding,
            embedding_model=cfg.embeddings.model,
            content_hash=prepared.fingerprint,
            redaction_policy_version=REDACTION_POLICY_VERSION,
            lock_max_retries=lock_max_retries,
            lock_backoff_base_seconds=lock_backoff_base_seconds,
        )
        upsert_message_fts_redacted(
            conn,
            msg_id=msg_id,
            account_email=acct,
            thread_id=prepared.thread_id,
            labels=prepared.labels,
            redacted_content=prepared.source_text_redacted,
        )
        stats["lock_retries"] += upsert_vector_state(
            conn,
            msg_id=msg_id,
            index_level=chosen_index_level,
            content_hash=prepared.fingerprint,
            redaction_policy_version=REDACTION_POLICY_VERSION,
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
                    "position": prepared.row_idx,
                    "total": prepared.total_rows,
                    "msg_id": msg_id,
                    "account_email": acct,
                    "substep": "upsert-message-vector",
                    "elapsed_ms": int((time.perf_counter() - prepared.message_started) * 1000),
                    "indexed": int(stats["indexed"]),
                    "chunks_indexed": int(stats["chunks_indexed"]),
                    "failed": int(stats["failed"]),
                }
            )
        return False

    try:
        stats["lock_retries"] += delete_message_chunk_vectors(
            conn,
            msg_id=msg_id,
            index_level=chosen_index_level,
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
                    "position": prepared.row_idx,
                    "total": prepared.total_rows,
                    "msg_id": msg_id,
                    "account_email": acct,
                    "substep": "delete-old-chunks",
                    "elapsed_ms": int((time.perf_counter() - prepared.message_started) * 1000),
                    "indexed": int(stats["indexed"]),
                    "chunks_indexed": int(stats["chunks_indexed"]),
                    "failed": int(stats["failed"]),
                }
            )
        return False

    for chunk, chunk_redacted, chunk_input, chunk_embedding in zip(
        prepared.chunks,
        prepared.chunk_text_redacted,
        prepared.chunk_embedding_inputs,
        chunk_embeddings,
    ):
        chunk_fingerprint = _vector_content_hash(
            source_text=(
                f"{msg_id}|{chunk.chunk_type}|{chunk.chunk_start}|{chunk.chunk_end}|{chunk_input}"
            ),
            index_level=chosen_index_level,
        )
        try:
            stats["lock_retries"] += upsert_message_chunk_vector(
                conn,
                chunk_id=chunk.chunk_id,
                index_level=chosen_index_level,
                msg_id=msg_id,
                account_email=acct,
                thread_id=prepared.thread_id,
                labels=prepared.labels,
                chunk_index=chunk.chunk_index,
                chunk_type=chunk.chunk_type,
                chunk_start=chunk.chunk_start,
                chunk_end=chunk.chunk_end,
                chunk_text=chunk.text,
                chunk_text_redacted=chunk_redacted,
                embedding=chunk_embedding,
                embedding_model=cfg.embeddings.model,
                content_hash=chunk_fingerprint,
                redaction_policy_version=REDACTION_POLICY_VERSION,
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
                        "position": prepared.row_idx,
                        "total": prepared.total_rows,
                        "msg_id": msg_id,
                        "account_email": acct,
                        "substep": "upsert-chunk",
                        "chunk_id": chunk.chunk_id,
                        "elapsed_ms": int((time.perf_counter() - prepared.message_started) * 1000),
                        "indexed": int(stats["indexed"]),
                        "chunks_indexed": int(stats["chunks_indexed"]),
                        "failed": int(stats["failed"]),
                    }
                )
            return False
        stats["chunks_indexed"] += 1

    stats["indexed"] += 1
    if progress_callback:
        elapsed_s = max(0.0001, time.perf_counter() - prepared.message_started)
        progress_callback(
            {
                "event": "message_done",
                "position": prepared.row_idx,
                "total": prepared.total_rows,
                "msg_id": msg_id,
                "account_email": acct,
                "chunk_count": len(prepared.chunks),
                "elapsed_ms": int(elapsed_s * 1000),
                "messages_per_min": round(60.0 / elapsed_s, 2),
                "indexed": int(stats["indexed"]),
                "chunks_indexed": int(stats["chunks_indexed"]),
                "failed": int(stats["failed"]),
            }
        )
    return True


def index_vectors(
    conn,
    cfg: AppConfig,
    *,
    index_level: str = INDEX_LEVEL_REDACTED,
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
    skip_applied_light: bool = False,
) -> dict[str, int | str]:
    chosen_index_level = (index_level or INDEX_LEVEL_REDACTED).strip().lower()
    if chosen_index_level not in {INDEX_LEVEL_REDACTED, INDEX_LEVEL_FULL}:
        raise ValueError(f"Unsupported index_level: {index_level}")

    stats: dict[str, int | str] = {
        "scanned": 0,
        "indexed": 0,
        "unchanged": 0,
        "skipped_filtered": 0,
        "skipped_triage_light": 0,
        "failed": 0,
        "chunks_indexed": 0,
        "lock_retries": 0,
        "lock_errors": 0,
        "redaction_entries_added": 0,
        "redaction_entries_pruned": 0,
        "index_level": chosen_index_level,
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
            index_level=chosen_index_level,
            skip_applied_light=skip_applied_light,
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
    applied_light_ids = fetch_applied_light_message_ids(conn) if skip_applied_light else set()

    redaction_maps_by_account: dict[str, PersistentRedactionMap] = {}
    pruned_accounts: set[str] = set()
    pending_prepared: list[_PreparedIndexMessage] = []

    def _flush_pending_batch() -> None:
        nonlocal indexed_since_last_commit, pending_prepared
        if not pending_prepared:
            return

        embedding_results, embedding_failures = _embed_prepared_messages(cfg, pending_prepared)
        for prepared in pending_prepared:
            if prepared.msg_id in embedding_failures:
                stats["failed"] += 1
                if progress_callback:
                    progress_callback(
                        {
                            "event": "message_failed",
                            "position": prepared.row_idx,
                            "total": prepared.total_rows,
                            "msg_id": prepared.msg_id,
                            "account_email": prepared.account_email,
                            "substep": "embed-batch",
                            "elapsed_ms": int(
                                (time.perf_counter() - prepared.message_started) * 1000
                            ),
                            "indexed": int(stats["indexed"]),
                            "chunks_indexed": int(stats["chunks_indexed"]),
                            "failed": int(stats["failed"]),
                        }
                    )
                continue

            message_embedding, chunk_embeddings = embedding_results[prepared.msg_id]
            if not _persist_prepared_message(
                conn,
                cfg,
                prepared=prepared,
                chosen_index_level=chosen_index_level,
                message_embedding=message_embedding,
                chunk_embeddings=chunk_embeddings,
                stats=stats,
                lock_max_retries=lock_max_retries,
                lock_backoff_base_seconds=lock_backoff_base_seconds,
                progress_callback=progress_callback,
            ):
                continue

            indexed_since_last_commit += 1
            if indexed_since_last_commit >= safe_commit_every_messages:
                conn.commit()
                indexed_since_last_commit = 0

        pending_prepared = []

    for row_idx, (msg_id, acct, thread_id, subject, snippet, body_text, labels_json) in enumerate(
        rows, start=1
    ):
        message_started = time.perf_counter()
        if msg_id in applied_light_ids:
            stats["skipped_triage_light"] += 1
            continue
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
        fingerprint = _vector_content_hash(source_text=source_text, index_level=chosen_index_level)
        prior = get_vector_state(conn, msg_id, index_level=chosen_index_level)
        if (
            prior
            and str(prior[0]) == fingerprint
            and str(prior[1] or "") == REDACTION_POLICY_VERSION
            and not force
        ):
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
            if acct not in pruned_accounts:
                stats["redaction_entries_pruned"] += prune_invalid_redaction_entries(
                    conn,
                    scope_type="account",
                    scope_id=acct,
                    lock_max_retries=lock_max_retries,
                    lock_backoff_base_seconds=lock_backoff_base_seconds,
                )
                pruned_accounts.add(acct)
            scope_rows = fetch_redaction_entries(conn, scope_type="account", scope_id=acct)
            redaction_map = PersistentRedactionMap.from_rows(scope_rows)
            redaction_maps_by_account[acct] = redaction_map

        try:
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
            embed_source_text = (
                source_text_redacted if chosen_index_level == INDEX_LEVEL_REDACTED else source_text
            )
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

        if progress_callback:
            progress_callback(
                {
                    "event": "message_substep",
                    "position": row_idx,
                    "total": total_rows,
                    "msg_id": msg_id,
                    "account_email": acct,
                    "substep": "embed-batch-queued",
                    "chunk_count": len(chunks),
                }
            )

        pending_prepared.append(
            _PreparedIndexMessage(
                row_idx=row_idx,
                total_rows=total_rows,
                message_started=message_started,
                msg_id=msg_id,
                account_email=acct,
                thread_id=thread_id,
                labels=labels,
                source_text=source_text,
                source_text_redacted=source_text_redacted,
                embed_source_text=embed_source_text,
                fingerprint=fingerprint,
                chunks=chunks,
                chunk_text_redacted=redaction_result.chunk_text_redacted,
                chunk_embedding_inputs=[
                    item if chosen_index_level == INDEX_LEVEL_REDACTED else chunk.text
                    for chunk, item in zip(chunks, redaction_result.chunk_text_redacted)
                ],
                persisted_entries=redaction_result.persisted_entries,
            )
        )
        if len(pending_prepared) >= _INDEX_EMBED_MESSAGE_BATCH_SIZE:
            _flush_pending_batch()

    _flush_pending_batch()
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


def _aggregate_chunk_candidates(
    conn,
    chunks: list[_ChunkCandidate],
    *,
    index_level: str,
    top_messages: int,
    from_ts_ms: int | None,
    to_ts_ms: int | None,
) -> list[_Candidate]:
    if not chunks:
        return []

    msg_rows = fetch_messages_by_ids(conn, [item.msg_id for item in chunks], index_level=index_level)
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
    index_level: str,
    account_email: str | None,
    label: str | None,
    from_ts_ms: int | None,
    to_ts_ms: int | None,
) -> list[_Candidate]:
    qvec = embedding_vector(cfg.embeddings, query)

    chunk_limit = max(1, int(cfg.retrieval.dense_candidate_k))

    chunk_rows = fetch_chunk_vectors_for_search(
        conn,
        index_level=index_level,
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
            index_level=index_level,
            top_messages=chunk_limit,
            from_ts_ms=from_ts_ms,
            to_ts_ms=to_ts_ms,
        )
    rows = fetch_vectors_for_search(
        conn,
        index_level=index_level,
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


def _lexical_candidates(
    conn,
    cfg: AppConfig,
    query: str,
    *,
    index_level: str,
    account_email: str | None,
    label: str | None,
    from_ts_ms: int | None,
    to_ts_ms: int | None,
) -> list[_Candidate]:
    if cfg.retrieval.lexical_backend != "fts5":
        return []

    try:
        lexical_fn = lexical_search_rows if index_level == INDEX_LEVEL_FULL else lexical_search_rows_redacted
        rows = lexical_fn(
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
        rows = lexical_fn(
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
    search_level: str = INDEX_LEVEL_AUTO,
    strategy: str | None = None,
    from_ts_ms: int | None = None,
    to_ts_ms: int | None = None,
    include_diagnostics: bool = False,
) -> list[SearchResult] | tuple[list[SearchResult], SearchDiagnostics]:
    chosen = (strategy or cfg.retrieval.search_strategy).strip().lower()
    if chosen not in {"dense", "lexical", "hybrid"}:
        chosen = "hybrid"
    diagnostics = _resolve_effective_search_level(
        conn,
        clearance=clearance,
        search_level=search_level,
    )

    dense = (
        _dense_candidates(
            conn,
            cfg,
            query,
            index_level=diagnostics.used_level,
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
            index_level=diagnostics.used_level,
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
    if include_diagnostics:
        return out, diagnostics
    return out
