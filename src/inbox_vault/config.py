from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AccountConfig:
    name: str
    email: str
    credentials_file: str
    token_file: str


@dataclass(slots=True)
class LLMConfig:
    enabled: bool = True
    endpoint: str = "http://localhost:8080"
    model: str = "local-model"
    timeout_seconds: float = 60.0


@dataclass(slots=True)
class DBConfig:
    path: str = "data/inbox_vault.db"
    password_env: str = "INBOX_VAULT_DB_PASSWORD"


@dataclass(slots=True)
class EmbeddingConfig:
    endpoint: str = "http://localhost:8080"
    model: str = "local-embedding-model"
    timeout_seconds: float = 60.0
    max_retries: int = 4
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 8.0
    fallback: str = "none"
    fallback_dim: int = 256


@dataclass(slots=True)
class RedactionConfig:
    mode: str = "hybrid"
    profile: str = "standard"
    instruction: str = ""
    chunk_chars: int = 1200
    model: str | None = None


@dataclass(slots=True)
class RetrievalConfig:
    search_strategy: str = "hybrid"
    lexical_backend: str = "fts5"
    rrf_k: int = 60
    dense_candidate_k: int = 100
    lexical_candidate_k: int = 100
    chunk_chars: int = 900
    chunk_overlap_chars: int = 150


@dataclass(slots=True)
class IndexingConfig:
    include_labels: list[str] = field(default_factory=lambda: ["INBOX", "SENT"])
    exclude_labels: list[str] = field(
        default_factory=lambda: [
            "SPAM",
            "TRASH",
            "CATEGORY_PROMOTIONS",
            "CATEGORY_SOCIAL",
            "CATEGORY_FORUMS",
        ]
    )
    strip_zero_width: bool = True
    collapse_whitespace: bool = True
    max_index_chars: int = 20000
    auto_index_after_ingest: bool = False
    auto_index_pending_only: bool = True
    auto_index_limit: int | None = None


@dataclass(slots=True)
class RerankConfig:
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 20


@dataclass(slots=True)
class ProfileContextConfig:
    deep_context_max_threads: int = 8
    deep_context_max_messages: int = 36
    deep_context_max_chars: int = 18000
    deep_prompt_budget_chars: int = 6000
    gog_history_enabled: bool = False
    gog_history_account: str = ""
    gog_history_command: str = "gog"
    gog_history_max_messages: int = 8
    gog_history_timeout_seconds: float = 15.0


@dataclass(slots=True)
class IngestTriageConfig:
    enabled: bool = False
    mode: str = "observe"


@dataclass(slots=True)
class AppConfig:
    accounts: list[AccountConfig]
    llm: LLMConfig
    db: DBConfig
    embeddings: EmbeddingConfig
    redaction: RedactionConfig
    retrieval: RetrievalConfig
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    profiles: ProfileContextConfig = field(default_factory=ProfileContextConfig)
    ingest_triage: IngestTriageConfig = field(default_factory=IngestTriageConfig)
    gmail_query: str = "label:inbox OR label:sent"
    gmail_idle_backfill_query: str | None = None
    gmail_idle_backfill_limit: int | None = None
    gmail_request_timeout_seconds: float = 60.0
    gmail_progress_every: int = 100


DEFAULT_CONFIG_PATH = "config.toml"


def _require_str(mapping: dict[str, Any], key: str, *, ctx: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid {ctx}.{key}: expected non-empty string")
    return value.strip()


def _parse_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Invalid {key}: expected boolean")


def _parse_positive_int(value: Any, *, key: str, min_value: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {key}: expected integer") from exc
    if parsed < min_value:
        raise ValueError(f"Invalid {key}: must be >= {min_value}")
    return parsed


def _parse_str_list(value: Any, *, key: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return [part for part in parts if part]
    if isinstance(value, list):
        out: list[str] = []
        for idx, item in enumerate(value):
            if not isinstance(item, str):
                raise ValueError(f"Invalid {key}[{idx}]: expected string")
            val = item.strip()
            if val:
                out.append(val)
        return out
    raise ValueError(f"Invalid {key}: expected list or comma-separated string")


def _parse_optional_positive_int(value: Any, *, key: str, min_value: int = 1) -> int | None:
    if value is None:
        return None
    return _parse_positive_int(value, key=key, min_value=min_value)


def load_config(path: str | None = None) -> AppConfig:
    cfg_path = Path(path or os.environ.get("INBOX_VAULT_CONFIG", DEFAULT_CONFIG_PATH))
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {cfg_path}. Copy config.example.toml to config.toml and edit."
        )

    raw = tomllib.loads(cfg_path.read_text())

    accounts_raw = raw.get("accounts", [])
    if not isinstance(accounts_raw, list) or not accounts_raw:
        raise ValueError("No [[accounts]] configured")

    accounts: list[AccountConfig] = []
    seen_emails: set[str] = set()
    for idx, item in enumerate(accounts_raw):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid accounts[{idx}]: expected table/object")

        email = _require_str(item, "email", ctx=f"accounts[{idx}]").lower()
        if email in seen_emails:
            raise ValueError(f"Duplicate account email in config: {email}")
        seen_emails.add(email)

        accounts.append(
            AccountConfig(
                name=(item.get("name") or email).strip(),
                email=email,
                credentials_file=_require_str(item, "credentials_file", ctx=f"accounts[{idx}]"),
                token_file=_require_str(item, "token_file", ctx=f"accounts[{idx}]"),
            )
        )

    db_raw = raw.get("database", {})
    llm_raw = raw.get("llm", {})
    embeddings_raw = raw.get("embeddings", {})
    redaction_raw = raw.get("redaction", {})
    retrieval_raw = raw.get("retrieval", {})
    indexing_raw = raw.get("indexing", {})
    rerank_raw = raw.get("rerank", {})
    profiles_raw = raw.get("profiles", {})
    ingest_triage_raw = raw.get("ingest_triage", {})
    gmail_raw = raw.get("gmail", {})

    if not isinstance(db_raw, dict):
        raise ValueError("Invalid [database] section: expected table/object")
    if not isinstance(llm_raw, dict):
        raise ValueError("Invalid [llm] section: expected table/object")
    if not isinstance(embeddings_raw, dict):
        raise ValueError("Invalid [embeddings] section: expected table/object")
    if not isinstance(redaction_raw, dict):
        raise ValueError("Invalid [redaction] section: expected table/object")
    if not isinstance(retrieval_raw, dict):
        raise ValueError("Invalid [retrieval] section: expected table/object")
    if not isinstance(indexing_raw, dict):
        raise ValueError("Invalid [indexing] section: expected table/object")
    if not isinstance(rerank_raw, dict):
        raise ValueError("Invalid [rerank] section: expected table/object")
    if not isinstance(profiles_raw, dict):
        raise ValueError("Invalid [profiles] section: expected table/object")
    if not isinstance(ingest_triage_raw, dict):
        raise ValueError("Invalid [ingest_triage] section: expected table/object")
    if not isinstance(gmail_raw, dict):
        raise ValueError("Invalid [gmail] section: expected table/object")

    llm_enabled = _parse_bool(llm_raw.get("enabled", True), key="llm.enabled")

    timeout_raw = llm_raw.get("timeout_seconds", 60.0)
    try:
        timeout_seconds = float(timeout_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid llm.timeout_seconds: expected number") from exc
    if timeout_seconds <= 0:
        raise ValueError("Invalid llm.timeout_seconds: must be > 0")

    gmail_query = gmail_raw.get("query", "label:inbox OR label:sent")
    if not isinstance(gmail_query, str) or not gmail_query.strip():
        raise ValueError("Invalid gmail.query: expected non-empty string")

    gmail_idle_backfill_query_raw = gmail_raw.get("idle_backfill_query")
    if gmail_idle_backfill_query_raw is None:
        gmail_idle_backfill_query = None
    elif isinstance(gmail_idle_backfill_query_raw, str) and gmail_idle_backfill_query_raw.strip():
        gmail_idle_backfill_query = gmail_idle_backfill_query_raw.strip()
    else:
        raise ValueError("Invalid gmail.idle_backfill_query: expected non-empty string when set")

    gmail_idle_backfill_limit_raw = gmail_raw.get("idle_backfill_limit")
    if gmail_idle_backfill_limit_raw is None:
        gmail_idle_backfill_limit = None
    else:
        gmail_idle_backfill_limit = _parse_positive_int(
            gmail_idle_backfill_limit_raw,
            key="gmail.idle_backfill_limit",
            min_value=0,
        )

    gmail_timeout_raw = gmail_raw.get("request_timeout_seconds", 60.0)
    try:
        gmail_request_timeout_seconds = float(gmail_timeout_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid gmail.request_timeout_seconds: expected number") from exc
    if gmail_request_timeout_seconds <= 0:
        raise ValueError("Invalid gmail.request_timeout_seconds: must be > 0")

    gmail_progress_every = _parse_positive_int(
        gmail_raw.get("progress_every", 100),
        key="gmail.progress_every",
        min_value=1,
    )

    llm_endpoint = str(llm_raw.get("endpoint", "http://localhost:8080")).strip()
    llm_model = str(llm_raw.get("model", "local-model")).strip()

    embed_timeout_raw = embeddings_raw.get("timeout_seconds", 60.0)
    try:
        embed_timeout_seconds = float(embed_timeout_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid embeddings.timeout_seconds: expected number") from exc
    if embed_timeout_seconds <= 0:
        raise ValueError("Invalid embeddings.timeout_seconds: must be > 0")

    embed_endpoint = str(embeddings_raw.get("endpoint", "http://localhost:8080")).strip()
    embed_model = str(embeddings_raw.get("model", "local-embedding-model")).strip()
    embed_max_retries = _parse_positive_int(
        embeddings_raw.get("max_retries", 4),
        key="embeddings.max_retries",
        min_value=0,
    )

    embed_backoff_base_raw = embeddings_raw.get("backoff_base_seconds", 0.5)
    try:
        embed_backoff_base_seconds = float(embed_backoff_base_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid embeddings.backoff_base_seconds: expected number") from exc
    if embed_backoff_base_seconds < 0:
        raise ValueError("Invalid embeddings.backoff_base_seconds: must be >= 0")

    embed_backoff_max_raw = embeddings_raw.get("backoff_max_seconds", 8.0)
    try:
        embed_backoff_max_seconds = float(embed_backoff_max_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid embeddings.backoff_max_seconds: expected number") from exc
    if embed_backoff_max_seconds < 0:
        raise ValueError("Invalid embeddings.backoff_max_seconds: must be >= 0")
    if embed_backoff_max_seconds < embed_backoff_base_seconds:
        raise ValueError(
            "Invalid embeddings.backoff_max_seconds: must be >= embeddings.backoff_base_seconds"
        )

    embed_fallback = str(embeddings_raw.get("fallback", "none")).strip().lower()
    if embed_fallback not in {"none", "hash"}:
        raise ValueError("Invalid embeddings.fallback: expected one of none|hash")

    embed_fallback_dim = _parse_positive_int(
        embeddings_raw.get("fallback_dim", 256),
        key="embeddings.fallback_dim",
        min_value=8,
    )

    db_path = str(db_raw.get("path", "data/inbox_vault.db")).strip()
    db_password_env = str(db_raw.get("password_env", "INBOX_VAULT_DB_PASSWORD")).strip()

    redaction_mode = str(redaction_raw.get("mode", "hybrid")).strip().lower()
    if redaction_mode not in {"regex", "model", "hybrid"}:
        raise ValueError("Invalid redaction.mode: expected one of regex|model|hybrid")

    redaction_profile = str(redaction_raw.get("profile", "standard")).strip()
    redaction_instruction = str(redaction_raw.get("instruction", "")).strip()
    redaction_model_raw = redaction_raw.get("model")
    redaction_model = None if redaction_model_raw is None else str(redaction_model_raw).strip()

    redaction_chunk_chars = _parse_positive_int(
        redaction_raw.get("chunk_chars", 1200),
        key="redaction.chunk_chars",
        min_value=200,
    )

    retrieval_strategy = str(retrieval_raw.get("search_strategy", "hybrid")).strip().lower()
    if retrieval_strategy not in {"dense", "lexical", "hybrid"}:
        raise ValueError("Invalid retrieval.search_strategy: expected one of dense|lexical|hybrid")

    lexical_backend = str(retrieval_raw.get("lexical_backend", "fts5")).strip().lower()
    if lexical_backend not in {"fts5", "none"}:
        raise ValueError("Invalid retrieval.lexical_backend: expected one of fts5|none")

    rrf_k = _parse_positive_int(retrieval_raw.get("rrf_k", 60), key="retrieval.rrf_k")
    dense_candidate_k = _parse_positive_int(
        retrieval_raw.get("dense_candidate_k", 100),
        key="retrieval.dense_candidate_k",
    )
    lexical_candidate_k = _parse_positive_int(
        retrieval_raw.get("lexical_candidate_k", 100),
        key="retrieval.lexical_candidate_k",
    )
    chunk_chars = _parse_positive_int(
        retrieval_raw.get("chunk_chars", 900),
        key="retrieval.chunk_chars",
        min_value=200,
    )
    chunk_overlap_chars = _parse_positive_int(
        retrieval_raw.get("chunk_overlap_chars", 150),
        key="retrieval.chunk_overlap_chars",
        min_value=0,
    )
    if chunk_overlap_chars >= chunk_chars:
        raise ValueError("Invalid retrieval.chunk_overlap_chars: must be < retrieval.chunk_chars")

    indexing_include_labels = [
        label.strip().upper()
        for label in _parse_str_list(
            indexing_raw.get("include_labels", ["INBOX", "SENT"]),
            key="indexing.include_labels",
        )
        if label.strip()
    ]
    indexing_exclude_labels = [
        label.strip().upper()
        for label in _parse_str_list(
            indexing_raw.get(
                "exclude_labels",
                ["SPAM", "TRASH", "CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_FORUMS"],
            ),
            key="indexing.exclude_labels",
        )
        if label.strip()
    ]
    indexing_strip_zero_width = _parse_bool(
        indexing_raw.get("strip_zero_width", True), key="indexing.strip_zero_width"
    )
    indexing_collapse_whitespace = _parse_bool(
        indexing_raw.get("collapse_whitespace", True), key="indexing.collapse_whitespace"
    )
    indexing_max_chars = _parse_positive_int(
        indexing_raw.get("max_index_chars", 20000),
        key="indexing.max_index_chars",
        min_value=200,
    )
    indexing_auto_after_ingest = _parse_bool(
        indexing_raw.get("auto_index_after_ingest", False),
        key="indexing.auto_index_after_ingest",
    )
    indexing_auto_pending_only = _parse_bool(
        indexing_raw.get("auto_index_pending_only", True),
        key="indexing.auto_index_pending_only",
    )
    indexing_auto_limit = _parse_optional_positive_int(
        indexing_raw.get("auto_index_limit"),
        key="indexing.auto_index_limit",
        min_value=1,
    )

    rerank_enabled = _parse_bool(rerank_raw.get("enabled", False), key="rerank.enabled")
    rerank_model = str(rerank_raw.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")).strip()
    rerank_top_n = _parse_positive_int(rerank_raw.get("top_n", 20), key="rerank.top_n")

    deep_context_max_threads = _parse_positive_int(
        profiles_raw.get("deep_context_max_threads", 8),
        key="profiles.deep_context_max_threads",
        min_value=1,
    )
    deep_context_max_messages = _parse_positive_int(
        profiles_raw.get("deep_context_max_messages", 36),
        key="profiles.deep_context_max_messages",
        min_value=2,
    )
    deep_context_max_chars = _parse_positive_int(
        profiles_raw.get("deep_context_max_chars", 18000),
        key="profiles.deep_context_max_chars",
        min_value=500,
    )
    deep_prompt_budget_chars = _parse_positive_int(
        profiles_raw.get("deep_prompt_budget_chars", 6000),
        key="profiles.deep_prompt_budget_chars",
        min_value=1200,
    )
    gog_history_enabled = _parse_bool(
        profiles_raw.get("gog_history_enabled", False),
        key="profiles.gog_history_enabled",
    )
    gog_history_account = str(profiles_raw.get("gog_history_account", "")).strip()
    gog_history_command = str(profiles_raw.get("gog_history_command", "gog")).strip()
    gog_history_max_messages = _parse_positive_int(
        profiles_raw.get("gog_history_max_messages", 8),
        key="profiles.gog_history_max_messages",
        min_value=1,
    )
    gog_timeout_raw = profiles_raw.get("gog_history_timeout_seconds", 15.0)
    try:
        gog_history_timeout_seconds = float(gog_timeout_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid profiles.gog_history_timeout_seconds: expected number") from exc
    if gog_history_timeout_seconds <= 0:
        raise ValueError("Invalid profiles.gog_history_timeout_seconds: must be > 0")

    if not llm_endpoint:
        raise ValueError("Invalid llm.endpoint: expected non-empty string")
    if not llm_model:
        raise ValueError("Invalid llm.model: expected non-empty string")
    if not embed_endpoint:
        raise ValueError("Invalid embeddings.endpoint: expected non-empty string")
    if not embed_model:
        raise ValueError("Invalid embeddings.model: expected non-empty string")
    if not db_path:
        raise ValueError("Invalid database.path: expected non-empty string")
    if not db_password_env:
        raise ValueError("Invalid database.password_env: expected non-empty string")
    if rerank_enabled and not rerank_model:
        raise ValueError("Invalid rerank.model: expected non-empty string")
    if gog_history_enabled and not gog_history_command:
        raise ValueError(
            "Invalid profiles.gog_history_command: expected non-empty string when enabled"
        )

    ingest_triage_enabled = _parse_bool(
        ingest_triage_raw.get("enabled", False),
        key="ingest_triage.enabled",
    )
    ingest_triage_mode = str(ingest_triage_raw.get("mode", "observe")).strip().lower()
    if ingest_triage_mode not in {"observe", "enforce"}:
        raise ValueError("Invalid ingest_triage.mode: expected one of observe|enforce")

    return AppConfig(
        accounts=accounts,
        llm=LLMConfig(
            enabled=llm_enabled,
            endpoint=llm_endpoint,
            model=llm_model,
            timeout_seconds=timeout_seconds,
        ),
        db=DBConfig(
            path=db_path,
            password_env=db_password_env,
        ),
        embeddings=EmbeddingConfig(
            endpoint=embed_endpoint,
            model=embed_model,
            timeout_seconds=embed_timeout_seconds,
            max_retries=embed_max_retries,
            backoff_base_seconds=embed_backoff_base_seconds,
            backoff_max_seconds=embed_backoff_max_seconds,
            fallback=embed_fallback,
            fallback_dim=embed_fallback_dim,
        ),
        redaction=RedactionConfig(
            mode=redaction_mode,
            profile=redaction_profile,
            instruction=redaction_instruction,
            chunk_chars=redaction_chunk_chars,
            model=redaction_model or None,
        ),
        retrieval=RetrievalConfig(
            search_strategy=retrieval_strategy,
            lexical_backend=lexical_backend,
            rrf_k=rrf_k,
            dense_candidate_k=dense_candidate_k,
            lexical_candidate_k=lexical_candidate_k,
            chunk_chars=chunk_chars,
            chunk_overlap_chars=chunk_overlap_chars,
        ),
        indexing=IndexingConfig(
            include_labels=indexing_include_labels,
            exclude_labels=indexing_exclude_labels,
            strip_zero_width=indexing_strip_zero_width,
            collapse_whitespace=indexing_collapse_whitespace,
            max_index_chars=indexing_max_chars,
            auto_index_after_ingest=indexing_auto_after_ingest,
            auto_index_pending_only=indexing_auto_pending_only,
            auto_index_limit=indexing_auto_limit,
        ),
        rerank=RerankConfig(
            enabled=rerank_enabled,
            model=rerank_model,
            top_n=rerank_top_n,
        ),
        profiles=ProfileContextConfig(
            deep_context_max_threads=deep_context_max_threads,
            deep_context_max_messages=deep_context_max_messages,
            deep_context_max_chars=deep_context_max_chars,
            deep_prompt_budget_chars=deep_prompt_budget_chars,
            gog_history_enabled=gog_history_enabled,
            gog_history_account=gog_history_account,
            gog_history_command=gog_history_command,
            gog_history_max_messages=gog_history_max_messages,
            gog_history_timeout_seconds=gog_history_timeout_seconds,
        ),
        ingest_triage=IngestTriageConfig(
            enabled=ingest_triage_enabled,
            mode=ingest_triage_mode,
        ),
        gmail_query=gmail_query.strip(),
        gmail_idle_backfill_query=gmail_idle_backfill_query,
        gmail_idle_backfill_limit=gmail_idle_backfill_limit,
        gmail_request_timeout_seconds=gmail_request_timeout_seconds,
        gmail_progress_every=gmail_progress_every,
    )


def resolve_password(db_cfg: DBConfig) -> str:
    pwd = os.environ.get(db_cfg.password_env)
    if pwd is None or not pwd.strip():
        raise RuntimeError(
            f"Missing DB password env var: {db_cfg.password_env}. Set it before running."
        )
    return pwd
