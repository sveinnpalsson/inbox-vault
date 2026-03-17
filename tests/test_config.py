from __future__ import annotations

from pathlib import Path

import pytest

from inbox_vault.config import DEFAULT_CONFIG_PATH, load_config, resolve_password


def test_load_config_from_env_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        """
[[accounts]]
email = "acct@example.com"
credentials_file = "credentials.json"
token_file = "token.json"

[database]
path = "data/unit.db"
password_env = "UNIT_DB_PASSWORD"

[llm]
enabled = false

[gmail]
query = "label:inbox"
""".strip()
    )
    monkeypatch.setenv("INBOX_VAULT_CONFIG", str(cfg_path))

    cfg = load_config()
    assert cfg.accounts[0].name == "acct@example.com"
    assert cfg.accounts[0].email == "acct@example.com"
    assert cfg.db.password_env == "UNIT_DB_PASSWORD"
    assert cfg.db.path == "data/unit.db"
    assert cfg.llm.enabled is False
    assert cfg.embeddings.endpoint == "http://localhost:8080"
    assert cfg.embeddings.max_retries == 4
    assert cfg.embeddings.fallback == "none"
    assert cfg.redaction.mode == "hybrid"
    assert cfg.retrieval.search_strategy == "hybrid"
    assert cfg.retrieval.vector_backend == "sqlite"
    assert cfg.retrieval.chunk_chars == 900
    assert cfg.retrieval.chunk_overlap_chars == 150
    assert cfg.indexing.include_labels == ["INBOX", "SENT"]
    assert "SPAM" in cfg.indexing.exclude_labels
    assert cfg.indexing.strip_zero_width is True
    assert cfg.indexing.max_index_chars == 20000
    assert cfg.indexing.auto_index_after_ingest is False
    assert cfg.indexing.auto_index_pending_only is True
    assert cfg.indexing.auto_index_limit is None
    assert cfg.rerank.enabled is False
    assert cfg.profiles.deep_context_max_threads == 8
    assert cfg.profiles.deep_context_max_messages == 36
    assert cfg.profiles.deep_context_max_chars == 18000
    assert cfg.profiles.deep_prompt_budget_chars == 6000
    assert cfg.profiles.gog_history_enabled is False
    assert cfg.profiles.gog_history_max_messages == 8
    assert cfg.gmail_query == "label:inbox"
    assert cfg.gmail_idle_backfill_query is None
    assert cfg.gmail_idle_backfill_limit is None
    assert cfg.gmail_request_timeout_seconds == 60.0
    assert cfg.gmail_progress_every == 100


def test_load_config_uses_default_when_no_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = tmp_path / DEFAULT_CONFIG_PATH
    cfg_path.write_text(
        """
[[accounts]]
name = "Inbox"
email = "acct@example.com"
credentials_file = "credentials.json"
token_file = "token.json"
""".strip()
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("INBOX_VAULT_CONFIG", raising=False)

    cfg = load_config(None)
    assert cfg.accounts[0].name == "Inbox"


def test_load_config_validation_errors(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "missing.toml"))

    cfg_path = tmp_path / "invalid.toml"
    cfg_path.write_text("[database]\npath='x.db'")
    with pytest.raises(ValueError, match=r"No \[\[accounts\]\] configured"):
        load_config(str(cfg_path))


def test_load_config_parses_bool_and_validates_duplicates(tmp_path: Path):
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        """
[[accounts]]
email = "acct@example.com"
credentials_file = "credentials.json"
token_file = "token.json"

[llm]
enabled = "false"

[redaction]
mode = "hybrid"
profile = "restricted"
instruction = "Mask people names as well"
chunk_chars = 900
model = "redactor-v1"
""".strip()
    )

    cfg = load_config(str(cfg_path))
    assert cfg.llm.enabled is False
    assert cfg.redaction.mode == "hybrid"
    assert cfg.redaction.profile == "restricted"
    assert cfg.redaction.model == "redactor-v1"

    dup_path = tmp_path / "dup.toml"
    dup_path.write_text(
        """
[[accounts]]
email = "acct@example.com"
credentials_file = "a.json"
token_file = "a-token.json"

[[accounts]]
email = "ACCT@example.com"
credentials_file = "b.json"
token_file = "b-token.json"
""".strip()
    )

    with pytest.raises(ValueError, match="Duplicate account email"):
        load_config(str(dup_path))


def test_load_config_rejects_invalid_redaction_mode(tmp_path: Path):
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        """
[[accounts]]
email = "acct@example.com"
credentials_file = "credentials.json"
token_file = "token.json"

[redaction]
mode = "llm-only"
""".strip()
    )

    with pytest.raises(ValueError, match="redaction.mode"):
        load_config(str(cfg_path))


def test_load_config_retrieval_and_rerank_sections(tmp_path: Path):
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        """
[[accounts]]
email = "acct@example.com"
credentials_file = "credentials.json"
token_file = "token.json"

[retrieval]
search_strategy = "lexical"
vector_backend = "lancedb"
lexical_backend = "fts5"
lancedb_path = "data/custom_lancedb"
lancedb_table = "emails"
rrf_k = 75
chunk_chars = 1200
chunk_overlap_chars = 200

[rerank]
enabled = true
model = "cross-encoder/test"
top_n = 15
""".strip()
    )

    cfg = load_config(str(cfg_path))
    assert cfg.retrieval.search_strategy == "lexical"
    assert cfg.retrieval.vector_backend == "lancedb"
    assert cfg.retrieval.lancedb_table == "emails"
    assert cfg.retrieval.rrf_k == 75
    assert cfg.retrieval.chunk_chars == 1200
    assert cfg.retrieval.chunk_overlap_chars == 200
    assert cfg.rerank.enabled is True
    assert cfg.rerank.top_n == 15


def test_load_config_rejects_invalid_chunk_overlap(tmp_path: Path):
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        """
[[accounts]]
email = "acct@example.com"
credentials_file = "credentials.json"
token_file = "token.json"

[retrieval]
chunk_chars = 500
chunk_overlap_chars = 500
""".strip()
    )

    with pytest.raises(ValueError, match="chunk_overlap_chars"):
        load_config(str(cfg_path))


def test_load_config_indexing_section(tmp_path: Path):
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        """
[[accounts]]
email = "acct@example.com"
credentials_file = "credentials.json"
token_file = "token.json"

[indexing]
include_labels = ["INBOX", "IMPORTANT"]
exclude_labels = "SPAM,TRASH"
strip_zero_width = false
collapse_whitespace = true
max_index_chars = 5000
auto_index_after_ingest = true
auto_index_pending_only = false
auto_index_limit = 250
""".strip()
    )

    cfg = load_config(str(cfg_path))
    assert cfg.indexing.include_labels == ["INBOX", "IMPORTANT"]
    assert cfg.indexing.exclude_labels == ["SPAM", "TRASH"]
    assert cfg.indexing.strip_zero_width is False
    assert cfg.indexing.collapse_whitespace is True
    assert cfg.indexing.max_index_chars == 5000
    assert cfg.indexing.auto_index_after_ingest is True
    assert cfg.indexing.auto_index_pending_only is False
    assert cfg.indexing.auto_index_limit == 250


def test_load_config_rejects_invalid_auto_index_limit(tmp_path: Path):
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        """
[[accounts]]
email = "acct@example.com"
credentials_file = "credentials.json"
token_file = "token.json"

[indexing]
auto_index_limit = 0
""".strip()
    )

    with pytest.raises(ValueError, match="indexing.auto_index_limit"):
        load_config(str(cfg_path))


def test_load_config_profiles_context_caps(tmp_path: Path):
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        """
[[accounts]]
email = "acct@example.com"
credentials_file = "credentials.json"
token_file = "token.json"

[profiles]
deep_context_max_threads = 3
deep_context_max_messages = 18
deep_context_max_chars = 6400
deep_prompt_budget_chars = 4200
gog_history_enabled = true
gog_history_account = "acct@example.com"
gog_history_command = "gog"
gog_history_max_messages = 5
gog_history_timeout_seconds = 9
""".strip()
    )

    cfg = load_config(str(cfg_path))
    assert cfg.profiles.deep_context_max_threads == 3
    assert cfg.profiles.deep_context_max_messages == 18
    assert cfg.profiles.deep_context_max_chars == 6400
    assert cfg.profiles.deep_prompt_budget_chars == 4200
    assert cfg.profiles.gog_history_enabled is True
    assert cfg.profiles.gog_history_account == "acct@example.com"
    assert cfg.profiles.gog_history_command == "gog"
    assert cfg.profiles.gog_history_max_messages == 5
    assert cfg.profiles.gog_history_timeout_seconds == 9


def test_resolve_password(monkeypatch: pytest.MonkeyPatch):
    class Obj:
        password_env = "UNIT_DB_PASSWORD"

    with pytest.raises(RuntimeError, match="UNIT_DB_PASSWORD"):
        resolve_password(Obj())

    monkeypatch.setenv("UNIT_DB_PASSWORD", "   ")
    with pytest.raises(RuntimeError, match="UNIT_DB_PASSWORD"):
        resolve_password(Obj())

    monkeypatch.setenv("UNIT_DB_PASSWORD", "s3cret")
    assert resolve_password(Obj()) == "s3cret"


def test_load_config_rejects_negative_gmail_idle_backfill_limit(tmp_path: Path):
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        """
[[accounts]]
email = "acct@example.com"
credentials_file = "credentials.json"
token_file = "token.json"

[gmail]
idle_backfill_limit = -1
""".strip()
    )

    with pytest.raises(ValueError, match="gmail.idle_backfill_limit"):
        load_config(str(cfg_path))


def test_load_config_parses_gmail_idle_backfill_limit(tmp_path: Path):
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        """
[[accounts]]
email = "acct@example.com"
credentials_file = "credentials.json"
token_file = "token.json"

[gmail]
query = "label:inbox"
idle_backfill_limit = 10
idle_backfill_query = "label:inbox OR label:sent"
""".strip()
    )

    cfg = load_config(str(cfg_path))
    assert cfg.gmail_idle_backfill_limit == 10
    assert cfg.gmail_idle_backfill_query == "label:inbox OR label:sent"


def test_load_config_rejects_empty_gmail_idle_backfill_query(tmp_path: Path):
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        """
[[accounts]]
email = "acct@example.com"
credentials_file = "credentials.json"
token_file = "token.json"

[gmail]
query = "label:inbox"
idle_backfill_query = "   "
""".strip()
    )

    with pytest.raises(ValueError, match="gmail.idle_backfill_query"):
        load_config(str(cfg_path))


def test_load_config_parses_gmail_timeout_and_progress_controls(tmp_path: Path):
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        """
[[accounts]]
email = "acct@example.com"
credentials_file = "credentials.json"
token_file = "token.json"

[gmail]
request_timeout_seconds = 42
progress_every = 25
""".strip()
    )

    cfg = load_config(str(cfg_path))
    assert cfg.gmail_request_timeout_seconds == 42.0
    assert cfg.gmail_progress_every == 25


def test_load_config_rejects_invalid_gmail_timeout_and_progress(tmp_path: Path):
    bad_timeout = tmp_path / "cfg-timeout.toml"
    bad_timeout.write_text(
        """
[[accounts]]
email = "acct@example.com"
credentials_file = "credentials.json"
token_file = "token.json"

[gmail]
request_timeout_seconds = 0
""".strip()
    )
    with pytest.raises(ValueError, match="gmail.request_timeout_seconds"):
        load_config(str(bad_timeout))

    bad_progress = tmp_path / "cfg-progress.toml"
    bad_progress.write_text(
        """
[[accounts]]
email = "acct@example.com"
credentials_file = "credentials.json"
token_file = "token.json"

[gmail]
progress_every = 0
""".strip()
    )
    with pytest.raises(ValueError, match="gmail.progress_every"):
        load_config(str(bad_progress))


def test_load_config_keeps_multi_account_order(tmp_path: Path):
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        """
[[accounts]]
name = "primary"
email = "operator@example.com"
credentials_file = "env/local/credentials.json"
token_file = "env/local/token_main.json"

[[accounts]]
name = "alt"
email = "operator+alt@example.com"
credentials_file = "env/local/credentials.json"
token_file = "env/local/token_alt.json"
""".strip()
    )

    cfg = load_config(str(cfg_path))
    assert [acct.email for acct in cfg.accounts] == [
        "operator@example.com",
        "operator+alt@example.com",
    ]
    assert cfg.accounts[1].token_file == "env/local/token_alt.json"
