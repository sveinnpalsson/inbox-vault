from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from inbox_vault.config import (
    AccountConfig,
    AppConfig,
    DBConfig,
    EmbeddingConfig,
    LLMConfig,
    RedactionConfig,
    RerankConfig,
    RetrievalConfig,
)
from inbox_vault.db import get_conn


@pytest.fixture
def app_cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
        accounts=[
            AccountConfig(
                name="primary",
                email="acct@example.com",
                credentials_file="fake-credentials.json",
                token_file="fake-token.json",
            )
        ],
        llm=LLMConfig(
            enabled=True,
            endpoint="http://local-llm.invalid",
            model="test-model",
            timeout_seconds=1.0,
        ),
        db=DBConfig(path=str(tmp_path / "test.db"), password_env="TEST_DB_PASSWORD"),
        embeddings=EmbeddingConfig(
            endpoint="http://local-embedding.invalid",
            model="test-embed-model",
            timeout_seconds=1.0,
        ),
        redaction=RedactionConfig(
            mode="regex", profile="standard", instruction="", chunk_chars=600
        ),
        retrieval=RetrievalConfig(search_strategy="hybrid", lexical_backend="fts5"),
        rerank=RerankConfig(enabled=False),
        gmail_query="label:inbox OR label:sent",
    )


@pytest.fixture
def conn(tmp_path: Path):
    db_path = tmp_path / "test.db"
    c = get_conn(str(db_path), "unit-test-password")
    try:
        yield c
    finally:
        c.close()
