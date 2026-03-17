from pathlib import Path


def test_gitignore_covers_runtime_artifacts() -> None:
    content = Path(".gitignore").read_text(encoding="utf-8")
    for expected in [
        ".runs/",
        ".stress-runs/",
        "logs/",
        "tmp/",
        "env/",
        "data/",
    ]:
        assert expected in content


def test_precommit_has_sensitive_guard_hook() -> None:
    content = Path(".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "sensitive-artifact-guard" in content
    assert "scripts/precommit_sensitive_guard.py" in content
    assert "detect-private-key" in content


def test_sensitive_guard_does_not_allowlist_account_configs() -> None:
    content = Path("scripts/precommit_sensitive_guard.py").read_text(encoding="utf-8")
    assert '"config.example.toml"' in content
    assert '"config.account-main.toml"' not in content
    assert '"config.multi-account.example.toml"' in content
    assert '"config.live20.toml"' not in content
