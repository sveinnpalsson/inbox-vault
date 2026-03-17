import stat
from pathlib import Path


def test_gitignore_covers_runtime_artifacts() -> None:
    content = Path(".gitignore").read_text(encoding="utf-8")
    for expected in [
        ".runs/",
        ".stress-runs/",
        "build/",
        "dist/",
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


def test_setup_docs_require_python311_and_pip_upgrade() -> None:
    for path in [Path("README.md"), Path("CONTRIBUTING.md")]:
        content = path.read_text(encoding="utf-8")
        assert "python3.11 -m venv .venv" in content
        assert "python -m pip install --upgrade pip" in content


def test_shell_scripts_are_executable() -> None:
    for path in Path("scripts").glob("*.sh"):
        mode = path.stat().st_mode
        assert mode & stat.S_IXUSR, f"{path} must be executable"
