from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_console_script() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["inbox-vault"] == "inbox_vault.cli:main"
    assert data["tool"]["setuptools"]["package-dir"] == {"": "src"}
    assert data["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]


def test_llm_vault_bridge_doc_covers_contract() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    bridge = (REPO_ROOT / "docs" / "llm-vault-bridge.md").read_text(encoding="utf-8")

    assert "`llm-vault`" in bridge
    assert "[mail_bridge]" in bridge
    assert "INBOX_VAULT_DB_PASSWORD" in bridge
    assert "read-only" in bridge
    assert "does **not** currently ship its own OpenClaw plugin/tool surface" in bridge
    assert "canonical unified skill currently lives in the `llm-vault` repo" in readme


def test_editable_install_creates_console_script(tmp_path: Path) -> None:
    venv_dir = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
        check=True,
    )

    bin_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    python_bin = bin_dir / ("python.exe" if os.name == "nt" else "python")
    script_name = "inbox-vault.exe" if os.name == "nt" else "inbox-vault"
    script_path = bin_dir / script_name

    subprocess.run(
        [
            str(python_bin),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "-e",
            str(REPO_ROOT),
        ],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert script_path.exists(), f"editable install did not create {script_name}"

    help_result = subprocess.run(
        [str(script_path), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "privacy-safe defaults" in help_result.stdout
    assert "update" in help_result.stdout
