from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "cron_helper.sh"
BEGIN_MARKER = "# >>> inbox-vault managed cron block >>>"


def test_cron_helper_print_block_contains_expected_entries() -> None:
    result = subprocess.run(
        [str(SCRIPT), "--print-only"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    output = result.stdout
    assert BEGIN_MARKER in output
    assert "SHELL=/bin/bash" in output
    assert "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" in output
    assert "*/15 * * * *" in output
    assert "0 3 * * 0" in output
    assert "run_inbox_sync_once.sh" in output
    assert "run_build_profiles_weekly_once.sh" in output


def test_cron_helper_install_merges_without_duplicate_managed_block(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    state_dir = tmp_path / "state"
    fake_bin.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    current = state_dir / "current"
    current.write_text(
        "\n".join(
            [
                "MAILTO=alerts@example.com",
                BEGIN_MARKER,
                "0 0 * * * echo old-entry",
                "# <<< inbox-vault managed cron block <<<",
                "30 2 * * * /usr/local/bin/backup",
                "",
            ]
        ),
        encoding="utf-8",
    )

    fake_crontab = fake_bin / "crontab"
    fake_crontab.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
STATE_DIR=\"$FAKE_CRON_STATE_DIR\"
CURRENT=\"$STATE_DIR/current\"

if [[ \"${1:-}\" == \"-l\" ]]; then
  cat \"$CURRENT\"
  exit 0
fi

cat \"$1\" > \"$CURRENT\"
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    env = os.environ.copy()
    env["CRONTAB_CMD"] = str(fake_crontab)
    env["FAKE_CRON_STATE_DIR"] = str(state_dir)

    subprocess.run(
        [str(SCRIPT), "--install"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    merged = current.read_text(encoding="utf-8")
    assert merged.count(BEGIN_MARKER) == 1
    assert "30 2 * * * /usr/local/bin/backup" in merged
    assert "run_inbox_sync_once.sh" in merged
    assert "run_build_profiles_weekly_once.sh" in merged


def test_cron_helper_install_permission_fallback_message(tmp_path: Path) -> None:
    fake_crontab = tmp_path / "crontab"
    fake_crontab.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ \"${1:-}\" == \"-l\" ]]; then
  echo \"crontab: permission denied\" >&2
  exit 1
fi
exit 1
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    env = os.environ.copy()
    env["CRONTAB_CMD"] = str(fake_crontab)

    result = subprocess.run(
        [str(SCRIPT), "--install"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "Fallback" in result.stderr
    assert "--print-only" in result.stderr
    assert "permission denied" in result.stderr
