#!/usr/bin/env python3
"""Pre-commit guardrail for obvious secret/runtime artifact leaks.

This hook is intentionally explicit and low-complexity:
- blocks staging files under local runtime dirs (data/env/logs/.runs/.stress-runs/tmp)
- blocks known credential/token/key file types
- blocks account-specific config files that embed OAuth credential/token paths
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

FORBIDDEN_PATH_PATTERNS = [
    re.compile(r"^(?:env|data|logs|tmp)/"),
    re.compile(r"^\.runs/"),
    re.compile(r"^\.stress-runs/"),
    re.compile(r"^\.tmux/"),
]

FORBIDDEN_FILE_PATTERNS = [
    re.compile(r"(^|/)\.env(\..+)?$"),
    re.compile(r"(^|/)config\.toml$"),
    re.compile(r"(^|/)config\.local.*\.toml$"),
    re.compile(r"(^|/)config\..*\.local\.toml$"),
    re.compile(r"(^|/)credentials[^/]*\.json$", re.IGNORECASE),
    re.compile(r"(^|/)token[^/]*\.json$", re.IGNORECASE),
    re.compile(r"(^|/)id_(?:rsa|dsa|ecdsa|ed25519)$", re.IGNORECASE),
    re.compile(r"\.(?:db|sqlite|sqlite3|db-wal|db-shm)$", re.IGNORECASE),
]

CONFIG_GLOB = "config*.toml"
CONFIG_ALLOWLIST = {
    "config.example.toml",
    "config.multi-account.example.toml",
}
SENSITIVE_CONFIG_MARKERS = [
    re.compile(r"^\s*credentials_file\s*=", re.MULTILINE),
    re.compile(r"^\s*token_file\s*=", re.MULTILINE),
]



def normalize(path: str) -> str:
    return path.strip().replace("\\", "/")



def violates_path_rules(path: str) -> str | None:
    for pattern in FORBIDDEN_PATH_PATTERNS:
        if pattern.search(path):
            return "runtime/local-data path"
    for pattern in FORBIDDEN_FILE_PATTERNS:
        if pattern.search(path):
            return "sensitive filename pattern"
    return None



def is_sensitive_account_config(path: str) -> tuple[bool, str | None]:
    p = Path(path)
    if p.parent.as_posix() not in {".", ""}:
        return False, None
    if not p.match(CONFIG_GLOB):
        return False, None
    if p.name in CONFIG_ALLOWLIST:
        return False, None
    if not p.exists():
        return False, None

    content = p.read_text(encoding="utf-8", errors="ignore")
    for marker in SENSITIVE_CONFIG_MARKERS:
        if marker.search(content):
            return True, "account config contains OAuth credential/token file paths"
    return False, None



def evaluate(paths: Iterable[str]) -> list[tuple[str, str]]:
    violations: list[tuple[str, str]] = []
    for raw in paths:
        path = normalize(raw)
        if not path:
            continue

        reason = violates_path_rules(path)
        if reason:
            violations.append((path, reason))
            continue

        config_violation, config_reason = is_sensitive_account_config(path)
        if config_violation and config_reason:
            violations.append((path, config_reason))

    return violations



def main(argv: list[str]) -> int:
    violations = evaluate(argv[1:])
    if not violations:
        return 0

    print("\n[inbox-vault-sensitive-artifact-guard] Blocked commit due to sensitive artifacts:\n")
    for path, reason in violations:
        print(f" - {path}: {reason}")

    print(
        "\nIf you need a shareable config, copy from config.example.toml "
        "and keep token/credential paths local."
    )
    print("Runtime outputs belong in ignored folders (.runs/.stress-runs/logs/tmp/data/env).")
    return 1



if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
