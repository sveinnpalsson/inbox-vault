# Local release-pass checklist

Use this checklist for local-only readiness passes.

## 1) Hygiene

- Keep generated runtime data out of commits (`data/`, `logs/`, `.runs/`, `.stress-runs/`, `tmp/`).
- Avoid tracking account-specific secrets/tokens.
- Keep examples sanitized (`config.example.toml`).

## 2) Skill smoke check

- Smoke check local wrapper:
  ```bash
  skills/inbox-vault-local/scripts/iv-cli.sh --config config.toml status --json
  ```

## 3) Quality gate

- Lint:
  ```bash
  ruff check .
  ```
- Tests (fast subset first):
  ```bash
  pytest -q tests/test_config.py tests/test_cli.py tests/test_repo_hygiene.py
  ```
