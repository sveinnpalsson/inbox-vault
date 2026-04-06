---
name: inbox-vault-local
description: Run and sanity-check inbox-vault locally from a repo checkout (no publishing). Use when asked to run sync/index/profile/status commands, execute local quality checks (ruff/pytest), or verify release-readiness changes safely on one machine.
---

# inbox-vault local operator skill

Use this skill to run the project in local-only mode with consistent config handling.

## Quick usage

From repo root, use:

```bash
skills/inbox-vault-local/scripts/iv-cli.sh <command> [args...]
```

Config resolution order:
1. `--config <path>` (optional override)
2. `INBOX_VAULT_CONFIG` env var
3. `config.toml` in repo root

Examples:

```bash
skills/inbox-vault-local/scripts/iv-cli.sh status --json
skills/inbox-vault-local/scripts/iv-cli.sh --config config.toml latest --limit 5
skills/inbox-vault-local/scripts/iv-cli.sh update --index-vectors --index-pending-only

# date-scoped retrieval (UTC window [from-date, to-date))
skills/inbox-vault-local/scripts/iv-cli.sh search "budget approvals" \
  --from-date 2026-03-01 --to-date 2026-03-31 --clearance redacted

# latest raw vs redacted previews in a date window
skills/inbox-vault-local/scripts/iv-cli.sh latest --limit 5 --clearance full --from-date 2026-03-01 --to-date 2026-03-31 --json
skills/inbox-vault-local/scripts/iv-cli.sh latest --limit 5 --clearance redacted --from-date 2026-03-01 --to-date 2026-03-31 --json
```

Date-window semantics are explicit and consistent across retrieval commands:
- `--from-date`: inclusive lower bound (UTC)
- `--to-date`: exclusive upper bound (UTC)
- date-only `--to-date YYYY-MM-DD` means next-day midnight UTC (inclusive calendar-day behavior)

The wrapper keeps day-to-day usage simple, but you can pass any advanced `inbox-vault` subcommand/flags through it.

## Local checks before completion

- `command -v inbox-vault && inbox-vault --help`
- `ruff check .`
- targeted `pytest` first, then broader subsets as needed.

## References

- For release-pass checks, read `references/local-release-checklist.md`.
- For using Inbox Vault as the mail source behind `llm-vault`, read `../../docs/llm-vault-bridge.md`.
- If you mirror the unified cross-vault skill locally, treat the copy in `llm-vault` as canonical.
