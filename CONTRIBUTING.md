# Contributing

Thanks for contributing to `inbox-vault`.

This project handles private email data, so contributions should prioritize correctness, privacy, and reviewability.

## Quick setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Optional extras:

```bash
python -m pip install -e .[redaction]
python -m pip install -e .[retrieval]
```

Use `config.example.toml` as your starting point for local config.
Keep runtime artifacts and local secrets under local-only paths (never in tracked repo files).

## Before opening a PR

Run baseline checks:

```bash
ruff check .
pytest
```

If your change touches retrieval, enrichment, or sync flow, include targeted tests or a short manual validation note.

## Commit style

- Keep commits small and focused.
- Use clear, scoped messages when possible (for example: `docs: clarify local runtime paths`).
- Separate refactors from behavior changes.
- Update docs when behavior or operator workflow changes.

## Privacy expectations

- Never commit credentials, OAuth tokens, mailbox contents, or unredacted exports.
- Do not include real personal data in tests, fixtures, screenshots, or issue text.
- Keep redaction defaults intact unless a change explicitly improves privacy handling.

## Change scope guidance

- For larger design or workflow changes, open an issue first to align on scope.
- Prefer incremental PRs over broad rewrites.
- If unsure whether data handling is safe, ask before merging.
