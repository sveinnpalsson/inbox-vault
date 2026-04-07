# Inbox Vault as an `llm-vault` mail source

Use Inbox Vault when you want Gmail sync, encrypted local mail storage, and mail-side enrichment to stay separate from `llm-vault`'s docs/photos/mail retrieval layer.

## Boundary

- **Inbox Vault** owns Gmail auth, sync, repair, local enrichment, and the encrypted mail database.
- **`llm-vault`** reads from Inbox Vault through a **read-only mail bridge**.
- Inbox Vault does **not** currently ship its own OpenClaw plugin/tool surface for autonomous agents. For agent-facing mail retrieval, prefer the safe surface exposed by `llm-vault`.
- **`llm-vault`** is also the canonical owner of the shared redaction contract and benchmark story. Inbox Vault should keep mail-specific validation, but not a separate competing benchmark track.

## Straight-line setup

### 1. Install and configure Inbox Vault

From the Inbox Vault repo:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
cp config.example.toml config.toml
export INBOX_VAULT_DB_PASSWORD='choose-a-strong-passphrase'
```

Then add your Gmail credentials and local endpoint settings to `config.toml`.

### 2. Run a first successful mail sync

```bash
inbox-vault update --backfill 10
inbox-vault status --json
inbox-vault search "budget approval" --top-k 3
```

Before wiring the bridge, make sure Inbox Vault can already:
- authenticate to Gmail
- create/open its encrypted DB
- return local search results from the mailbox you care about

### 3. Point `llm-vault` at the Inbox Vault DB

In `llm-vault`, configure the mail bridge with the Inbox Vault DB path and password env:

```toml
[mail_bridge]
db_path = "/absolute/path/to/inbox_vault.db"
password_env = "INBOX_VAULT_DB_PASSWORD"
include_accounts = ["you@gmail.com"]
import_summary = true
```

Notes:
- `db_path` should point at the **Inbox Vault** SQLCipher DB, not an export.
- `password_env` should stay `INBOX_VAULT_DB_PASSWORD` unless you intentionally changed the env name in Inbox Vault config.
- `include_accounts` should match the accounts you actually synced in Inbox Vault.
- `import_summary = true` allows `llm-vault` to import available mail-side summaries when present.

### 4. Keep the bridge read-only

The intended contract is:
- Inbox Vault remains the source of truth for Gmail sync and mail processing.
- `llm-vault` reads from it but does not take over Gmail sync responsibilities.
- Operational mail repair/backfill stays on the Inbox Vault side.

### 5. Validate from the `llm-vault` side

Once the bridge is configured in `llm-vault`, use the normal `llm-vault` status/update flow there to confirm mail is visible through the bridge.

## Operator guidance

Use Inbox Vault directly when you need:
- Gmail auth setup
- first sync / repair / backfill
- mailbox-specific validation
- operator-clearance mail inspection

Use `llm-vault` when you need:
- unified retrieval across docs + photos + mail
- the current OpenClaw-facing safe search/tool surface
- one redacted-first retrieval layer across sources

## Unified skill note

If you mirror `vault-unified-local` into this repo for a local deployment, treat the copy in `llm-vault` as canonical and keep the mirror aligned before release.
