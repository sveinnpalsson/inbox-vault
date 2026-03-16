#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${INBOX_VAULT_REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LOG_FILE="$REPO_DIR/logs/iv-sync-15m-direct.log"

cd "$REPO_DIR"
mkdir -p logs .runs

if [[ -z "${INBOX_VAULT_DB_PASSWORD:-}" && -f "$HOME/.config/inbox-vault/secrets.env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.config/inbox-vault/secrets.env"
fi

if [[ -z "${INBOX_VAULT_DB_PASSWORD:-}" ]]; then
  echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] ERROR missing INBOX_VAULT_DB_PASSWORD" | tee -a "$LOG_FILE" >&2
  exit 2
fi

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] START inbox sync" | tee -a "$LOG_FILE"
inbox-vault --config "${CONFIG:-config.toml}" update 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}
if [[ $status -ne 0 ]]; then
  echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] FAIL inbox sync exit=$status" | tee -a "$LOG_FILE" >&2
  exit "$status"
fi

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] OK inbox sync" | tee -a "$LOG_FILE"
