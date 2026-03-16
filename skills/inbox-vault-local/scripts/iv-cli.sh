#!/usr/bin/env bash
set -euo pipefail

# Simple local wrapper for inbox-vault.
# Usage:
#   iv-cli.sh [--config <path>] <command> [args...]
# Examples:
#   iv-cli.sh status --json
#   iv-cli.sh --config config.toml update --index-vectors

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 [--config <path>] <command> [args...]" >&2
  exit 2
fi

CONFIG_PATH="${INBOX_VAULT_CONFIG:-$REPO_ROOT/config.toml}"
if [[ "${1:-}" == "--config" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "error: --config requires a path" >&2
    exit 2
  fi
  CONFIG_PATH="$2"
  shift 2
fi

if [[ $# -eq 0 ]]; then
  echo "usage: $0 [--config <path>] <command> [args...]" >&2
  exit 2
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "error: config file not found: $CONFIG_PATH" >&2
  echo "hint: set INBOX_VAULT_CONFIG or pass --config /path/to/config.toml" >&2
  exit 1
fi

if command -v inbox-vault >/dev/null 2>&1; then
  exec inbox-vault --config "$CONFIG_PATH" "$@"
fi

PYTHON_BIN="python3"
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m inbox_vault.cli --config "$CONFIG_PATH" "$@"
