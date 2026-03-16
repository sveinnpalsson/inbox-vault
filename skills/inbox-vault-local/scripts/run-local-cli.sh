#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible shim.
# Prefer: skills/inbox-vault-local/scripts/iv-cli.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/iv-cli.sh" "$@"
