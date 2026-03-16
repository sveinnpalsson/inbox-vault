#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BEGIN_MARKER="# >>> inbox-vault managed cron block >>>"
END_MARKER="# <<< inbox-vault managed cron block <<<"

CRONTAB_CMD="${CRONTAB_CMD:-crontab}"
HOME_DIR="${HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)}"
SECRETS_FILE="${SECRETS_FILE:-$HOME_DIR/.config/inbox-vault/secrets.env}"
CRON_LOG_FILE="${CRON_LOG_FILE:-$REPO_DIR/logs/cron.log}"
SYNC_SCRIPT="${SYNC_SCRIPT:-$REPO_DIR/scripts/run_inbox_sync_once.sh}"
WEEKLY_SCRIPT="${WEEKLY_SCRIPT:-$REPO_DIR/scripts/run_build_profiles_weekly_once.sh}"

usage() {
  cat <<'EOF'
Usage:
  scripts/cron_helper.sh [--print-only]
  scripts/cron_helper.sh --install

Options:
  --print-only   Print managed cron block to stdout (default)
  --install      Merge managed cron block into existing crontab
  -h, --help     Show this help message

Notes:
  - No destructive overwrite by default.
  - --install removes only the managed block markers and re-adds updated entries.
  - Secrets are sourced at runtime from SECRETS_FILE (default: ~/.config/inbox-vault/secrets.env).
EOF
}

print_block() {
  cat <<EOF
$BEGIN_MARKER
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HOME=$HOME_DIR

# 15-minute inbox sync + default processing pipeline
*/15 * * * * cd "$REPO_DIR" && source "$SECRETS_FILE" && "$SYNC_SCRIPT" >> "$CRON_LOG_FILE" 2>&1

# Weekly profile refresh (Sunday 03:00)
0 3 * * 0 cd "$REPO_DIR" && source "$SECRETS_FILE" && "$WEEKLY_SCRIPT" >> "$CRON_LOG_FILE" 2>&1
$END_MARKER
EOF
}

strip_managed_block() {
  awk -v begin="$BEGIN_MARKER" -v end="$END_MARKER" '
    $0 == begin {skip=1; next}
    $0 == end {skip=0; next}
    !skip {print}
  '
}

attempt_install() {
  local existing=""
  local current_err
  current_err="$(mktemp)"

  if ! existing="$($CRONTAB_CMD -l 2>"$current_err")"; then
    local err_text
    err_text="$(cat "$current_err")"
    if grep -qiE "no crontab for|no crontab" "$current_err"; then
      existing=""
    else
      echo "Could not read current crontab via '$CRONTAB_CMD -l'." >&2
      [[ -n "$err_text" ]] && echo "$err_text" >&2
      echo >&2
      echo "Fallback: print and install manually with your user context:" >&2
      echo "  $SCRIPT_DIR/cron_helper.sh --print-only" >&2
      echo "  ( crontab -l 2>/dev/null; $SCRIPT_DIR/cron_helper.sh --print-only ) | crontab -" >&2
      rm -f "$current_err"
      return 1
    fi
  fi

  rm -f "$current_err"

  local merged_file
  merged_file="$(mktemp)"

  {
    printf "%s\n" "$existing" | strip_managed_block
    [[ -n "$existing" ]] && printf "\n"
    print_block
  } > "$merged_file"

  if "$CRONTAB_CMD" "$merged_file"; then
    echo "Installed managed inbox-vault cron block successfully."
    echo "Verify with: $CRONTAB_CMD -l | sed -n '/$BEGIN_MARKER/,/$END_MARKER/p'"
    rm -f "$merged_file"
    return 0
  fi

  echo "Failed to write merged crontab via '$CRONTAB_CMD'." >&2
  echo "Fallback: inspect generated block and install manually:" >&2
  echo "  $SCRIPT_DIR/cron_helper.sh --print-only" >&2
  echo "  ( crontab -l 2>/dev/null; $SCRIPT_DIR/cron_helper.sh --print-only ) | crontab -" >&2
  rm -f "$merged_file"
  return 1
}

mode="print"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --print-only)
      mode="print"
      shift
      ;;
    --install)
      mode="install"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$mode" == "install" ]]; then
  attempt_install
else
  print_block
fi
