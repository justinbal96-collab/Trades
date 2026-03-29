#!/usr/bin/env bash
set -euo pipefail

UID_NUM="$(id -u)"
API_LABEL="com.justin.nq.api"
NOTIFIER_LABEL="com.justin.nq.ntfy.notifier"

filter_lines() {
  local pattern="$1"
  local file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -n "$pattern" "$file" || true
  else
    grep -nE "$pattern" "$file" || true
  fi
}

show_one() {
  local label="$1"
  echo "=== $label ==="
  if launchctl print "gui/$UID_NUM/$label" >/tmp/nq_launchd_print.txt 2>/tmp/nq_launchd_print.err; then
    filter_lines "state =|pid =|last exit code =|path =|program =|arguments =|cd \\\"" /tmp/nq_launchd_print.txt
  else
    echo "not loaded"
    sed -n '1,4p' /tmp/nq_launchd_print.err || true
  fi
  echo ""
}

show_one "$API_LABEL"
show_one "$NOTIFIER_LABEL"

echo "Recent logs:"
echo "--- /tmp/nq_api_launchd.err.log ---"
tail -n 20 /tmp/nq_api_launchd.err.log 2>/dev/null || true
echo "--- /tmp/nq_ntfy_launchd.err.log ---"
tail -n 20 /tmp/nq_ntfy_launchd.err.log 2>/dev/null || true
