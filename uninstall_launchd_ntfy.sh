#!/usr/bin/env bash
set -euo pipefail

UID_NUM="$(id -u)"
AGENT_DIR="$HOME/Library/LaunchAgents"

API_LABEL="com.justin.nq.api"
NOTIFIER_LABEL="com.justin.nq.ntfy.notifier"

API_PLIST="$AGENT_DIR/${API_LABEL}.plist"
NOTIFIER_PLIST="$AGENT_DIR/${NOTIFIER_LABEL}.plist"

launchctl bootout "gui/$UID_NUM/$API_LABEL" >/dev/null 2>&1 || true
launchctl bootout "gui/$UID_NUM/$NOTIFIER_LABEL" >/dev/null 2>&1 || true

rm -f "$API_PLIST" "$NOTIFIER_PLIST"

echo "Uninstalled launch agents:"
echo "  $API_LABEL"
echo "  $NOTIFIER_LABEL"
