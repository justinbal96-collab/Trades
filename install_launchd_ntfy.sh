#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_WORKSPACE="$(cd "$SOURCE_ROOT/.." && pwd)"
RUNTIME_PARENT="${NQ_RUNTIME_PARENT:-$HOME/nq-runtime}"
ROOT_DIR="$RUNTIME_PARENT/eigenstate-inspired-dashboard"
ENV_FILE="$ROOT_DIR/.env.ntfy"
AGENT_DIR="$HOME/Library/LaunchAgents"
UID_NUM="$(id -u)"

API_LABEL="com.justin.nq.api"
NOTIFIER_LABEL="com.justin.nq.ntfy.notifier"

API_PLIST="$AGENT_DIR/${API_LABEL}.plist"
NOTIFIER_PLIST="$AGENT_DIR/${NOTIFIER_LABEL}.plist"
ENV_FILE_ESCAPED="$ROOT_DIR/.env.ntfy"

API_CMD="cd \"$ROOT_DIR\" && export PATH=\"/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:\$PATH\" && export PYTHONUNBUFFERED=1 && exec python3 -u server.py"
NOTIFIER_CMD="cd \"$ROOT_DIR\" && export PATH=\"/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:\$PATH\" && export PYTHONUNBUFFERED=1 && if [ -f \"$ENV_FILE_ESCAPED\" ]; then set -a; source \"$ENV_FILE_ESCAPED\"; set +a; fi && export NQ_ALERT_CHANNEL=ntfy && export NQ_DASHBOARD_API_URL=\"\${NQ_DASHBOARD_API_URL:-http://localhost:8080/api/dashboard}\" && export NQ_ALERT_POLL_SEC=\"\${NQ_ALERT_POLL_SEC:-5}\" && export NQ_ALERT_FORMAT=\"\${NQ_ALERT_FORMAT:-compact}\" && exec python3 -u scripts/sms_trade_notifier.py --api-url \"\${NQ_DASHBOARD_API_URL}\" --poll-sec \"\${NQ_ALERT_POLL_SEC}\" --channel ntfy --format \"\${NQ_ALERT_FORMAT}\""

sync_dir() {
  local src="$1"
  local dst="$2"
  mkdir -p "$dst"
  rsync -a --delete \
    --exclude '.DS_Store' \
    --exclude 'node_modules' \
    --exclude 'snapshots' \
    "$src/" "$dst/"
}

mkdir -p "$AGENT_DIR" "$RUNTIME_PARENT"

echo "Syncing runtime workspace to: $RUNTIME_PARENT"
sync_dir "$SOURCE_ROOT" "$ROOT_DIR"
if [[ -d "$SOURCE_WORKSPACE/quantitative-portfolio-optimization" ]]; then
  sync_dir "$SOURCE_WORKSPACE/quantitative-portfolio-optimization" "$RUNTIME_PARENT/quantitative-portfolio-optimization"
fi
if [[ -d "$SOURCE_WORKSPACE/nvidia-kx-samples" ]]; then
  sync_dir "$SOURCE_WORKSPACE/nvidia-kx-samples" "$RUNTIME_PARENT/nvidia-kx-samples"
fi

if [[ ! -f "$ENV_FILE" ]]; then
  if [[ -f "$SOURCE_ROOT/.env.ntfy" ]]; then
    cp -f "$SOURCE_ROOT/.env.ntfy" "$ENV_FILE"
  else
    echo "No .env.ntfy found. Creating one now..."
    bash "$ROOT_DIR/scripts/setup_ntfy_local.sh"
  fi
fi

cat >"$API_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$API_LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>-lc</string>
    <string>$API_CMD</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ThrottleInterval</key>
  <integer>10</integer>
  <key>StandardOutPath</key>
  <string>/tmp/nq_api_launchd.out.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/nq_api_launchd.err.log</string>
</dict>
</plist>
EOF

cat >"$NOTIFIER_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$NOTIFIER_LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>-lc</string>
    <string>$NOTIFIER_CMD</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ThrottleInterval</key>
  <integer>10</integer>
  <key>StandardOutPath</key>
  <string>/tmp/nq_ntfy_launchd.out.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/nq_ntfy_launchd.err.log</string>
</dict>
</plist>
EOF

# Unload existing services if present.
launchctl bootout "gui/$UID_NUM/$API_LABEL" >/dev/null 2>&1 || true
launchctl bootout "gui/$UID_NUM/$NOTIFIER_LABEL" >/dev/null 2>&1 || true

# Load fresh services.
launchctl bootstrap "gui/$UID_NUM" "$API_PLIST"
launchctl bootstrap "gui/$UID_NUM" "$NOTIFIER_PLIST"

# Force immediate start.
launchctl kickstart -k "gui/$UID_NUM/$API_LABEL"
launchctl kickstart -k "gui/$UID_NUM/$NOTIFIER_LABEL"

echo "Installed launch agents:"
echo "  $API_LABEL"
echo "  $NOTIFIER_LABEL"
echo "Runtime root: $ROOT_DIR"
echo ""
echo "Logs:"
echo "  /tmp/nq_api_launchd.out.log"
echo "  /tmp/nq_api_launchd.err.log"
echo "  /tmp/nq_ntfy_launchd.out.log"
echo "  /tmp/nq_ntfy_launchd.err.log"
echo ""
echo "Check status:"
echo "  bash scripts/status_launchd_ntfy.sh"
