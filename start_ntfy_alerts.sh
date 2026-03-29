#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env.ntfy"
NOTIFIER_PID_FILE="/tmp/nq_ntfy_notifier.pid"
API_PID_FILE="/tmp/nq_local_api.pid"

cd "$ROOT_DIR"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$ENV_FILE"
fi

export NQ_ALERT_CHANNEL="ntfy"
export NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
export NTFY_TOPIC="${NTFY_TOPIC:-}"
export NTFY_TITLE="${NTFY_TITLE:-NQ Trade Alert}"
export NTFY_PRIORITY="${NTFY_PRIORITY:-4}"
export NTFY_TAGS="${NTFY_TAGS:-chart_with_upwards_trend,rotating_light}"
export NQ_DASHBOARD_API_URL="${NQ_DASHBOARD_API_URL:-http://localhost:8080/api/dashboard}"
export NQ_ALERT_POLL_SEC="${NQ_ALERT_POLL_SEC:-5}"
export NQ_REQUIRE_ELIGIBLE="${NQ_REQUIRE_ELIGIBLE:-true}"
export NQ_SEND_EVERY_DIRECTIONAL="${NQ_SEND_EVERY_DIRECTIONAL:-false}"

if [[ -z "$NTFY_TOPIC" ]]; then
  echo "Missing NTFY_TOPIC. Run: bash scripts/setup_ntfy_local.sh"
  exit 1
fi

if [[ "$NQ_DASHBOARD_API_URL" == http://localhost:8080/* || "$NQ_DASHBOARD_API_URL" == http://127.0.0.1:8080/* ]]; then
  if ! lsof -nP -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1; then
    nohup python3 server.py >/tmp/nq_local_api.log 2>&1 &
    echo $! >"$API_PID_FILE"
    sleep 1
  fi
fi

if [[ -f "$NOTIFIER_PID_FILE" ]]; then
  old_pid="$(cat "$NOTIFIER_PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" >/dev/null 2>&1; then
    echo "Notifier already running (pid $old_pid)."
    echo "Log: /tmp/nq_ntfy_notifier.log"
    exit 0
  fi
fi

nohup python3 scripts/sms_trade_notifier.py \
  --api-url "$NQ_DASHBOARD_API_URL" \
  --poll-sec "$NQ_ALERT_POLL_SEC" \
  --channel ntfy \
  >/tmp/nq_ntfy_notifier.log 2>&1 &
echo $! >"$NOTIFIER_PID_FILE"
sleep 1

# Fire a startup test message.
curl -sS \
  -H "Title: NQ Alerts Started" \
  -H "Priority: 3" \
  -d "Notifier is live. Topic: $NTFY_TOPIC" \
  "${NTFY_SERVER%/}/$NTFY_TOPIC" >/dev/null || true

echo "Started ntfy notifier."
echo "Topic: $NTFY_TOPIC"
echo "Notifier log: /tmp/nq_ntfy_notifier.log"
echo "Tail logs: tail -f /tmp/nq_ntfy_notifier.log"
