#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env.ntfy"

cd "$ROOT_DIR"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export PYTHONUNBUFFERED=1

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
export NQ_ALERT_FORMAT="${NQ_ALERT_FORMAT:-compact}"

if [[ -z "$NTFY_TOPIC" ]]; then
  echo "Missing NTFY_TOPIC in $ENV_FILE. Run: bash scripts/setup_ntfy_local.sh"
  exit 1
fi

exec python3 -u scripts/sms_trade_notifier.py \
  --api-url "$NQ_DASHBOARD_API_URL" \
  --poll-sec "$NQ_ALERT_POLL_SEC" \
  --channel ntfy \
  --format "$NQ_ALERT_FORMAT"
