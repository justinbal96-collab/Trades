#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env.ntfy"

if [[ -f "$ENV_FILE" ]]; then
  echo ".env.ntfy already exists: $ENV_FILE"
else
  stamp="$(date +%m%d)"
  topic="justin-nq-alerts-${stamp}-$RANDOM"
  cat >"$ENV_FILE" <<EOF
NTFY_SERVER=https://ntfy.sh
NTFY_TOPIC=$topic
NTFY_TITLE="NQ Trade Alert"
NTFY_PRIORITY=4
NTFY_TAGS=chart_with_upwards_trend,rotating_light
NQ_DASHBOARD_API_URL=http://localhost:8080/api/dashboard
NQ_ALERT_POLL_SEC=5
NQ_REQUIRE_ELIGIBLE=true
NQ_SEND_EVERY_DIRECTIONAL=false
EOF
  echo "Created $ENV_FILE"
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

echo ""
echo "Subscribe in ntfy app to topic:"
echo "  $NTFY_TOPIC"
echo ""
echo "Phone test command:"
echo "  curl -H \"Title: NQ Test\" -d \"ntfy connected\" ${NTFY_SERVER%/}/$NTFY_TOPIC"
echo ""
echo "Then start alerts with:"
echo "  bash scripts/start_ntfy_alerts.sh"
