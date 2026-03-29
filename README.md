# Eigenstate-Inspired NQ Dashboard (React + Live Quant API)

This app now uses:
- **Bundled local React app** (no CDN imports at runtime)
- **Routed product surface**: `#/`, `#/signals`, `#/risk`
- A local Python API (`/api/dashboard`) that pulls **live NQ futures** (`NQ=F`) bars
- Cost-adjusted backtest metrics + CVaR sleeve sizing
- GitHub Pages snapshot fallback via `dashboard-fallback.json` when no live backend URL is set

## Run

```bash
cd /Users/justin/Downloads/skalz/eigenstate-inspired-dashboard
npm run build
python3 server.py
```

Then open: [http://localhost:8080](http://localhost:8080)

For GitHub upload/deploy, use:
- `DEPLOY_GITHUB.md`

For realtime Render API + alert worker deploy, use:
- `DEPLOY_RENDER_REALTIME.md`

## Data Source

- Yahoo Finance chart APIs (`query1.finance.yahoo.com`)
- 5-minute NQ futures bars over the last 20 days (default `BACKTEST_RANGE_5M=20d`)

## Notes

- API payload is cached in 30-second buckets to reduce rate pressure.
- Frontend refreshes data every 30 seconds and supports manual refresh from the top bar.
- Frontend API target is configurable in `site-config.js` via `window.__NQ_DASHBOARD_API_URL__`.
- Trade entries are persistently journaled to:
  - `trade_logs/nq_trade_journal.jsonl`
  - `trade_logs/nq_trade_journal.csv`
  This log grows over time and is shown in `Signal Studio`.

## SMS Alerts (Entry + TP + SL)

You can text yourself actionable setups from the live `execution_plan`:

```bash
cd /Users/justin/Downloads/skalz/eigenstate-inspired-dashboard
python3 scripts/sms_trade_notifier.py --once --dry-run
```

For real SMS via Twilio:

```bash
export TWILIO_ACCOUNT_SID="ACxxxxxxxx"
export TWILIO_AUTH_TOKEN="xxxxxxxx"
export TWILIO_FROM_NUMBER="+1xxxxxxxxxx"
export TWILIO_TO_NUMBER="+1xxxxxxxxxx"
python3 scripts/sms_trade_notifier.py
```

Optional flags:

- `--once`: one poll and exit
- `--dry-run`: print message without sending
- `--poll-sec 20`: polling interval
- `--allow-ineligible`: alert even if prop checks fail
- `--send-every-directional`: send each BUY/SELL poll instead of only new setup changes

## WhatsApp Alerts (Twilio)

The same notifier supports WhatsApp:

```bash
cd /Users/justin/Downloads/skalz/eigenstate-inspired-dashboard
export TWILIO_ACCOUNT_SID="ACxxxxxxxx"
export TWILIO_AUTH_TOKEN="xxxxxxxx"
export TWILIO_TO_NUMBER="+13473587624"
export TWILIO_WHATSAPP_FROM="+14155238886"  # Twilio WhatsApp sandbox or approved WA sender
python3 scripts/sms_trade_notifier.py --channel whatsapp --once --dry-run
python3 scripts/sms_trade_notifier.py --channel whatsapp
```

Notes:
- Twilio trial accounts require your destination WhatsApp number to be joined/verified in the Twilio WhatsApp sandbox first.
- You can also use `TWILIO_FROM_NUMBER` as fallback for WhatsApp if it is your approved WhatsApp sender.
- Alert timestamps are formatted as NYC time in the message body.

## Free Phone Push Alerts (ntfy)

You can send free real-time push alerts to your phone without Twilio:

```bash
cd /Users/justin/Downloads/skalz/eigenstate-inspired-dashboard
export NQ_ALERT_CHANNEL="ntfy"
export NTFY_SERVER="https://ntfy.sh"
export NTFY_TOPIC="your-unique-topic-name"
python3 scripts/sms_trade_notifier.py --once --dry-run
python3 scripts/sms_trade_notifier.py --poll-sec 5
```

Optional ntfy env vars:
- `NTFY_TITLE="NQ Trade Alert"`
- `NTFY_PRIORITY="4"` (1-5)
- `NTFY_TAGS="chart_with_upwards_trend"`
- `NTFY_TOKEN="<bearer-token>"` for protected topics
- `NQ_ALERT_FORMAT="compact"` (default) or `"full"`

Quick local setup (auto topic + one-command start):

```bash
cd /Users/justin/Downloads/skalz/eigenstate-inspired-dashboard
bash scripts/setup_ntfy_local.sh
bash scripts/start_ntfy_alerts.sh
```

Stop:

```bash
bash scripts/stop_ntfy_alerts.sh
```

Run automatically at login (works even when Codex is closed):

```bash
cd /Users/justin/Downloads/skalz/eigenstate-inspired-dashboard
bash scripts/install_launchd_ntfy.sh
bash scripts/status_launchd_ntfy.sh
```

Remove auto-run:

```bash
bash scripts/uninstall_launchd_ntfy.sh
```
