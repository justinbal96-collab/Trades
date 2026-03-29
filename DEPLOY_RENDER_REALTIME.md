# Render Realtime Alerts (API + Worker)

This gets you:
- 24/7 live API at `/api/dashboard`
- 24/7 free phone push alerts via ntfy
- GitHub Pages frontend reading live data

## 0) Before you start

1. Install the `ntfy` phone app and subscribe to a topic name (example: `justin-nq-alerts-347`).
2. Confirm these files are in your repo root:
   - `server.py`
   - `scripts/sms_trade_notifier.py`
   - `requirements.txt`
   - `render.yaml`

## 1) Deploy the API service

1. Go to Render dashboard.
2. Click `New +` -> `Blueprint`.
3. Connect/select your GitHub repo.
4. Select branch (usually `main`).
5. Render reads `render.yaml` and shows 2 services.
6. Create the blueprint.
7. Wait for `nq-dashboard-api` to show `Live`.
8. Open:
   - `https://<your-api-service>.onrender.com/api/dashboard`
9. Verify you see JSON payload (not 404).

## 2) Configure the alert worker

1. Open service `nq-trade-alert-worker`.
2. Go to `Environment`.
3. Set these values:
   - `NQ_DASHBOARD_API_URL` = `https://<your-api-service>.onrender.com/api/dashboard`
   - `NQ_ALERT_CHANNEL` = `ntfy`
   - `NTFY_SERVER` = `https://ntfy.sh`
   - `NTFY_TOPIC` = your topic from phone app (example: `justin-nq-alerts-347`)
4. Optional:
   - `NTFY_TITLE` = `NQ Trade Alert`
   - `NTFY_PRIORITY` = `4`
   - `NTFY_TAGS` = `chart_with_upwards_trend,rotating_light`
   - `NTFY_TOKEN` = token only if topic is protected/self-hosted
5. Save changes and redeploy worker.

## 3) Confirm alerts are running

In worker logs, look for:
- `[SKIP]` lines on no new setup
- `[SENT] ntfy id=...` when a trade setup is sent

## 4) Point GitHub Pages frontend to live API

Edit `site-config.js` in your GitHub repo:

```js
window.__NQ_DASHBOARD_API_URL__ = "https://<your-api-service>.onrender.com/api/dashboard";
```

Push, wait for Pages deploy, then hard refresh browser.

## 5) Realtime expectations

- Strategy runs on 5-minute bars.
- Polling is every 5s (`NQ_ALERT_POLL_SEC=5`), so alerts arrive shortly after bar close when criteria trigger.
- For strict 24/7 availability, use a non-sleeping plan for both web and worker.
