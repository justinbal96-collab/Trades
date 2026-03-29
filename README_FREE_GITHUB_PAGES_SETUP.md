# Free Deployment Walkthrough (GitHub Pages + GitHub Actions)

This setup gives you:
- Free website hosting on GitHub Pages
- Free scheduled strategy refreshes (every 15 minutes)
- Persistent trade history that grows over time
- No paid backend required

## What this package contains
Copy these files/folders into your repo root:
- `.github/workflows/pages.yml`
- `.github/workflows/refresh-live-data.yml`
- `scripts/build_live_snapshot.py`
- `site-config.js`
- `data/` (entire folder)
- `trade_logs/` (entire folder)
- `dashboard-fallback.json`

## Step 1) Copy files into your repo
If your repo local path is `/Users/justin/Downloads/skalz/Trades`, run:

```bash
cd /Users/justin/Downloads/skalz
mkdir -p Trades/.github/workflows
cp live/.github/workflows/pages.yml Trades/.github/workflows/pages.yml
cp live/.github/workflows/refresh-live-data.yml Trades/.github/workflows/refresh-live-data.yml
rsync -av live/scripts/build_live_snapshot.py Trades/scripts/
rsync -av live/site-config.js Trades/
rsync -av live/data/ Trades/data/
rsync -av live/trade_logs/ Trades/trade_logs/
rsync -av live/dashboard-fallback.json Trades/
```

## Step 2) Commit and push

```bash
cd /Users/justin/Downloads/skalz/Trades
git add .
git commit -m "Enable free live snapshot mode with persistent trade history"
git push origin main
```

## Step 3) Enable GitHub Actions permissions (one-time)
In GitHub repo settings:
1. `Settings` -> `Actions` -> `General`
2. Ensure Actions are enabled
3. Under workflow permissions, choose **Read and write permissions**
4. Save

The `refresh-live-data.yml` workflow commits updated snapshot files back to `main`, so write permission is required.

## Step 4) Run refresh workflow once manually
1. Open repo -> `Actions` tab
2. Choose **Refresh Live Snapshot Data**
3. Click **Run workflow**
4. Wait for green check

This creates your first fresh `data/dashboard.json` and populated `trade_logs/*`.

## Step 5) Confirm Pages deploy
1. Open `Actions` and confirm **Deploy GitHub Pages** ran after the refresh commit.
2. Open your site URL and hard refresh (`Cmd+Shift+R`).
3. Verify:
   - top metrics changed from old static values
   - Signal Studio journal starts filling over time

## How trade history is tracked
- Entry journal source: `trade_logs/nq_trade_journal.jsonl` + `.csv`
- Closed-trade ledger with PnL:
  - `data/trade_history.csv`
  - `data/trade_history.json`
- Fields include:
  - entry time
  - exit time
  - entry price
  - exit price
  - side
  - per-trade PnL
  - cumulative total PnL

## Notes
- This is near-real-time (15-minute refresh), not tick-by-tick realtime.
- GitHub Pages is static; the scheduled workflow is what keeps data fresh.
- The default test window is set to 60 days (`BACKTEST_RANGE_5M=60d`) in the workflow.
