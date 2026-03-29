# GitHub Upload + Launch Guide

This repo is set up for:
- Static frontend on GitHub Pages
- Python API deployed separately (Render/Railway/Fly/etc.)

## 1) Upload to GitHub

If your repo does not exist yet:

```bash
cd /path/to/eigenstate-inspired-dashboard
git init
git add .
git commit -m "Initial dashboard deploy"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

If your repo already exists, just `git add/commit/push`.

## 2) Enable GitHub Pages

1. Open your repo on GitHub.
2. Go to `Settings -> Pages`.
3. Set `Source` to `GitHub Actions`.
4. Push to `main` (or run workflow manually in `Actions`).

The workflow is already included:
- `.github/workflows/pages.yml`

If you upload files manually through the GitHub web UI, include both:
- `dist/app.js` (and `dist/app.js.map`)
- `app.js` (and `app.js.map`)

`index.html` now auto-falls back to `app.js` if `dist/app.js` is missing, so the app still boots.

## 3) Deploy the Python API

GitHub Pages cannot run `server.py`.
Deploy `server.py` to a Python host (Render/Railway/etc.) and get a URL like:

`https://your-backend.example.com/api/dashboard`

## 4) Point frontend to your API

Edit:
- `site-config.js`

Set:

```js
window.__NQ_DASHBOARD_API_URL__ = "https://your-backend.example.com/api/dashboard";
```

Commit + push again. Pages will rebuild and use your live API.

If `site-config.js` is left as `/api/dashboard` on GitHub Pages, the app now auto-falls back to:
- `dashboard-fallback.json`

So the page still renders (snapshot mode), but it is not live until you set the backend URL above.

## 5) CORS

`server.py` already includes permissive CORS headers (`Access-Control-Allow-Origin: *`) so your GitHub Pages frontend can call the API.
