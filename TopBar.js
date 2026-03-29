import { html } from "../lib.js";

export function TopBar({ meta, loading, onRefresh }) {
  const asOf = meta?.as_of_et || "--:--:-- ET";

  return html`
    <header className="topbar">
      <div className="brand-block">
        <div className="brand-mark" aria-hidden="true"></div>
        <p className="overline">NQ STRATEGY LAB • REPO-DRIVEN</p>
        <h1>Eigenstate-Inspired Desk</h1>
      </div>

      <div className="topbar-meta">
        <div className="status-pill">
          <span className="dot"></span>
          ${loading ? "Syncing" : "Live Engine"}
        </div>
        <div className="timestamp-block">
          <span className="timestamp-label">AS OF</span>
          <div className="timestamp">${asOf}</div>
        </div>
        <button className="action-btn" onClick=${onRefresh}>Refresh Data</button>
      </div>
    </header>
  `;
}
