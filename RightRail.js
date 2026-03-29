import { html } from "../lib.js";

export function RightRail({ risk = {}, health = {}, filters = [], repoSources = [] }) {
  const allocated = risk.allocated_pct ?? 0;

  return html`
    <aside className="right-rail">
      <section className="rail-section">
        <p className="rail-label">RISK BUDGET</p>
        <h4>$${(risk.account_size ?? 0).toLocaleString()} account</h4>
        <div className="meter"><div className="meter-fill" style=${{ width: `${allocated}%` }}></div></div>
        <div className="small-note">${allocated.toFixed(0)}% allocated • ${(100 - allocated).toFixed(0)}% dry powder</div>
      </section>

      <section className="rail-section">
        <p className="rail-label">TRADE FILTERS</p>
        <div className="chip-row">
          ${filters.map(
            (item) => html`<span key=${item.label} className=${item.active ? "chip active" : "chip"}>${item.label}</span>`
          )}
        </div>
      </section>

      <section className="rail-section">
        <p className="rail-label">SYSTEM HEALTH</p>
        <ul className="health-list">
          <li><span>Signal model</span><b>${health.model || "--"}</b></li>
          <li><span>Data feed</span><b>${health.feed || "--"}</b></li>
          <li><span>Backtest drift</span><b>${health.drift || "--"}</b></li>
        </ul>
      </section>

      <section className="rail-section">
        <p className="rail-label">QUANT SOURCES</p>
        <ul className="health-list">
          ${repoSources.map(
            (source) => html`<li key=${source.name}><span>${source.name}</span><b>${source.status}</b></li>`
          )}
        </ul>
      </section>
    </aside>
  `;
}
