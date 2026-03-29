import { html } from "../lib.js";

export function RegimeTape({ regimes = [] }) {
  return html`
    <section className="panel regime-panel">
      <div className="section-head">
        <h3>Regime Tape</h3>
        <span>Last ${regimes.length || 0} windows</span>
      </div>
      <div className="regime-track">
        ${regimes.map(
          (item, idx) =>
            html`<div className=${`regime-state ${item.kind || "neutral"}`} key=${idx}>${item.label}</div>`
        )}
      </div>
    </section>
  `;
}
