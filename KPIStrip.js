import { html } from "../lib.js";

export function KPIStrip({ kpis = {} }) {
  const rows = [
    ["Forecast Confidence", `${(kpis.forecast_confidence_pct ?? 0).toFixed(1)}%`],
    [
      "Expected Session Return",
      `${(kpis.expected_session_return_pct ?? 0) >= 0 ? "+" : ""}${(kpis.expected_session_return_pct ?? 0).toFixed(2)}%`,
    ],
    ["Projected Max Drawdown", `${(kpis.projected_max_drawdown_pct ?? 0).toFixed(2)}%`],
    ["Sharpe (Rolling 20)", `${(kpis.sharpe_rolling_20 ?? 0).toFixed(2)}`],
  ];

  return html`
    <section className="panel kpi-strip">
      ${rows.map(
        ([label, value]) => html`
          <article className="kpi-card" key=${label}>
            <p>${label}</p>
            <h3>${value}</h3>
          </article>
        `
      )}
    </section>
  `;
}
