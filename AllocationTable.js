import { html } from "../lib.js";

export function AllocationTable({ allocation = {}, alpha = 0.95, riskAversion = 3.0 }) {
  const rows = allocation.rows || [];

  return html`
    <section className="panel positions-panel">
      <div className="section-head">
        <h3>CVaR Sleeve Allocation</h3>
        <span>alpha ${alpha.toFixed(2)} • risk aversion ${riskAversion.toFixed(1)}</span>
      </div>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Sleeve</th>
              <th>Weight</th>
              <th>Daily VaR</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            ${rows.map(
              (row) => html`
                <tr key=${row.sleeve}>
                  <td>${row.sleeve}</td>
                  <td>${row.weight.toFixed(2)}</td>
                  <td>${row.daily_var_pct.toFixed(2)}%</td>
                  <td>
                    <span className=${row.status === "Active" ? "tag live" : row.status === "Reserve" ? "tag muted" : "tag"}
                      >${row.status}</span
                    >
                  </td>
                </tr>
              `
            )}
          </tbody>
        </table>
      </div>
    </section>
  `;
}
