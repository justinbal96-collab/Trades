import { html } from "../lib.js";

export function RiskPage({ data }) {
  const k = data.kpis;
  const qpo = data.portfolio_optimization_stats || {};
  const kx = data.distillation_stats || {};
  const kxBacktest = kx.backtest || {};
  const kxCvar = kx.cvar_sized || {};

  return html`
    <main className="main-grid">
      <section className="panel wide">
        <p className="panel-label">RISK ENVELOPE</p>
        <h2>Backtest quality and downside profile from live NQ bars.</h2>
        <p>
          Metrics below are recomputed from the current 5-day NQ window using a cost-aware
          long/short signal model and CVaR sleeve sizing.
        </p>
      </section>

      <section className="panel">
        <p className="panel-label">PERFORMANCE</p>
        <ul className="table-list">
          <li><span>Sharpe (annualized)</span><b>${k.sharpe_rolling_20.toFixed(2)}</b></li>
          <li><span>Total return (5d)</span><b>${k.total_return_pct.toFixed(2)}%</b></li>
          <li><span>Win rate</span><b>${k.win_rate_pct.toFixed(1)}%</b></li>
          <li><span>Trades</span><b>${k.n_trades}</b></li>
        </ul>
      </section>

      <section className="panel">
        <p className="panel-label">DOWNSIDE</p>
        <ul className="table-list">
          <li><span>Max drawdown</span><b>${k.max_drawdown_pct.toFixed(2)}%</b></li>
          <li><span>Projected drawdown</span><b>${k.projected_max_drawdown_pct.toFixed(2)}%</b></li>
          <li><span>CVaR alpha</span><b>${data.config.cvar_alpha.toFixed(2)}</b></li>
          <li><span>Risk aversion</span><b>${data.config.risk_aversion.toFixed(1)}</b></li>
        </ul>
      </section>

      <section className="panel wide">
        <div className="section-head">
          <h3>Sleeve Breakdown</h3>
          <span>Long/Short/Cash with CVaR objective</span>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Sleeve</th>
                <th>Weight</th>
                <th>Expected Return</th>
                <th>CVaR Loss</th>
                <th>Role</th>
              </tr>
            </thead>
            <tbody>
              ${data.allocation.rows.map(
                (row) => html`
                  <tr key=${row.sleeve}>
                    <td>${row.sleeve}</td>
                    <td>${row.weight.toFixed(2)}</td>
                    <td>${row.expected_return_pct.toFixed(3)}%</td>
                    <td>${row.cvar_loss_pct.toFixed(3)}%</td>
                    <td>${row.status}</td>
                  </tr>
                `
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel wide">
        <div className="section-head">
          <h3>QPO Efficient Frontier Snapshot</h3>
          <span>Gaussian scenarios from multi-asset LOG returns</span>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Risk Aversion</th>
                <th>Expected Return</th>
                <th>CVaR Loss</th>
                <th>Objective</th>
              </tr>
            </thead>
            <tbody>
              ${(qpo.frontier || []).map(
                (row) => html`
                  <tr key=${row.risk_aversion}>
                    <td>${row.risk_aversion.toFixed(1)}</td>
                    <td>${row.expected_return_pct.toFixed(3)}%</td>
                    <td>${row.cvar_loss_pct.toFixed(3)}%</td>
                    <td>${row.objective.toFixed(6)}</td>
                  </tr>
                `
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel wide">
        <div className="section-head">
          <h3>KX Backtest Overlay</h3>
          <span>Signal economics from distillation dataset mapping</span>
        </div>
        <ul className="table-list">
          <li><span>Raw Sharpe</span><b>${(kxBacktest.sharpe ?? 0).toFixed(2)}</b></li>
          <li><span>Raw Max Drawdown</span><b>${((kxBacktest.max_drawdown ?? 0) * 100).toFixed(2)}%</b></li>
          <li><span>Raw Total Return</span><b>${((kxBacktest.total_return ?? 0) * 100).toFixed(2)}%</b></li>
          <li><span>Raw Win Rate</span><b>${((kxBacktest.win_rate ?? 0) * 100).toFixed(1)}%</b></li>
          <li><span>CVaR-Sized Sharpe</span><b>${(kxCvar.sharpe ?? 0).toFixed(2)}</b></li>
          <li><span>CVaR-Sized Return</span><b>${((kxCvar.total_return ?? 0) * 100).toFixed(2)}%</b></li>
        </ul>
      </section>
    </main>
  `;
}
