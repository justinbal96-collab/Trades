import { html } from "../lib.js";

export function SignalsPage({ data }) {
  const signals = data.signals || [];
  const latest = data.kpis || {};
  const distill = data.distillation_stats || {};
  const labelMix = distill.label_mix || {};
  const dirMix = distill.direction_mix || {};
  const journal = data.trade_journal || {};
  const journalSummary = journal.summary || {};
  const journalRecent = (journal.recent || []).slice(-20).reverse();

  return html`
    <main className="main-grid">
      <section className="panel wide">
        <p className="panel-label">SIGNAL STUDIO</p>
        <h2>Live NQ signal stream from real 5-minute futures candles.</h2>
        <p>
          Signals are generated from rolling momentum and volatility filters, then cost-adjusted
          with transaction friction before backtest scoring.
        </p>
      </section>

      <section className="panel">
        <p className="panel-label">MODEL SNAPSHOT</p>
        <ul className="table-list">
          <li><span>Signal confidence</span><b>${latest.forecast_confidence_pct.toFixed(1)}%</b></li>
          <li><span>Expected return</span><b>${latest.expected_session_return_pct.toFixed(2)}%</b></li>
          <li><span>Win rate</span><b>${latest.win_rate_pct.toFixed(1)}%</b></li>
          <li><span>Trades (5d)</span><b>${latest.n_trades}</b></li>
        </ul>
      </section>

      <section className="panel">
        <p className="panel-label">SIGNAL DISTRIBUTION</p>
        <ul className="table-list">
          <li><span>BUY bars</span><b>${data.signal_mix.buy}</b></li>
          <li><span>SELL bars</span><b>${data.signal_mix.sell}</b></li>
          <li><span>HOLD bars</span><b>${data.signal_mix.hold}</b></li>
          <li><span>Latest close</span><b>${data.meta.last_price.toLocaleString()}</b></li>
        </ul>
      </section>

      <section className="panel">
        <p className="panel-label">KX DISTILLATION MIX</p>
        <ul className="table-list">
          <li><span>Positive labels</span><b>${labelMix.positive || 0}</b></li>
          <li><span>Negative labels</span><b>${labelMix.negative || 0}</b></li>
          <li><span>Neutral labels</span><b>${labelMix.neutral || 0}</b></li>
          <li><span>Mapped BUY/SELL/HOLD</span><b>${dirMix.BUY || 0}/${dirMix.SELL || 0}/${dirMix.HOLD || 0}</b></li>
        </ul>
      </section>

      <section className="panel">
        <p className="panel-label">TOP FINGPT SYMBOLS</p>
        <ul className="table-list">
          ${(distill.top_symbols || []).slice(0, 4).map(
            (row) => html`<li key=${row.symbol}><span>${row.symbol}</span><b>${row.count}</b></li>`
          )}
        </ul>
      </section>

      <section className="panel wide">
        <div className="section-head">
          <h3>Recent Signals</h3>
          <span>Real-time derived from NQ=F</span>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Time (ET)</th>
                <th>Close</th>
                <th>Signal</th>
                <th>Bar Return</th>
                <th>Strategy Return</th>
              </tr>
            </thead>
            <tbody>
              ${signals.map(
                (s) => html`
                  <tr key=${s.time + s.close}>
                    <td>${s.time}</td>
                    <td>${s.close.toLocaleString()}</td>
                    <td>
                      <span className=${s.signal === "BUY" ? "tag live" : s.signal === "SELL" ? "tag" : "tag muted"}
                        >${s.signal}</span
                      >
                    </td>
                    <td>${s.bar_return_pct.toFixed(3)}%</td>
                    <td>${s.strategy_return_pct.toFixed(3)}%</td>
                  </tr>
                `
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel wide">
        <div className="section-head">
          <h3>Persistent Trade Journal</h3>
          <span>auto-logged on directional, eligible signal changes</span>
        </div>
        <ul className="table-list">
          <li><span>Total logged trades</span><b>${journalSummary.total_logged || 0}</b></li>
          <li><span>BUY / SELL count</span><b>${`${journalSummary.buy_count || 0} / ${journalSummary.sell_count || 0}`}</b></li>
          <li><span>Avg risk per trade</span><b>${`$${(journalSummary.avg_risk_usd || 0).toFixed(2)}`}</b></li>
          <li><span>First logged</span><b>${journalSummary.first_logged_at_et || "--"}</b></li>
          <li><span>Last logged</span><b>${journalSummary.last_logged_at_et || "--"}</b></li>
          <li><span>Storage path</span><b>${journal.path || "--"}</b></li>
        </ul>
      </section>

      <section className="panel wide">
        <div className="section-head">
          <h3>Recent Logged Trades</h3>
          <span>latest 20 journal entries</span>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Execute (ET)</th>
                <th>Action</th>
                <th>Entry</th>
                <th>SL / TP</th>
                <th>Size (NQ / MNQ)</th>
                <th>Risk</th>
              </tr>
            </thead>
            <tbody>
              ${journalRecent.map(
                (row) => html`
                  <tr key=${row.event_id}>
                    <td>${row.execute_at_et || "--"}</td>
                    <td>
                      <span className=${row.action === "BUY" ? "tag live" : row.action === "SELL" ? "tag" : "tag muted"}
                        >${row.action || "HOLD"}</span
                      >
                    </td>
                    <td>${row.entry_reference ?? "--"}</td>
                    <td>${`${row.stop_price ?? "--"} / ${row.target_price ?? "--"}`}</td>
                    <td>${`${row.nq_contracts ?? 0} / ${row.mnq_contracts ?? 0}`}</td>
                    <td>${`$${Number(row.risk_per_trade_usd || 0).toFixed(2)}`}</td>
                  </tr>
                `
              )}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  `;
}
