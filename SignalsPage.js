import { html } from "../lib.js";

function _formatEtFromUnix(unixSec) {
  const n = Number(unixSec);
  if (!Number.isFinite(n) || n <= 0) return null;
  const dt = new Date(n * 1000);
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).formatToParts(dt);
  const pick = (type) => parts.find((p) => p.type === type)?.value || "";
  const y = pick("year");
  const m = pick("month");
  const d = pick("day");
  const hh = pick("hour");
  const mm = pick("minute");
  const ss = pick("second");
  if (!y || !m || !d || !hh || !mm || !ss) return null;
  return `${y}-${m}-${d} ${hh}:${mm}:${ss} ET`;
}

function _formatTradeEt(row, side) {
  const isEntry = side === "entry";
  const label = isEntry
    ? row.entry_time_et || row.entry_et
    : row.exit_time_et || row.exit_et;
  if (typeof label === "string" && /^\d{4}-\d{2}-\d{2}\s/.test(label)) {
    return label;
  }
  const unixRaw = isEntry
    ? row.entry_unix ?? row.entry_time_unix
    : row.exit_unix ?? row.exit_time_unix;
  const fromUnix = _formatEtFromUnix(unixRaw);
  if (fromUnix) return fromUnix;
  return label || "--";
}

export function SignalsPage({ data }) {
  const signals = data.signals || [];
  const latest = data.kpis || {};
  const distill = data.distillation_stats || {};
  const labelMix = distill.label_mix || {};
  const dirMix = distill.direction_mix || {};
  const journal = data.trade_journal || {};
  const journalSummary = journal.summary || {};
  const history = data.trade_history || {};
  const historySummary = history.summary || {};
  const historyRows = (history.all || history.recent || []).slice().reverse();

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
          <li><span>Total stored trades</span><b>${historySummary.total_trades || historyRows.length || 0}</b></li>
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
          <span>stores all future directional signal events (no 20-row cap)</span>
        </div>
        <ul className="table-list">
          <li><span>Total logged trades</span><b>${journalSummary.total_logged || 0}</b></li>
          <li><span>BUY / SELL count</span><b>${`${journalSummary.buy_count || 0} / ${journalSummary.sell_count || 0}`}</b></li>
          <li><span>Avg risk per trade</span><b>${`$${(journalSummary.avg_risk_usd || 0).toFixed(2)}`}</b></li>
          <li><span>First logged</span><b>${journalSummary.first_logged_at_et || "--"}</b></li>
          <li><span>Last logged</span><b>${journalSummary.last_logged_at_et || "--"}</b></li>
          <li><span>Storage path</span><b>${journal.path || "--"}</b></li>
          <li><span>Backtest + live history path</span><b>${history.path || "--"}</b></li>
        </ul>
      </section>

      <section className="panel wide">
        <div className="section-head">
          <h3>All Stored Trades (Backtest + Live)</h3>
          <span>full retained history (most recent first)</span>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Entry (ET)</th>
                <th>Exit (ET)</th>
                <th>Direction</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>PnL %</th>
                <th>PnL $</th>
                <th>Cumulative $</th>
              </tr>
            </thead>
            <tbody>
              ${historyRows.map(
                (row) => html`
                  <tr key=${row.trade_id || row.event_id || `${row.entry_time_et || row.entry_et}-${row.exit_time_et || row.exit_et}`}>
                    <td>${_formatTradeEt(row, "entry")}</td>
                    <td>${_formatTradeEt(row, "exit")}</td>
                    <td>
                      <span className=${String(row.direction || "").includes("LONG") ? "tag live" : "tag"}
                        >${row.direction || row.action || "--"}</span
                      >
                    </td>
                    <td>${row.entry_price ?? row.entry_reference ?? "--"}</td>
                    <td>${row.exit_price ?? "--"}</td>
                    <td>${`${Number(row.pnl_pct ?? row.trade_profit_pct ?? 0).toFixed(3)}%`}</td>
                    <td>${`$${Number(row.pnl_usd ?? row.trade_profit_usd ?? row.trade_pnl_usd ?? 0).toFixed(2)}`}</td>
                    <td>${`$${Number(row.cumulative_pnl_usd ?? row.total_pnl_usd ?? 0).toFixed(2)}`}</td>
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
