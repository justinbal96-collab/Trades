import { html } from "../lib.js";
import { AllocationTable } from "../components/AllocationTable.js";
import { EventFeed } from "../components/EventFeed.js";
import { ForecastChart } from "../components/ForecastChart.js";
import { KPIStrip } from "../components/KPIStrip.js";
import { RegimeTape } from "../components/RegimeTape.js";

function money(value) {
  const n = Number(value) || 0;
  return `${n < 0 ? "-" : ""}$${Math.abs(n).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

function pct(value, digits = 2) {
  const n = Number(value) || 0;
  return `${n >= 0 ? "+" : ""}${n.toFixed(digits)}%`;
}

function metric(value, digits = 2) {
  const n = Number(value) || 0;
  return n.toFixed(digits);
}

function rowByName(rows, name) {
  return rows.find((r) => r.sleeve === name) || { weight: 0 };
}

function buildActionDecision(data) {
  const k = data.kpis || {};
  const mix = data.signal_mix || {};
  const latestSignal = (data.signals || []).at(-1)?.signal || "HOLD";
  const rows = data.allocation?.rows || [];
  const longW = Number(rowByName(rows, "Long Signal").weight || 0);
  const shortW = Number(rowByName(rows, "Short Signal").weight || 0);
  const conf = Number(k.forecast_confidence_pct || 0);
  const expected = Number(k.expected_session_return_pct || 0);
  const directionalBars = Number(mix.buy || 0) + Number(mix.sell || 0);

  const reasons = [
    `Latest bar signal: ${latestSignal}`,
    `Sleeve tilt: long ${longW.toFixed(2)} vs short ${shortW.toFixed(2)}`,
    `Expected session return: ${pct(expected, 2)}`,
  ];

  if (conf < 40 || directionalBars < 20) {
    return {
      action: "NO-TRADE",
      tone: "neutral",
      reasons: [...reasons, "Confidence is low, so protect capital."],
    };
  }

  if (expected <= -0.25 && shortW - longW >= 0.1) {
    if (latestSignal === "BUY") {
      return {
        action: "NO-TRADE",
        tone: "neutral",
        reasons: [...reasons, "Signal conflicts with bearish sleeve + return regime."],
      };
    }
    return {
      action: "SELL",
      tone: "bad",
      reasons: [...reasons, "Bearish alignment across return expectation and sleeve weights."],
    };
  }

  if (expected >= 0.25 && longW - shortW >= 0.1) {
    if (latestSignal === "SELL") {
      return {
        action: "NO-TRADE",
        tone: "neutral",
        reasons: [...reasons, "Signal conflicts with bullish sleeve + return regime."],
      };
    }
    return {
      action: "BUY",
      tone: "good",
      reasons: [...reasons, "Bullish alignment across return expectation and sleeve weights."],
    };
  }

  if (latestSignal === "HOLD") {
    return {
      action: "HOLD",
      tone: "neutral",
      reasons: [...reasons, "No directional edge on the latest bar."],
    };
  }

  return {
    action: "NO-TRADE",
    tone: "neutral",
    reasons: [...reasons, "Mixed evidence; wait for alignment."],
  };
}

export function DashboardPage({ data }) {
  const qpo = data.portfolio_optimization_stats || {};
  const kx = data.distillation_stats || {};
  const pdfTrend = data.pdf_trend || {};
  const execPlan = data.execution_plan || {};
  const accountSize = Number(data.risk?.account_size || 50000);
  const action = buildActionDecision(data);
  const k = data.kpis || {};
  const checks = execPlan.prop_rules?.checks || [];

  const th = data.trade_history?.summary || {};
  const liveTrades = Number(th.total_trades ?? th.total_closed ?? 0);
  const livePnlUsd = Number(th.total_pnl_usd ?? th.latest_total_pnl_usd ?? 0);
  const perfRows = [
    {
      name: "Main Live NQ Model",
      returnPct: livePnlUsd / accountSize * 100,
      pnl: livePnlUsd,
      sharpe: Number(k.sharpe_rolling_20 || 0),
      maxDdPct: Number(k.max_drawdown_pct || 0),
      maxDdUsd: accountSize * (Number(k.max_drawdown_pct || 0) / 100),
      winPct: Number(th.win_rate_pct || k.win_rate_pct || 0),
      trades: liveTrades || Number(k.n_trades || 0),
    },
    {
      name: "KX Distillation Backtest",
      returnPct: Number(kx.backtest?.total_return || 0) * 100,
      pnl: accountSize * Number(kx.backtest?.total_return || 0),
      sharpe: Number(kx.backtest?.sharpe || 0),
      maxDdPct: Number(kx.backtest?.max_drawdown || 0) * 100,
      maxDdUsd: accountSize * Number(kx.backtest?.max_drawdown || 0),
      winPct: Number(kx.backtest?.win_rate || 0) * 100,
      trades: Number(kx.backtest?.n_trades || 0),
    },
    {
      name: "KX CVaR-Sized Overlay",
      returnPct: Number(kx.cvar_sized?.total_return || 0) * 100,
      pnl: accountSize * Number(kx.cvar_sized?.total_return || 0),
      sharpe: Number(kx.cvar_sized?.sharpe || 0),
      maxDdPct: Number(kx.cvar_sized?.max_drawdown || 0) * 100,
      maxDdUsd: accountSize * Number(kx.cvar_sized?.max_drawdown || 0),
      winPct: Number(kx.cvar_sized?.win_rate || 0) * 100,
      trades: Number(kx.cvar_sized?.n_trades || 0),
    },
  ];

  return html`
    <main className="main-grid">
      <section className="panel headline">
        <p className="panel-label">Current Bias</p>
        <h2>${data.headline}</h2>
        <p>${data.subheadline}</p>
      </section>

      <${KPIStrip} kpis=${data.kpis} />
      <section className="panel action-panel">
        <div className="section-head">
          <h3>Action Engine</h3>
          <span>buy / sell / hold gate</span>
        </div>
        <p className=${`action-badge ${action.tone}`}>${action.action}</p>
        <ul className="table-list compact">
          ${action.reasons.map((reason) => html`<li><span>${reason}</span></li>`)}
        </ul>
      </section>
      <section className="panel action-panel">
        <div className="section-head">
          <h3>Prop Execution Plan</h3>
          <span>${execPlan.next_bar_et || "next bar ET pending"}</span>
        </div>
        <p className=${`action-badge ${execPlan.eligible ? "good" : "neutral"}`}>
          ${execPlan.eligible ? "ELIGIBLE" : "PAUSE"}
        </p>
        <ul className="table-list">
          <li><span>Action next 5m bar</span><b>${execPlan.action_next_bar || "HOLD"}</b></li>
          <li><span>Signal changed this bar</span><b>${execPlan.signal_changed ? "YES" : "NO"}</b></li>
          <li><span>Current position</span><b>${execPlan.current_position || "HOLD"}</b></li>
          <li><span>Entry reference</span><b>${execPlan.entry_reference || "--"}</b></li>
          <li><span>Stop / Target</span><b>${execPlan.stop_price && execPlan.target_price ? `${execPlan.stop_price} / ${execPlan.target_price}` : "-- / --"}</b></li>
          <li><span>Contracts (NQ / MNQ)</span><b>${`${execPlan.contract_plan?.nq ?? 0} / ${execPlan.contract_plan?.mnq ?? 0}`}</b></li>
          <li><span>Risk per trade</span><b>${money(execPlan.risk_per_trade_usd || 0)}</b></li>
          <li><span>Trades today / cap</span><b>${`${execPlan.prop_rules?.trades_today ?? 0} / ${execPlan.prop_rules?.max_trades_per_day ?? 0}`}</b></li>
          <li><span>Model day PnL</span><b className=${(execPlan.prop_rules?.daily_model_pnl_usd ?? 0) >= 0 ? "good" : "bad"}>${money(execPlan.prop_rules?.daily_model_pnl_usd || 0)}</b></li>
        </ul>
        <p className="small-note">${execPlan.notes || "Use this as execution guidance, not financial advice."}</p>
      </section>
      <section className="panel">
        <div className="section-head">
          <h3>Decision Inputs</h3>
          <span>what drives Action Engine</span>
        </div>
        <ul className="table-list">
          <li><span>Forecast confidence</span><b>${pct(k.forecast_confidence_pct || 0, 1)}</b></li>
          <li><span>Expected session return</span><b>${pct(k.expected_session_return_pct || 0, 2)}</b></li>
          <li><span>Signal bars (buy / sell / hold)</span><b>${`${data.signal_mix.buy} / ${data.signal_mix.sell} / ${data.signal_mix.hold}`}</b></li>
          <li><span>Latest executable signal</span><b>${(data.signals || []).at(-1)?.signal || "HOLD"}</b></li>
          <li><span>PDF 63d macro bias</span><b>${pdfTrend.bias > 0 ? "LONG" : pdfTrend.bias < 0 ? "SHORT" : "FLAT"}</b></li>
          <li><span>PDF 63d momentum</span><b>${pct(pdfTrend.momentum_return_pct || 0, 2)}</b></li>
        </ul>
      </section>
      <${ForecastChart} forecast=${data.forecast} />
      <${RegimeTape} regimes=${data.regimes} />
      <${AllocationTable}
        allocation=${data.allocation}
        alpha=${data.config.cvar_alpha}
        riskAversion=${data.config.risk_aversion}
      />
      <section className="panel wide">
        <div className="section-head">
          <h3>Profitability Breakdown</h3>
          <span>normalized to $${accountSize.toLocaleString()} account</span>
        </div>
        <div className="table-wrap">
          <table className="data-table profitability-table">
            <thead>
              <tr>
                <th>Model Layer</th>
                <th>Total Return</th>
                <th>PnL</th>
                <th>Sharpe</th>
                <th>Max Drawdown</th>
                <th>Win Rate</th>
                <th>Trades</th>
              </tr>
            </thead>
            <tbody>
              ${perfRows.map(
                (r) => html`
                  <tr>
                    <td>${r.name}</td>
                    <td className=${r.returnPct >= 0 ? "good" : "bad"}>${pct(r.returnPct)}</td>
                    <td className=${r.pnl >= 0 ? "good" : "bad"}>${money(r.pnl)}</td>
                    <td>${metric(r.sharpe)}</td>
                    <td className="bad">${`${pct(r.maxDdPct)} (${money(r.maxDdUsd)})`}</td>
                    <td>${pct(r.winPct)}</td>
                    <td>${r.trades}</td>
                  </tr>
                `
              )}
            </tbody>
          </table>
        </div>
        <p className="small-note">
          Main model = live NQ signal engine on recent bars. KX rows = repo-derived backtest overlays.
          CVaR-sized row is the risk-controlled version.
        </p>
        <ul className="table-list">
          <li><span>Backtest window</span><b>${`${execPlan.backtest_window?.start_et || "--"} to ${execPlan.backtest_window?.end_et || "--"}`}</b></li>
          <li><span>Bars / session-equivalent</span><b>${`${execPlan.backtest_window?.bars || 0} / ${execPlan.backtest_window?.sessions_equiv || 0}`}</b></li>
          <li><span>Trades per session</span><b>${execPlan.backtest_window?.trades_per_session || 0}</b></li>
          <li><span>Average return per trade</span><b>${pct(execPlan.backtest_window?.avg_return_per_trade_pct || 0, 3)}</b></li>
        </ul>
      </section>
      <section className="panel wide">
        <div className="section-head">
          <h3>Prop Rule Checks</h3>
          <span>execution guardrails</span>
        </div>
        <ul className="table-list">
          ${checks.map(
            (c) => html`<li><span>${c.name}</span><b className=${c.pass ? "good" : "bad"}>${c.pass ? "PASS" : "FAIL"} • ${c.detail}</b></li>`
          )}
        </ul>
      </section>
      <section className="panel wide">
        <div className="section-head">
          <h3>Repository-Integrated Quant Layer</h3>
          <span>NVIDIA QPO + KX distillation adapters</span>
        </div>
        <ul className="table-list">
          <li><span>QPO scenarios</span><b>${qpo.scenario_count || 0}</b></li>
          <li><span>QPO fit / return type</span><b>${qpo.fit_type || "--"} / ${qpo.return_type || "--"}</b></li>
          <li><span>KX sentiment records</span><b>${kx.dataset_records || 0}</b></li>
          <li><span>KX hold period</span><b>${kx.config?.hold_period || "--"}</b></li>
          <li><span>KX cost (effective bps)</span><b>${(kx.config?.effective_cost_bps ?? 0).toFixed(2)}</b></li>
          <li><span>KX CVaR-sized Sharpe</span><b>${(kx.cvar_sized?.sharpe ?? 0).toFixed(2)}</b></li>
          <li><span>PDF target annual vol</span><b>${pct(pdfTrend.target_ann_vol_pct || 0, 1)}</b></li>
          <li><span>PDF realized annual vol</span><b>${pct(pdfTrend.realized_ann_vol_pct || 0, 1)}</b></li>
          <li><span>PDF target leverage</span><b>${metric(pdfTrend.target_leverage || 0, 2)}x</b></li>
        </ul>
      </section>
      <${EventFeed} events=${data.events} />
    </main>
  `;
}
