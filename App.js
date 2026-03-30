import { html } from "./lib.js";
import { LeftRail } from "./components/LeftRail.js";
import { RightRail } from "./components/RightRail.js";
import { TopBar } from "./components/TopBar.js";
import { DashboardPage } from "./pages/DashboardPage.js";
import { RiskPage } from "./pages/RiskPage.js";
import { SignalsPage } from "./pages/SignalsPage.js";
import { Route, Routes } from "./router.js";

const EMPTY = {
  meta: {},
  headline: "Loading dashboard...",
  subheadline: "Fetching live NQ futures data and backtest analytics.",
  watchlist: [],
  filters: [],
  health: {},
  risk: { account_size: 50000, allocated_pct: 0 },
  kpis: {
    forecast_confidence_pct: 0,
    expected_session_return_pct: 0,
    projected_max_drawdown_pct: 0,
    sharpe_rolling_20: 0,
    max_drawdown_pct: 0,
    total_return_pct: 0,
    win_rate_pct: 0,
    n_trades: 0,
  },
  forecast: { points: [] },
  regimes: [],
  allocation: { rows: [] },
  events: [],
  signals: [],
  trade_journal: {
    path: "",
    csv_path: "",
    summary: {
      total_logged: 0,
      buy_count: 0,
      sell_count: 0,
      avg_risk_usd: 0,
      first_logged_at_et: null,
      last_logged_at_et: null,
    },
    recent: [],
    all: [],
  },
  trade_history: {
    path: "",
    csv_path: "",
    summary: {
      total_trades: 0,
      winning_trades: 0,
      losing_trades: 0,
      win_rate_pct: 0,
      avg_pnl_pct: 0,
      total_pnl_pct: 0,
      avg_pnl_usd: 0,
      total_pnl_usd: 0,
      first_entry_et: null,
      last_exit_et: null,
    },
    recent: [],
    all: [],
  },
  signal_mix: { buy: 0, sell: 0, hold: 0 },
  config: { cvar_alpha: 0.95, risk_aversion: 3.0 },
  pdf_trend: {
    bias: 0,
    momentum_lookback_days: 63,
    momentum_return_pct: 0,
    realized_ann_vol_pct: 0,
    target_ann_vol_pct: 12,
    target_leverage: 0,
  },
  repo_sources: [],
  distillation_stats: {
    dataset_records: 0,
    seed_records: 0,
    label_mix: { positive: 0, negative: 0, neutral: 0 },
    direction_mix: { BUY: 0, SELL: 0, HOLD: 0 },
    top_symbols: [],
    samples: [],
    config: {
      cost_bps: 5,
      effective_cost_bps: 5,
      min_signals: 10,
      hold_period: "1D",
      hold_bars_5m: 78,
      alpha: 0.95,
      risk_aversion: 3.0,
      max_sleeve_weight: 0.8,
    },
    backtest: { sharpe: 0, max_drawdown: 0, total_return: 0, win_rate: 0, n_trades: 0 },
    cvar_sized: {
      weights: { long_signal: 0, short_signal: 0, cash: 1 },
      objective: 0,
      expected_return: 0,
      cvar_loss: 0,
      sharpe: 0,
      max_drawdown: 0,
      total_return: 0,
      win_rate: 0,
      n_trades: 0,
    },
  },
  portfolio_optimization_stats: {
    assets: [],
    return_type: "LOG",
    fit_type: "gaussian",
    confidence: 0.95,
    scenario_count: 0,
    covariance_trace: 0,
    optimal_weights: [],
    frontier: [],
    window_start: "",
    window_end: "",
  },
};

export function App({ data, loading, error, refresh }) {
  const payload = data || EMPTY;

  return html`
    <div className="app-root">
      <div className="bg-noise" aria-hidden="true"></div>
      <div className="bg-halo" aria-hidden="true"></div>

      <div className="app-shell">
        <${TopBar} meta=${payload.meta} loading=${loading} onRefresh=${refresh} />

        <div className="workspace">
          <${LeftRail} watchlist=${payload.watchlist} />

          <div className="workspace-main">
            ${!data && loading
              ? html`<main className="main-grid"><div className="loading">Loading live quant data...</div></main>`
              : error && !data
                ? html`<main className="main-grid"><div className="error">${error}</div></main>`
                : html`
                    <${Routes}>
                      <${Route} path="/" element=${html`<${DashboardPage} data=${payload} />`} />
                      <${Route} path="/signals" element=${html`<${SignalsPage} data=${payload} />`} />
                      <${Route} path="/risk" element=${html`<${RiskPage} data=${payload} />`} />
                    <//>
                  `}
          </div>

          <${RightRail}
            risk=${payload.risk}
            health=${payload.health}
            filters=${payload.filters}
            repoSources=${payload.repo_sources}
          />
        </div>
      </div>
    </div>
  `;
}
