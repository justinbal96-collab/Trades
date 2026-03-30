#!/usr/bin/env python3
"""300-iteration optimizer for prop-firm constraints on NQ 5m data (60d).

Hard constraints:
- max hold: 90 minutes
- max active trade loss: -$500 with intrabar stop-touch model
- same-session-day trades only (no cross-session holds)

Ranking priority:
1) win rate
2) total return
3) total PnL
4) lower drawdown (less negative MDD)
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

# Fixed optimization window for all candidates.
os.environ["BACKTEST_RANGE_5M"] = "60d"
os.environ["QPO_RANGE_5M"] = "60d"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server

SERVER_FILE = ROOT / "server.py"
SNAPSHOT_DIR = ROOT / "snapshots"
RESULTS_DIR = ROOT / "results"

BASE_CONFIG_KEYS_ORDER = list(server.LIVE_SIGNAL_CONFIG.keys())

M_WINDOWS = [8, 10, 12, 14, 16, 18, 20, 24, 28]
V_WINDOWS = [12, 16, 20, 24, 28, 32, 36, 40]
T_WINDOWS = [24, 32, 40, 48, 64, 80, 96, 120, 144, 180]

TUNABLE_KEYS = [
    "momentum_window",
    "volatility_window",
    "trend_window",
    "momentum_threshold",
    "volatility_quantile_cap",
    "countertrend_multiplier",
    "min_hold_bars",
    "macro_countertrend_allow_multiplier",
    "trend_regime_multiplier",
    "neutral_regime_multiplier",
    "high_volatility_multiplier",
    "short_entry_multiplier",
    "disable_longs_when_macro_short",
    "trade_cooldown_bars",
]

HARD_CONSTRAINTS = {
    "max_hold_minutes": 90,
    "max_active_trade_loss_usd": 500.0,
    "intrabar_stop_touch": True,
    "enforce_same_day_trades": True,
}

# Activity/PnL profile (user asked for more trades + higher PnL).
TARGET_TRADES_PER_DAY = 1.0


@dataclass
class EvalResult:
    metrics: dict[str, Any]
    folds: list[dict[str, Any]]
    feasible: bool


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _log_uniform(lo: float, hi: float) -> float:
    return 10 ** random.uniform(math.log10(lo), math.log10(hi))


def _sanitize(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    if out.get("momentum_window") not in M_WINDOWS:
        out["momentum_window"] = random.choice(M_WINDOWS)
    if out.get("volatility_window") not in V_WINDOWS:
        out["volatility_window"] = random.choice(V_WINDOWS)
    if out.get("trend_window") not in T_WINDOWS:
        out["trend_window"] = random.choice(T_WINDOWS)

    out["momentum_window"] = int(out["momentum_window"])
    out["volatility_window"] = int(out["volatility_window"])
    out["trend_window"] = int(out["trend_window"])
    out["momentum_threshold"] = float(_clamp(float(out["momentum_threshold"]), 1.0e-5, 2.5e-4))
    out["volatility_quantile_cap"] = float(_clamp(float(out["volatility_quantile_cap"]), 0.40, 0.95))
    out["countertrend_multiplier"] = float(_clamp(float(out["countertrend_multiplier"]), 1.0, 4.2))
    out["min_hold_bars"] = int(_clamp(int(out["min_hold_bars"]), 1, 12))
    out["macro_countertrend_allow_multiplier"] = float(
        _clamp(float(out["macro_countertrend_allow_multiplier"]), 1.0, 7.0)
    )
    out["trend_regime_multiplier"] = float(_clamp(float(out["trend_regime_multiplier"]), 0.2, 3.5))
    out["neutral_regime_multiplier"] = float(_clamp(float(out["neutral_regime_multiplier"]), 0.6, 4.5))
    out["high_volatility_multiplier"] = float(_clamp(float(out["high_volatility_multiplier"]), 0.6, 2.2))
    out["short_entry_multiplier"] = float(_clamp(float(out["short_entry_multiplier"]), 0.25, 2.5))
    out["trade_cooldown_bars"] = int(_clamp(int(out["trade_cooldown_bars"]), 0, 8))
    out["disable_longs_when_macro_short"] = bool(out["disable_longs_when_macro_short"])

    # Keep constraints hard-fixed.
    out.update(HARD_CONSTRAINTS)
    return out


def _param_key(cfg: dict[str, Any]) -> tuple:
    return tuple(
        (
            round(float(cfg[k]), 9)
            if isinstance(cfg.get(k), float)
            else int(cfg[k])
            if isinstance(cfg.get(k), (bool, int))
            else cfg.get(k)
        )
        for k in TUNABLE_KEYS
    )


def _random_candidate(base: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(base)
    cfg.update(
        {
            "momentum_window": random.choice(M_WINDOWS),
            "volatility_window": random.choice(V_WINDOWS),
            "trend_window": random.choice(T_WINDOWS),
            "momentum_threshold": _log_uniform(1.0e-5, 2.2e-4),
            "volatility_quantile_cap": random.uniform(0.40, 0.95),
            "countertrend_multiplier": random.uniform(1.0, 4.0),
            "min_hold_bars": random.randint(1, 10),
            "macro_countertrend_allow_multiplier": random.uniform(1.0, 6.8),
            "trend_regime_multiplier": random.uniform(0.2, 3.2),
            "neutral_regime_multiplier": random.uniform(0.6, 4.2),
            "high_volatility_multiplier": random.uniform(0.6, 2.1),
            "short_entry_multiplier": random.uniform(0.3, 2.2),
            "disable_longs_when_macro_short": random.choice([False, True]),
            "trade_cooldown_bars": random.randint(0, 6),
        }
    )
    return _sanitize(cfg)


def _mutate(parent: dict[str, Any], strength: float) -> dict[str, Any]:
    cfg = dict(parent)
    if random.random() < 0.45:
        cfg["momentum_window"] = random.choice(M_WINDOWS)
    if random.random() < 0.45:
        cfg["volatility_window"] = random.choice(V_WINDOWS)
    if random.random() < 0.45:
        cfg["trend_window"] = random.choice(T_WINDOWS)

    cfg["momentum_threshold"] = float(cfg["momentum_threshold"]) * math.exp(random.gauss(0.0, 0.45 * strength))
    cfg["volatility_quantile_cap"] = float(cfg["volatility_quantile_cap"]) + random.gauss(0.0, 0.10 * strength)
    cfg["countertrend_multiplier"] = float(cfg["countertrend_multiplier"]) + random.gauss(0.0, 0.55 * strength)
    cfg["min_hold_bars"] = int(cfg["min_hold_bars"]) + int(round(random.gauss(0.0, 3.5 * strength)))
    cfg["macro_countertrend_allow_multiplier"] = float(cfg["macro_countertrend_allow_multiplier"]) + random.gauss(
        0.0, 0.75 * strength
    )
    cfg["trend_regime_multiplier"] = float(cfg["trend_regime_multiplier"]) + random.gauss(0.0, 0.30 * strength)
    cfg["neutral_regime_multiplier"] = float(cfg["neutral_regime_multiplier"]) + random.gauss(0.0, 0.38 * strength)
    cfg["high_volatility_multiplier"] = float(cfg["high_volatility_multiplier"]) + random.gauss(0.0, 0.22 * strength)
    cfg["short_entry_multiplier"] = float(cfg["short_entry_multiplier"]) + random.gauss(0.0, 0.30 * strength)
    cfg["trade_cooldown_bars"] = int(cfg["trade_cooldown_bars"]) + int(round(random.gauss(0.0, 2.5 * strength)))

    if random.random() < (0.08 * strength + 0.02):
        cfg["disable_longs_when_macro_short"] = not bool(cfg["disable_longs_when_macro_short"])

    return _sanitize(cfg)


def _format_live_config(full_cfg: dict[str, Any]) -> str:
    lines = ["LIVE_SIGNAL_CONFIG = {"]
    for key in BASE_CONFIG_KEYS_ORDER:
        val = full_cfg[key]
        lines.append(f'    "{key}": {repr(val)},')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _apply_live_config(full_cfg: dict[str, Any]) -> None:
    src = SERVER_FILE.read_text(encoding="utf-8")
    repl = _format_live_config(full_cfg)
    pattern = r"LIVE_SIGNAL_CONFIG = \{[\s\S]*?\n\}"
    if re.search(pattern, src) is None:
        raise RuntimeError("Failed to locate LIVE_SIGNAL_CONFIG block for update")
    updated = re.sub(pattern, repl.rstrip(), src, count=1)
    SERVER_FILE.write_text(updated, encoding="utf-8")


def _slice_segment(data: dict[str, list[float] | list[int]], start: int, end: int) -> dict[str, list[float] | list[int]]:
    # start/end are return indices [start, end).
    return {
        "timestamps": data["timestamps"][start : end + 1],
        "open": data["open"][start : end + 1],
        "high": data["high"][start : end + 1],
        "low": data["low"][start : end + 1],
        "close": data["close"][start : end + 1],
    }


def _count_trading_session_days(timestamps: list[int]) -> int:
    if not timestamps:
        return 0
    return len(
        {
            server._trading_session_id(datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(server.ET_TZ))
            for ts in timestamps
        }
    )


def _constraint_checks_from_trades(
    trades: list[dict[str, Any]],
    *,
    max_hold_minutes: float = 90.0,
    loss_cap_usd: float = 500.0,
) -> dict[str, Any]:
    exit_reason_distribution: dict[str, int] = {}
    max_hold_violations = 0
    loss_cap_violations = 0
    cross_session_violations = 0

    for t in trades:
        reason = str(t.get("exit_reason", "unknown"))
        exit_reason_distribution[reason] = exit_reason_distribution.get(reason, 0) + 1
        if float(t.get("minutes_held", 0.0) or 0.0) > max_hold_minutes + 1e-9:
            max_hold_violations += 1
        if float(t.get("pnl_usd", 0.0) or 0.0) < -loss_cap_usd - 1e-6:
            loss_cap_violations += 1
        if str(t.get("entry_session_id")) != str(t.get("exit_session_id")):
            cross_session_violations += 1

    return {
        "max_hold_violation_count": int(max_hold_violations),
        "loss_cap_violation_count": int(loss_cap_violations),
        "cross_session_violation_count": int(cross_session_violations),
        "exit_reason_distribution": exit_reason_distribution,
        "max_hold_minutes": int(max_hold_minutes),
        "max_active_trade_loss_usd": float(loss_cap_usd),
    }


def _compute_metrics_from_parts(
    trades: list[dict[str, Any]],
    strat_returns: list[float],
    timestamps: list[int],
    checks: dict[str, Any],
) -> dict[str, Any]:
    total_return_pct = (server._cum_equity(strat_returns)[-1] - 1.0) * 100.0 if strat_returns else 0.0
    max_drawdown_pct = server._max_drawdown(strat_returns) * 100.0 if strat_returns else 0.0
    mean_ret = server._mean(strat_returns) if strat_returns else 0.0
    std_ret = server._stdev(strat_returns) if strat_returns else 0.0
    sharpe = (mean_ret / std_ret * math.sqrt(252 * server.BARS_PER_SESSION)) if std_ret > 1e-9 else 0.0

    pnl_usd_vals = [float(t.get("pnl_usd", 0.0) or 0.0) for t in trades]
    wins = sum(1 for x in pnl_usd_vals if x > 0.0)
    losses = sum(1 for x in pnl_usd_vals if x <= 0.0)
    win_rate_pct = (wins / len(trades) * 100.0) if trades else 0.0
    total_pnl_usd = float(sum(pnl_usd_vals))

    trading_session_days = _count_trading_session_days(timestamps)
    trades_per_day = (len(trades) / trading_session_days) if trading_session_days > 0 else 0.0
    hold_minutes = [float(t.get("minutes_held", 0.0) or 0.0) for t in trades]
    avg_hold_minutes = (sum(hold_minutes) / len(hold_minutes)) if hold_minutes else 0.0

    checks = dict(checks)
    feasible = (
        int(checks.get("max_hold_violation_count", 0)) == 0
        and int(checks.get("loss_cap_violation_count", 0)) == 0
        and int(checks.get("cross_session_violation_count", 0)) == 0
        and all(float(t.get("minutes_held", 0.0)) <= 90.0 + 1e-9 for t in trades)
        and all(float(t.get("pnl_usd", 0.0)) >= -500.0 - 1e-6 for t in trades)
    )

    return {
        "total_return_pct": float(total_return_pct),
        "max_drawdown_pct": float(max_drawdown_pct),
        "sharpe": float(sharpe),
        "win_rate_pct": float(win_rate_pct),
        "n_trades": int(len(trades)),
        "trading_session_days": int(trading_session_days),
        "winning_trades": int(wins),
        "losing_trades": int(losses),
        "total_pnl_usd": float(total_pnl_usd),
        "trades_per_day": float(trades_per_day),
        "avg_hold_minutes": float(avg_hold_minutes),
        "exit_reason_distribution": dict(checks.get("exit_reason_distribution", {})),
        "constraint_checks": checks,
        "feasible": bool(feasible),
    }


def _compute_metrics_from_sim(sim: dict[str, Any], strat_returns: list[float], timestamps: list[int]) -> dict[str, Any]:
    trades = list(sim["trades"])
    checks = dict(sim.get("constraint_checks", {}))
    return _compute_metrics_from_parts(trades=trades, strat_returns=strat_returns, timestamps=timestamps, checks=checks)


def _evaluate_segment(cfg: dict[str, Any], segment: dict[str, list[float] | list[int]], macro_bias: int) -> dict[str, Any]:
    close = list(segment["close"])
    timestamps = list(segment["timestamps"])
    returns = [(close[i] - close[i - 1]) / close[i - 1] for i in range(1, len(close))]

    raw = server._generate_live_signals_with_config(
        returns=returns,
        cfg=cfg,
        kx_confluence=None,
        macro_bias_override=macro_bias,
    )
    desired = server._apply_execution_controls_with_config(
        exec_signal=[0] + raw[:-1],
        returns=returns,
        timestamps=timestamps,
        cfg=cfg,
    )
    sim = server._simulate_exec_with_constraints(
        desired,
        timestamps=timestamps,
        open_px=list(segment["open"]),
        high_px=list(segment["high"]),
        low_px=list(segment["low"]),
        close_px=close,
        cfg=cfg,
    )
    strat_returns = list(sim["strat_returns"])
    metrics = _compute_metrics_from_sim(sim, strat_returns, timestamps=timestamps)
    return {
        "metrics": metrics,
        "trades": list(sim["trades"]),
        "strat_returns": strat_returns,
        "timestamps": timestamps,
    }


def _evaluate_config(
    cfg: dict[str, Any],
    full_data: dict[str, list[float] | list[int]],
    folds: list[tuple[int, int]],
    macro_bias: int,
) -> EvalResult:
    full_eval = _evaluate_segment(cfg, full_data, macro_bias)
    full_metrics = dict(full_eval["metrics"])
    full_trades = list(full_eval["trades"])
    full_returns = list(full_eval["strat_returns"])
    full_timestamps = list(full_eval["timestamps"])

    fold_metrics: list[dict[str, Any]] = []
    for idx, (start, end) in enumerate(folds, start=1):
        strat_slice = full_returns[start:end]
        ts_start = int(full_timestamps[start])
        ts_end = int(full_timestamps[end])
        fold_trades = [
            t
            for t in full_trades
            if int(float(t.get("entry_unix", 0) or 0)) >= ts_start and int(float(t.get("exit_unix", 0) or 0)) <= ts_end
        ]
        fold_checks = _constraint_checks_from_trades(fold_trades)
        fm = _compute_metrics_from_parts(
            trades=fold_trades,
            strat_returns=strat_slice,
            timestamps=full_timestamps[start : end + 1],
            checks=fold_checks,
        )
        fm = dict(fm)
        fm["fold_id"] = idx
        fm["start_idx"] = int(start)
        fm["end_idx"] = int(end)
        fold_metrics.append(fm)

    median_oos_sharpe = float(median([f["sharpe"] for f in fold_metrics])) if fold_metrics else 0.0
    median_oos_return = float(median([f["total_return_pct"] for f in fold_metrics])) if fold_metrics else 0.0
    full_metrics["median_oos_sharpe"] = median_oos_sharpe
    full_metrics["median_oos_return_pct"] = median_oos_return

    feasible = bool(full_metrics.get("feasible", False)) and all(bool(f.get("feasible", False)) for f in fold_metrics)
    return EvalResult(metrics=full_metrics, folds=fold_metrics, feasible=feasible)


def _rank_tuple(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    # Lexicographic: total return > total PnL > activity > win rate > lower drawdown.
    trades_per_day = float(metrics.get("trades_per_day", 0.0))
    activity_fit = -abs(trades_per_day - TARGET_TRADES_PER_DAY)
    return (
        float(metrics.get("total_return_pct", 0.0)),
        float(metrics.get("total_pnl_usd", 0.0)),
        float(activity_fit),
        float(metrics.get("win_rate_pct", 0.0)),
        float(metrics.get("max_drawdown_pct", -1e9)),
    )


def _build_folds(n_returns: int) -> list[tuple[int, int]]:
    # Rolling out-of-sample folds on the full 60d window.
    # Each fold evaluates the next 10% window after an expanding prefix.
    folds: list[tuple[int, int]] = []
    for k in range(4, 9):
        start = int(n_returns * (k / 10.0))
        end = min(n_returns, int(n_returns * ((k + 1) / 10.0)))
        if end - start >= 120:
            folds.append((start, end))
    if not folds:
        size = max(120, n_returns // 5)
        for i in range(0, n_returns - size + 1, size):
            folds.append((i, min(n_returns, i + size)))
    return folds


def _phase_name(i: int) -> str:
    if i <= 120:
        return "broad"
    if i <= 240:
        return "neighborhood"
    return "fine"


def _run_intrabar_tests(base_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(base_cfg)
    cfg.update(HARD_CONSTRAINTS)
    cfg["enforce_same_day_trades"] = False

    def run_case(direction: int, stop_hit_low: float, stop_hit_high: float, timestamps: list[int]) -> dict[str, Any]:
        close = [100.0] * len(timestamps)
        open_px = [100.0] * len(timestamps)
        high = [101.0] * len(timestamps)
        low = [99.0] * len(timestamps)
        high[1] = stop_hit_high
        low[1] = stop_hit_low

        desired = [direction] * (len(timestamps) - 1)
        sim = server._simulate_exec_with_constraints(
            desired,
            timestamps=timestamps,
            open_px=open_px,
            high_px=high,
            low_px=low,
            close_px=close,
            cfg=cfg,
        )
        trades = list(sim["trades"])
        return trades[0] if trades else {}

    # Long stop-touch test.
    long_trade = run_case(direction=1, stop_hit_low=74.0, stop_hit_high=101.0, timestamps=[0, 300, 600])
    # Short stop-touch test.
    short_trade = run_case(direction=-1, stop_hit_low=99.0, stop_hit_high=126.0, timestamps=[0, 300, 600])
    # Precedence test: stop and time-cap true on same bar -> loss_cap should win.
    precedence_trade = run_case(direction=1, stop_hit_low=74.0, stop_hit_high=101.0, timestamps=[0, 5400, 5700])

    return {
        "long_intrabar_stop": long_trade.get("exit_reason") == "loss_cap",
        "short_intrabar_stop": short_trade.get("exit_reason") == "loss_cap",
        "stop_over_time_precedence": precedence_trade.get("exit_reason") == "loss_cap",
        "samples": {
            "long": long_trade,
            "short": short_trade,
            "precedence": precedence_trade,
        },
    }


def _table_line(label: str, baseline: dict[str, Any], best: dict[str, Any], key: str, fmt: str = ".2f") -> str:
    b = float(baseline.get(key, 0.0))
    w = float(best.get(key, 0.0))
    return f"| {label} | {format(b, fmt)} | {format(w, fmt)} | {format(w - b, fmt)} |"


def main() -> None:
    random.seed(173)
    np.random.seed(173)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = SNAPSHOT_DIR / f"server.py.{stamp}.bak"
    shutil.copy2(SERVER_FILE, snapshot_path)

    base_cfg = dict(server.LIVE_SIGNAL_CONFIG)
    base_cfg.update(HARD_CONSTRAINTS)
    base_cfg = _sanitize(base_cfg)

    bars = server._fetch_symbol_bars_ohlc("NQ=F", interval="5m", period="60d")
    full_data: dict[str, list[float] | list[int]] = {
        "timestamps": list(bars["timestamp"]),
        "open": list(bars["open"]),
        "high": list(bars["high"]),
        "low": list(bars["low"]),
        "close": list(bars["close"]),
    }

    n_returns = len(full_data["close"]) - 1
    folds = _build_folds(n_returns)

    try:
        macro_bias = int(server._pdf_daily_trend_context().get("bias", 0))
    except Exception:
        macro_bias = 0

    intrabar_tests = _run_intrabar_tests(base_cfg)

    baseline = _evaluate_config(base_cfg, full_data, folds, macro_bias)
    baseline_payload = {
        "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
        "window": "60d",
        "macro_bias": macro_bias,
        "constraints": HARD_CONSTRAINTS,
        "folds": baseline.folds,
        "metrics": baseline.metrics,
        "feasible": baseline.feasible,
        "params": {k: base_cfg[k] for k in TUNABLE_KEYS},
        "intrabar_tests": intrabar_tests,
    }
    (RESULTS_DIR / "baseline_60d.json").write_text(json.dumps(baseline_payload, indent=2), encoding="utf-8")

    cache: dict[tuple, EvalResult] = {}
    evaluated: list[dict[str, Any]] = []
    feasible_pool: list[dict[str, Any]] = []

    def evaluate_candidate(cfg: dict[str, Any], iteration: int, phase: str) -> dict[str, Any]:
        key = _param_key(cfg)
        if key in cache:
            eval_result = cache[key]
        else:
            eval_result = _evaluate_config(cfg, full_data, folds, macro_bias)
            cache[key] = eval_result

        row = {
            "iteration": int(iteration),
            "phase": phase,
            "feasible": bool(eval_result.feasible),
            "metrics": eval_result.metrics,
            "folds": eval_result.folds,
            "params": {k: cfg[k] for k in TUNABLE_KEYS},
        }
        evaluated.append(row)
        if eval_result.feasible:
            feasible_pool.append(row)
        return row

    # Seed the pool with baseline.
    baseline_row = {
        "iteration": 0,
        "phase": "baseline",
        "feasible": bool(baseline.feasible),
        "metrics": baseline.metrics,
        "folds": baseline.folds,
        "params": {k: base_cfg[k] for k in TUNABLE_KEYS},
    }
    evaluated.append(baseline_row)
    if baseline.feasible:
        feasible_pool.append(baseline_row)

    seen_param_keys: set[tuple] = {_param_key(base_cfg)}

    for i in range(1, 301):
        phase = _phase_name(i)

        if phase == "broad" or not feasible_pool:
            candidate = _random_candidate(base_cfg)
        elif phase == "neighborhood":
            ranked = sorted(feasible_pool, key=lambda r: _rank_tuple(r["metrics"]), reverse=True)
            top_decile = ranked[: max(1, len(ranked) // 10)]
            parent = random.choice(top_decile)["params"]
            full_parent = dict(base_cfg)
            full_parent.update(parent)
            candidate = _mutate(full_parent, strength=0.45)
        else:
            ranked = sorted(feasible_pool, key=lambda r: _rank_tuple(r["metrics"]), reverse=True)
            top3 = ranked[: max(1, min(3, len(ranked)))]
            parent = random.choice(top3)["params"]
            full_parent = dict(base_cfg)
            full_parent.update(parent)
            candidate = _mutate(full_parent, strength=0.18)

        candidate = _sanitize(candidate)
        key = _param_key(candidate)
        attempts = 0
        while key in seen_param_keys and attempts < 25:
            candidate = _mutate(candidate, strength=0.35 if phase != "fine" else 0.14)
            candidate = _sanitize(candidate)
            key = _param_key(candidate)
            attempts += 1
        seen_param_keys.add(key)

        evaluate_candidate(candidate, iteration=i, phase=phase)

    # Persist full search log.
    search_log_path = RESULTS_DIR / "search_log_300.jsonl"
    with search_log_path.open("w", encoding="utf-8") as fh:
        for row in evaluated:
            fh.write(json.dumps(row) + "\n")

    ranked_feasible = sorted(feasible_pool, key=lambda r: _rank_tuple(r["metrics"]), reverse=True)
    ranked_all = sorted(evaluated, key=lambda r: _rank_tuple(r["metrics"]), reverse=True)

    top10 = ranked_feasible[:10] if ranked_feasible else ranked_all[:10]
    (RESULTS_DIR / "top10_60d.json").write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
                "window": "60d",
                "selection_policy": "total_return > total_pnl > activity_fit(1/day) > win_rate > max_drawdown",
                "count": len(top10),
                "rows": top10,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    winner = ranked_feasible[0] if ranked_feasible else baseline_row
    baseline_metrics = baseline_row["metrics"]
    selected = winner
    selection_reason = (
        "Selected top-ranked feasible candidate under return-first ranking "
        "(return, PnL, then activity; hard risk/session constraints unchanged)."
    )
    activity_fit_baseline = -abs(float(baseline_metrics.get("trades_per_day", 0.0)) - TARGET_TRADES_PER_DAY)
    activity_fit_selected = -abs(float(selected["metrics"].get("trades_per_day", 0.0)) - TARGET_TRADES_PER_DAY)

    final_cfg = dict(base_cfg)
    final_cfg.update(selected["params"])
    final_cfg = _sanitize(final_cfg)

    pre_apply_snapshot = SNAPSHOT_DIR / f"server.py.pre_apply.{stamp}.bak"
    shutil.copy2(SERVER_FILE, pre_apply_snapshot)
    _apply_live_config(final_cfg)

    final_selection_md = [
        "# Final Selection (60d NQ, 5m)",
        "",
        "## Policy",
        "- Hard constraints: max hold 90m, intrabar -$500 stop-touch, same-session-day only",
        f"- Activity target: ~{TARGET_TRADES_PER_DAY:.1f} trades/day (ranking hint, not a hard reject gate)",
        "- Ranking: total return > total PnL > activity fit > win rate > max drawdown",
        "",
        "## Baseline vs Selected",
        "",
        "| Metric | Baseline | Selected | Delta |",
        "|---|---:|---:|---:|",
        _table_line("Win Rate %", baseline_metrics, selected["metrics"], "win_rate_pct"),
        _table_line("Total Return %", baseline_metrics, selected["metrics"], "total_return_pct"),
        _table_line("Total PnL $", baseline_metrics, selected["metrics"], "total_pnl_usd"),
        _table_line("Sharpe", baseline_metrics, selected["metrics"], "sharpe"),
        _table_line("Max Drawdown %", baseline_metrics, selected["metrics"], "max_drawdown_pct"),
        _table_line("Trades/Day", baseline_metrics, selected["metrics"], "trades_per_day"),
        _table_line("Avg Hold Mins", baseline_metrics, selected["metrics"], "avg_hold_minutes"),
        "",
        "## Constraint Checks (Selected)",
        f"- max_hold_violation_count: {int(selected['metrics']['constraint_checks'].get('max_hold_violation_count', 0))}",
        f"- loss_cap_violation_count: {int(selected['metrics']['constraint_checks'].get('loss_cap_violation_count', 0))}",
        f"- cross_session_violation_count: {int(selected['metrics']['constraint_checks'].get('cross_session_violation_count', 0))}",
        f"- exit_reason_distribution: {json.dumps(selected['metrics'].get('exit_reason_distribution', {}), sort_keys=True)}",
        "",
        "## Fold Metrics (Selected)",
    ]

    for f in selected["folds"]:
        final_selection_md.append(
            f"- Fold {f['fold_id']}: Sharpe {f['sharpe']:.3f}, Return {f['total_return_pct']:.3f}%, "
            f"WinRate {f['win_rate_pct']:.2f}%, Trades {f['n_trades']}"
        )

    final_selection_md.extend(
        [
            "",
            "## Intrabar Tests",
            f"- long_intrabar_stop: {intrabar_tests['long_intrabar_stop']}",
            f"- short_intrabar_stop: {intrabar_tests['short_intrabar_stop']}",
            f"- stop_over_time_precedence: {intrabar_tests['stop_over_time_precedence']}",
            "",
            "## Rationale",
            f"- {selection_reason}",
            f"- baseline_activity_fit: {activity_fit_baseline:.4f}",
            f"- selected_activity_fit: {activity_fit_selected:.4f}",
            "",
            "## Artifacts",
            "- results/baseline_60d.json",
            "- results/search_log_300.jsonl",
            "- results/top10_60d.json",
            "- results/final_selection.md",
        ]
    )

    (RESULTS_DIR / "final_selection.md").write_text("\n".join(final_selection_md) + "\n", encoding="utf-8")

    summary = {
        "snapshot": str(snapshot_path),
        "pre_apply_snapshot": str(pre_apply_snapshot),
        "selected_metrics": selected["metrics"],
        "selected_params": selected["params"],
        "baseline_metrics": baseline_metrics,
        "baseline_params": baseline_row["params"],
        "baseline_activity_fit": activity_fit_baseline,
        "selected_activity_fit": activity_fit_selected,
        "selection_reason": selection_reason,
        "artifacts": {
            "baseline": str(RESULTS_DIR / "baseline_60d.json"),
            "search_log": str(search_log_path),
            "top10": str(RESULTS_DIR / "top10_60d.json"),
            "final_selection": str(RESULTS_DIR / "final_selection.md"),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
