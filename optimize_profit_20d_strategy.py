#!/usr/bin/env python3
"""Snapshot optimizer for 20d NQ profitability with bounded trade cadence.

This script:
1) Uses cached 20d 5m bars from /tmp/nq_20d.json.
2) Runs 3 optimization rounds with snapshot + rollback.
3) Applies LIVE_SIGNAL_CONFIG only when a round improves profitability score.
"""

from __future__ import annotations

import json
import math
import random
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import server


ROOT = Path(__file__).resolve().parents[1]
SERVER_FILE = ROOT / "server.py"
SNAPSHOT_DIR = ROOT / "snapshots"
CACHE_BARS_FILE = Path("/tmp/nq_20d.json")

PARAM_KEYS = [
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
    "use_kx_confluence",
    "kx_confluence_strength",
    "kx_quality_floor",
]

M_WINDOWS = [8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40]
V_WINDOWS = [8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 48, 56]
T_WINDOWS = [16, 24, 32, 40, 48, 64, 80, 96, 120, 144, 180, 220, 280, 340]


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _log_uniform(lo: float, hi: float) -> float:
    return 10 ** random.uniform(math.log10(lo), math.log10(hi))


def _roll_mean(vals: np.ndarray, window: int) -> np.ndarray:
    c = np.cumsum(np.insert(vals, 0, 0.0))
    out = np.empty(len(vals), dtype=float)
    for i in range(len(vals)):
        s = max(0, i - window + 1)
        out[i] = (c[i + 1] - c[s]) / (i - s + 1)
    return out


def _roll_std(vals: np.ndarray, window: int) -> np.ndarray:
    out = np.empty(len(vals), dtype=float)
    for i in range(len(vals)):
        s = max(0, i - window + 1)
        seg = vals[s : i + 1]
        out[i] = np.std(seg, ddof=1) if len(seg) > 1 else 0.0
    return out


def _load_cached_bars() -> tuple[list[int], list[float]]:
    if not CACHE_BARS_FILE.exists():
        raise RuntimeError(f"Missing cache file: {CACHE_BARS_FILE}")
    payload = json.loads(CACHE_BARS_FILE.read_text(encoding="utf-8"))
    result = payload.get("chart", {}).get("result", [None])[0]
    if not result:
        raise RuntimeError("No chart result in cached 20d payload")
    timestamps = result.get("timestamp", [])
    closes = result.get("indicators", {}).get("quote", [{}])[0].get("close", [])
    rows = [(int(ts), float(c)) for ts, c in zip(timestamps, closes) if c is not None]
    if len(rows) < 1200:
        raise RuntimeError(f"Insufficient cached bars ({len(rows)}); expected 20d 5m sample")
    return [r[0] for r in rows], [r[1] for r in rows]


def _sanitize(params: dict[str, float]) -> dict[str, float]:
    out = dict(params)
    out["momentum_window"] = int(out.get("momentum_window", random.choice(M_WINDOWS)))
    out["volatility_window"] = int(out.get("volatility_window", random.choice(V_WINDOWS)))
    out["trend_window"] = int(out.get("trend_window", random.choice(T_WINDOWS)))
    if out["momentum_window"] not in M_WINDOWS:
        out["momentum_window"] = random.choice(M_WINDOWS)
    if out["volatility_window"] not in V_WINDOWS:
        out["volatility_window"] = random.choice(V_WINDOWS)
    if out["trend_window"] not in T_WINDOWS:
        out["trend_window"] = random.choice(T_WINDOWS)

    out["momentum_threshold"] = float(_clamp(float(out.get("momentum_threshold", 1e-4)), 1e-5, 4.5e-4))
    out["volatility_quantile_cap"] = float(_clamp(float(out.get("volatility_quantile_cap", 0.58)), 0.30, 0.97))
    out["countertrend_multiplier"] = float(_clamp(float(out.get("countertrend_multiplier", 2.0)), 1.0, 5.0))
    out["min_hold_bars"] = int(_clamp(float(out.get("min_hold_bars", 24)), 8, 64))
    out["macro_countertrend_allow_multiplier"] = float(
        _clamp(float(out.get("macro_countertrend_allow_multiplier", 4.2)), 0.8, 6.5)
    )
    out["trend_regime_multiplier"] = float(_clamp(float(out.get("trend_regime_multiplier", 1.2)), 0.1, 2.6))
    out["neutral_regime_multiplier"] = float(_clamp(float(out.get("neutral_regime_multiplier", 1.5)), 0.4, 4.5))
    out["high_volatility_multiplier"] = float(_clamp(float(out.get("high_volatility_multiplier", 0.9)), 0.45, 2.0))
    out["short_entry_multiplier"] = float(_clamp(float(out.get("short_entry_multiplier", 1.0)), 0.4, 2.3))
    out["disable_longs_when_macro_short"] = bool(out.get("disable_longs_when_macro_short", False))
    out["use_kx_confluence"] = bool(out.get("use_kx_confluence", True))
    out["kx_confluence_strength"] = float(_clamp(float(out.get("kx_confluence_strength", 1.0)), 0.0, 1.5))
    out["kx_quality_floor"] = float(_clamp(float(out.get("kx_quality_floor", 0.05)), 0.0, 0.2))
    return out


def _param_key(p: dict[str, float]) -> tuple:
    return (
        int(p["momentum_window"]),
        int(p["volatility_window"]),
        int(p["trend_window"]),
        round(float(p["momentum_threshold"]), 9),
        round(float(p["volatility_quantile_cap"]), 6),
        round(float(p["countertrend_multiplier"]), 6),
        int(p["min_hold_bars"]),
        round(float(p["macro_countertrend_allow_multiplier"]), 6),
        round(float(p["trend_regime_multiplier"]), 6),
        round(float(p["neutral_regime_multiplier"]), 6),
        round(float(p["high_volatility_multiplier"]), 6),
        round(float(p["short_entry_multiplier"]), 6),
        bool(p["disable_longs_when_macro_short"]),
        bool(p["use_kx_confluence"]),
        round(float(p["kx_confluence_strength"]), 6),
        round(float(p["kx_quality_floor"]), 6),
    )


def _format_live_config(params: dict[str, float]) -> str:
    return (
        "LIVE_SIGNAL_CONFIG = {\n"
        f'    "momentum_window": {int(params["momentum_window"])},\n'
        f'    "volatility_window": {int(params["volatility_window"])},\n'
        f'    "trend_window": {int(params["trend_window"])},\n'
        f'    "momentum_threshold": {float(params["momentum_threshold"])},\n'
        f'    "volatility_quantile_cap": {float(params["volatility_quantile_cap"])},\n'
        f'    "countertrend_multiplier": {float(params["countertrend_multiplier"])},\n'
        f'    "min_hold_bars": {int(params["min_hold_bars"])},\n'
        f'    "macro_countertrend_allow_multiplier": {float(params["macro_countertrend_allow_multiplier"])},\n'
        f'    "trend_regime_multiplier": {float(params["trend_regime_multiplier"])},\n'
        f'    "neutral_regime_multiplier": {float(params["neutral_regime_multiplier"])},\n'
        f'    "high_volatility_multiplier": {float(params["high_volatility_multiplier"])},\n'
        f'    "short_entry_multiplier": {float(params["short_entry_multiplier"])},\n'
        f'    "disable_longs_when_macro_short": {bool(params["disable_longs_when_macro_short"])},\n'
        f'    "use_kx_confluence": {bool(params["use_kx_confluence"])},\n'
        f'    "kx_confluence_strength": {float(params["kx_confluence_strength"])},\n'
        f'    "kx_quality_floor": {float(params["kx_quality_floor"])},\n'
        "}\n"
    )


def _apply_live_config(params: dict[str, float]) -> None:
    src = SERVER_FILE.read_text(encoding="utf-8")
    repl = _format_live_config(params).rstrip()
    out = re.sub(r"LIVE_SIGNAL_CONFIG = \{[\s\S]*?\n\}", repl, src, count=1)
    if out == src:
        raise RuntimeError("LIVE_SIGNAL_CONFIG block not found")
    SERVER_FILE.write_text(out, encoding="utf-8")


def _random_params() -> dict[str, float]:
    return {
        "momentum_window": random.choice(M_WINDOWS),
        "volatility_window": random.choice(V_WINDOWS),
        "trend_window": random.choice(T_WINDOWS),
        "momentum_threshold": _log_uniform(1e-5, 4.2e-4),
        "volatility_quantile_cap": random.uniform(0.34, 0.94),
        "countertrend_multiplier": random.uniform(1.0, 4.8),
        "min_hold_bars": random.randint(8, 60),
        "macro_countertrend_allow_multiplier": random.uniform(0.9, 6.2),
        "trend_regime_multiplier": random.uniform(0.15, 2.4),
        "neutral_regime_multiplier": random.uniform(0.5, 4.2),
        "high_volatility_multiplier": random.uniform(0.5, 1.9),
        "short_entry_multiplier": random.uniform(0.45, 2.2),
        "disable_longs_when_macro_short": random.choice([True, False]),
        "use_kx_confluence": random.choice([True, False]),
        "kx_confluence_strength": random.uniform(0.25, 1.35),
        "kx_quality_floor": random.uniform(0.0, 0.12),
    }


def _mutate(center: dict[str, float], strength: float = 0.28) -> dict[str, float]:
    out = dict(center)
    if random.random() < 0.42:
        out["momentum_window"] = random.choice(M_WINDOWS)
    if random.random() < 0.42:
        out["volatility_window"] = random.choice(V_WINDOWS)
    if random.random() < 0.44:
        out["trend_window"] = random.choice(T_WINDOWS)
    out["momentum_threshold"] *= math.exp(random.gauss(0.0, 0.45 * strength))
    out["volatility_quantile_cap"] += random.gauss(0.0, 0.12 * strength)
    out["countertrend_multiplier"] += random.gauss(0.0, 0.50 * strength)
    out["min_hold_bars"] += int(round(random.gauss(0.0, 4.0 * strength)))
    out["macro_countertrend_allow_multiplier"] += random.gauss(0.0, 0.65 * strength)
    out["trend_regime_multiplier"] += random.gauss(0.0, 0.26 * strength)
    out["neutral_regime_multiplier"] += random.gauss(0.0, 0.48 * strength)
    out["high_volatility_multiplier"] += random.gauss(0.0, 0.18 * strength)
    out["short_entry_multiplier"] += random.gauss(0.0, 0.24 * strength)
    if random.random() < 0.08:
        out["disable_longs_when_macro_short"] = not bool(out["disable_longs_when_macro_short"])
    if random.random() < 0.08:
        out["use_kx_confluence"] = not bool(out["use_kx_confluence"])
    out["kx_confluence_strength"] += random.gauss(0.0, 0.14 * strength)
    out["kx_quality_floor"] += random.gauss(0.0, 0.02 * strength)
    return _sanitize(out)


def main() -> None:
    random.seed(4207)
    np.random.seed(4207)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    timestamps, close = _load_cached_bars()
    returns = np.array([(close[i] - close[i - 1]) / close[i - 1] for i in range(1, len(close))], dtype=float)
    if len(returns) < 1200:
        raise RuntimeError("Insufficient returns for 20d optimization")
    bar_ts = timestamps[1:]

    # Current macro regime in dashboard context has been short.
    macro_bias = -1

    mom_map = {w: _roll_mean(returns, w) for w in M_WINDOWS}
    vol_map = {w: _roll_std(returns, w) for w in V_WINDOWS}
    trd_map = {w: _roll_mean(returns, w) for w in T_WINDOWS}

    day_count = max(
        1,
        len({datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(server.ET_TZ).date() for ts in bar_ts}),
    )
    sessions_equiv = max(1e-9, len(returns) / server.BARS_PER_SESSION)

    def simulate(params: dict[str, float]) -> dict[str, float]:
        p = _sanitize(params)
        mw = int(p["momentum_window"])
        vw = int(p["volatility_window"])
        tw = int(p["trend_window"])
        mthr = float(p["momentum_threshold"])
        vq = float(p["volatility_quantile_cap"])
        ctr = float(p["countertrend_multiplier"])
        min_hold = int(p["min_hold_bars"])
        allow = float(p["macro_countertrend_allow_multiplier"])
        trend_k = float(p["trend_regime_multiplier"])
        neutral_k = float(p["neutral_regime_multiplier"])
        highv_k = float(p["high_volatility_multiplier"])
        short_k = float(p["short_entry_multiplier"])
        no_long = bool(p["disable_longs_when_macro_short"])

        mom = mom_map[mw]
        vol = vol_map[vw]
        trd = trd_map[tw]
        vcap = float(np.quantile(vol, vq))

        sig = np.zeros(len(returns), dtype=int)
        pos = 0
        held = 0
        for i, (m, v, t) in enumerate(zip(mom, vol, trd)):
            s = 0
            if v <= vcap * highv_k:
                if t <= -mthr * trend_k:
                    if m <= -mthr * short_k:
                        s = -1
                    elif m >= mthr * ctr:
                        s = 1
                elif t >= mthr * trend_k:
                    if m >= mthr:
                        s = 1
                    elif m <= -mthr * ctr:
                        s = -1
                elif abs(m) >= mthr * neutral_k:
                    s = 1 if m > 0 else -1

            if macro_bias < 0:
                if no_long and s > 0:
                    s = 0
                elif s > 0 and m < mthr * allow:
                    s = 0
            elif macro_bias > 0 and s < 0 and m > -mthr * allow:
                s = 0

            if pos != 0 and held < min_hold and s != -pos:
                s = pos
                held += 1
            elif s != pos:
                pos = s
                held = 1 if s != 0 else 0
            else:
                held = held + 1 if s != 0 else 0
            sig[i] = s

        ex = np.empty_like(sig)
        ex[0] = 0
        ex[1:] = sig[:-1]
        c = 2.5 / 10000.0
        prev = 0
        trades = 0
        rs = np.empty_like(returns)
        for i, (s, r) in enumerate(zip(ex, returns)):
            switched = s != prev
            if switched:
                trades += 1
            rs[i] = s * r - (c if switched else 0.0)
            prev = s

        eq = np.cumprod(1 + rs)
        dd = float(np.min(eq / np.maximum.accumulate(eq) - 1.0) * 100.0)
        total = float((eq[-1] - 1.0) * 100.0)
        st = float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0
        sh = float((np.mean(rs) / st) * math.sqrt(252 * server.BARS_PER_SESSION)) if st > 1e-9 else 0.0
        win = float(np.mean(rs > 0) * 100.0)
        tpd = trades / day_count
        tps = trades / sessions_equiv
        recent_sig = sig[-120:]
        buy = int(np.sum(recent_sig == 1))
        sell = int(np.sum(recent_sig == -1))
        hold = int(np.sum(recent_sig == 0))
        return {
            "total": total,
            "dd": dd,
            "sh": sh,
            "tr": int(trades),
            "win": win,
            "trades_per_day": float(tpd),
            "trades_per_session": float(tps),
            "buy": buy,
            "sell": sell,
            "hold": hold,
            "score": 0.0,
        }

    current = _sanitize({k: server.LIVE_SIGNAL_CONFIG[k] for k in PARAM_KEYS})
    rounds: list[dict[str, object]] = []

    for rnd in range(1, 4):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot = SNAPSHOT_DIR / f"server.py.{stamp}.profit20d.round{rnd}.bak"
        shutil.copy2(SERVER_FILE, snapshot)

        base = simulate(current)
        target_tps = base["trades_per_session"]
        min_tps = max(0.55, target_tps - 0.22)
        max_tps = min(1.65, target_tps + 0.38)

        def score(m: dict[str, float]) -> float:
            freq_pen = abs(m["trades_per_session"] - target_tps)
            s = 2.15 * m["total"] + 0.24 * m["sh"] + 0.22 * m["dd"] - 1.15 * freq_pen
            if m["trades_per_session"] < min_tps:
                s -= (min_tps - m["trades_per_session"]) * 5.0
            if m["trades_per_session"] > max_tps:
                s -= (m["trades_per_session"] - max_tps) * 5.0
            if m["total"] < 0:
                s -= 8.0 + abs(m["total"]) * 1.4
            if m["dd"] < -8.0:
                s -= abs(m["dd"] + 8.0) * 1.8
            if m["sh"] < 0:
                s -= 1.8 + abs(m["sh"])
            return float(s)

        base["score"] = score(base)
        best_p = dict(current)
        best_m = dict(base)
        cache: dict[tuple, dict[str, float]] = {}

        seed = 12000 + rnd * 211
        random.seed(seed)
        np.random.seed(seed)

        def eval_params(p: dict[str, float]) -> dict[str, float]:
            sp = _sanitize(p)
            k = _param_key(sp)
            if k in cache:
                return cache[k]
            m = simulate(sp)
            m["score"] = score(m)
            cache[k] = m
            return m

        for _ in range(9000):
            cand = _random_params()
            m = eval_params(cand)
            if m["score"] > best_m["score"]:
                best_p = _sanitize(cand)
                best_m = dict(m)
        for _ in range(6000):
            cand = _mutate(best_p, strength=0.26)
            m = eval_params(cand)
            if m["score"] > best_m["score"]:
                best_p = _sanitize(cand)
                best_m = dict(m)
        for _ in range(3500):
            cand = _mutate(current, strength=0.18)
            m = eval_params(cand)
            if m["score"] > best_m["score"]:
                best_p = _sanitize(cand)
                best_m = dict(m)

        improved = (
            best_m["total"] > base["total"] + 0.06
            and best_m["trades_per_session"] >= min_tps
            and best_m["trades_per_session"] <= max_tps
            and best_m["dd"] >= base["dd"] - 0.90
            and best_m["score"] > base["score"] + 0.03
        )

        if improved:
            _apply_live_config(best_p)
            current = dict(best_p)
            action = "applied"
        else:
            shutil.copy2(snapshot, SERVER_FILE)
            action = "rolled_back"

        rounds.append(
            {
                "round": rnd,
                "seed": seed,
                "snapshot": str(snapshot),
                "base": base,
                "best": best_m,
                "improved": bool(improved),
                "action": action,
                "best_params": best_p,
                "trade_freq_band_tps": [min_tps, max_tps],
            }
        )

    final_metrics = simulate(current)
    summary = {
        "objective": "maximize 20d profitability while keeping trade cadence bounded",
        "cache_file": str(CACHE_BARS_FILE),
        "rounds": rounds,
        "final_params": current,
        "final_metrics": final_metrics,
    }
    out = SNAPSHOT_DIR / f"optimization_profit20d_3rounds.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"result_file={out}")


if __name__ == "__main__":
    main()
