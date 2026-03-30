#!/usr/bin/env python3
"""Local dashboard server with live NQ data + backtest metrics.

Routes:
- /api/dashboard : JSON payload with live NQ quant metrics
- /             : static React app
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
import time
from datetime import datetime, time as dt_time, timedelta, timezone
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
QPO_DIR = WORKSPACE_DIR / "quantitative-portfolio-optimization"
QPO_SRC_DIR = QPO_DIR / "src"
KX_DIR = WORKSPACE_DIR / "nvidia-kx-samples" / "ai-model-distillation-for-financial-data"
KX_REPO_DIR = BASE_DIR / "data" / "kx"
KX_CONFIG_FILE = (
    KX_DIR / "config" / "config.yaml"
    if (KX_DIR / "config" / "config.yaml").exists()
    else KX_REPO_DIR / "config.yaml"
)
KX_FINGPT_FILE = (
    KX_DIR / "data" / "fingpt_sentiment_1k.jsonl"
    if (KX_DIR / "data" / "fingpt_sentiment_1k.jsonl").exists()
    else KX_REPO_DIR / "fingpt_sentiment_1k.jsonl"
)
KX_TEST_FILE = (
    KX_DIR / "data" / "test_financial_data.jsonl"
    if (KX_DIR / "data" / "test_financial_data.jsonl").exists()
    else KX_REPO_DIR / "test_financial_data.jsonl"
)
TRADE_LOG_DIR = BASE_DIR / "trade_logs"
TRADE_JOURNAL_JSONL = TRADE_LOG_DIR / "nq_trade_journal.jsonl"
TRADE_JOURNAL_CSV = TRADE_LOG_DIR / "nq_trade_journal.csv"
TRADE_HISTORY_JSONL = TRADE_LOG_DIR / "nq_trade_history.jsonl"
TRADE_HISTORY_CSV = TRADE_LOG_DIR / "nq_trade_history.csv"

PORT = int(os.environ.get("PORT", "8080"))
ACCOUNT_SIZE = 50_000
BAR_INTERVAL_MIN = 5
BARS_PER_SESSION = 78
NQ_TICK_SIZE = 0.25
NQ_POINT_VALUE_USD = 20.0
MNQ_POINT_VALUE_USD = 2.0
BACKTEST_RANGE_5M = os.environ.get("BACKTEST_RANGE_5M", "20d")
QPO_RANGE_5M = os.environ.get("QPO_RANGE_5M", BACKTEST_RANGE_5M)
FREEZE_TO_LAST_CLOSED_SESSION = os.environ.get("FREEZE_TO_LAST_CLOSED_SESSION", "0").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
try:
    SIGNAL_MIX_LOOKBACK_BARS = max(30, int(os.environ.get("SIGNAL_MIX_LOOKBACK_BARS", "120")))
except ValueError:
    SIGNAL_MIX_LOOKBACK_BARS = 120
try:
    # 0 means unlimited retention.
    TRADE_JOURNAL_MAX_ROWS = max(0, int(os.environ.get("TRADE_JOURNAL_MAX_ROWS", "0")))
except ValueError:
    TRADE_JOURNAL_MAX_ROWS = 0
try:
    # 0 means unlimited retention.
    TRADE_HISTORY_MAX_ROWS = max(0, int(os.environ.get("TRADE_HISTORY_MAX_ROWS", "0")))
except ValueError:
    TRADE_HISTORY_MAX_ROWS = 0

# Tuned on recent 5m NQ bars to reduce churn and improve risk-adjusted behavior.
LIVE_SIGNAL_CONFIG = {
    "momentum_window": 28,
    "volatility_window": 36,
    "trend_window": 40,
    "momentum_threshold": 0.00018438018909355784,
    "volatility_quantile_cap": 0.678158976884079,
    "countertrend_multiplier": 3.750302668873535,
    "min_hold_bars": 2,
    "macro_countertrend_allow_multiplier": 2.2820783813623478,
    "trend_regime_multiplier": 0.8330451096715936,
    "neutral_regime_multiplier": 0.8593827413076662,
    "high_volatility_multiplier": 0.8105921662713327,
    "short_entry_multiplier": 2.141057748532962,
    "disable_longs_when_macro_short": True,
    "use_kx_confluence": False,
    "kx_confluence_strength": 0.42450751038322526,
    "kx_quality_floor": 0.08502140842228272,
    "trade_cooldown_bars": 6,
    "use_daily_loss_guard": True,
    "daily_loss_limit_pct": 0.8905184413162852,
    "max_trades_per_day": 12,
    "enforce_same_day_trades": True,
    "max_hold_minutes": 90,
    "max_active_trade_loss_usd": 500.0,
    "intrabar_stop_touch": True,
}

PDF_TREND_CONFIG = {
    "momentum_lookback_days": 63,
    "vol_lookback_days": 20,
    "target_ann_vol": 0.12,
    "max_leverage": 1.5,
}

PROP_EXECUTION_CONFIG = {
    "risk_per_trade_pct": 0.005,
    "daily_loss_limit_pct": 0.02,
    "max_trades_per_day": 12,
    "max_nq_contracts": 3,
    "max_mnq_contracts": 30,
    "target_rr": 1.6,
}
TRADING_SESSION_OPEN_ET = dt_time(18, 0, 0)
TRADING_SESSION_FLAT_TIME_ET = dt_time(16, 55, 0)


try:
    from zoneinfo import ZoneInfo

    ET_TZ = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    ET_TZ = timezone.utc


def _now_et() -> datetime:
    return datetime.now(tz=timezone.utc).astimezone(ET_TZ)


def _format_et(ts: int | float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(ET_TZ).strftime("%H:%M:%S ET")


def _format_et_trade(ts: int | float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(ET_TZ).strftime("%Y-%m-%d %H:%M:%S ET")


def _trading_session_id(et_dt: datetime) -> str:
    session_date = et_dt.date()
    if et_dt.time() >= TRADING_SESSION_OPEN_ET:
        session_date = session_date + timedelta(days=1)
    return session_date.isoformat()


def _is_within_trading_session_window(et_dt: datetime) -> bool:
    t = et_dt.time()
    # NQ futures session day: opens 18:00 ET (prior calendar day), closes 17:00 ET.
    # We flatten by 16:55 ET to avoid carrying into the close/maintenance break.
    return bool(t >= TRADING_SESSION_OPEN_ET or t < TRADING_SESSION_FLAT_TIME_ET)


def _resolve_evaluation_end_index(timestamps: list[int]) -> int:
    """Pick the last fully-closed weekday regular session bar when freeze mode is enabled."""
    if not timestamps:
        return -1
    if not FREEZE_TO_LAST_CLOSED_SESSION:
        return len(timestamps) - 1

    latest_dt = datetime.fromtimestamp(timestamps[-1], tz=timezone.utc).astimezone(ET_TZ)
    close_time = dt_time(16, 59, 59)
    open_time = dt_time(9, 30, 0)

    session_date = latest_dt.date()
    if latest_dt.weekday() >= 5 or latest_dt.time() < close_time:
        session_date -= timedelta(days=1)
    while session_date.weekday() >= 5:
        session_date -= timedelta(days=1)

    idx = -1
    for i, ts in enumerate(timestamps):
        et_dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(ET_TZ)
        if et_dt.date() == session_date and open_time <= et_dt.time() <= close_time:
            idx = i

    if idx >= 0:
        return idx

    target_close = datetime.combine(session_date, close_time, tzinfo=ET_TZ).astimezone(timezone.utc).timestamp()
    for i, ts in enumerate(timestamps):
        if ts <= target_close:
            idx = i
    return idx if idx >= 0 else len(timestamps) - 1


def _utc_iso_to_et_label(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ET_TZ).strftime("%Y-%m-%d %H:%M:%S ET")


def _format_et_short(ts: int | float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(ET_TZ).strftime("%H:%M")


def _round_to_tick(price: float, tick: float = NQ_TICK_SIZE) -> float:
    if tick <= 0:
        return price
    return round(price / tick) * tick


def _load_trade_journal_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not TRADE_JOURNAL_JSONL.exists():
        return rows
    try:
        for raw in TRADE_JOURNAL_JSONL.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                execute_at_unix = 0
                try:
                    execute_at_unix = int(float(row.get("execute_at_unix", 0) or 0))
                except (TypeError, ValueError):
                    execute_at_unix = 0
                if execute_at_unix > 0:
                    row["execute_at_et"] = _format_et_trade(execute_at_unix)
                logged_at_et = _utc_iso_to_et_label(row.get("logged_at_utc"))
                if logged_at_et:
                    row["logged_at_et"] = logged_at_et
                rows.append(row)
    except Exception:
        return []
    return rows


def _write_trade_journal(rows: list[dict[str, Any]]) -> None:
    TRADE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    trimmed = rows[-TRADE_JOURNAL_MAX_ROWS:] if TRADE_JOURNAL_MAX_ROWS > 0 else rows

    with TRADE_JOURNAL_JSONL.open("w", encoding="utf-8") as fh:
        for row in trimmed:
            fh.write(json.dumps(row) + "\n")

    fieldnames = [
        "event_id",
        "logged_at_et",
        "logged_at_utc",
        "signal_time_et",
        "execute_at_et",
        "execute_at_unix",
        "action",
        "entry_reference",
        "stop_price",
        "target_price",
        "nq_contracts",
        "mnq_contracts",
        "risk_per_trade_usd",
        "sleeve_weight",
        "eligible",
        "signal_changed",
        "notes",
    ]
    with TRADE_JOURNAL_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in trimmed:
            writer.writerow(row)


def _append_trade_journal(entry: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    rows = _load_trade_journal_rows()
    event_id = str(entry.get("event_id", "")).strip()
    if not event_id:
        return False, rows

    seen = {str(r.get("event_id", "")) for r in rows}
    if event_id in seen:
        return False, rows

    rows.append(entry)
    _write_trade_journal(rows)
    return True, rows


def _summarize_trade_journal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "total_logged": 0,
            "buy_count": 0,
            "sell_count": 0,
            "avg_risk_usd": 0.0,
            "first_logged_at_et": None,
            "last_logged_at_et": None,
        }

    buy_count = sum(1 for row in rows if str(row.get("action")) == "BUY")
    sell_count = sum(1 for row in rows if str(row.get("action")) == "SELL")
    risks = [float(row.get("risk_per_trade_usd", 0.0) or 0.0) for row in rows]
    avg_risk = (sum(risks) / len(risks)) if risks else 0.0
    return {
        "total_logged": len(rows),
        "buy_count": int(buy_count),
        "sell_count": int(sell_count),
        "avg_risk_usd": float(avg_risk),
        "first_logged_at_et": rows[0].get("logged_at_et"),
        "last_logged_at_et": rows[-1].get("logged_at_et"),
    }


def _load_trade_history_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not TRADE_HISTORY_JSONL.exists():
        return rows
    try:
        for raw in TRADE_HISTORY_JSONL.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    except Exception:
        return []
    return rows


def _write_trade_history(rows: list[dict[str, Any]]) -> None:
    TRADE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    trimmed = rows[-TRADE_HISTORY_MAX_ROWS:] if TRADE_HISTORY_MAX_ROWS > 0 else rows
    with TRADE_HISTORY_JSONL.open("w", encoding="utf-8") as fh:
        for row in trimmed:
            fh.write(json.dumps(row) + "\n")

    fieldnames = [
        "trade_id",
        "entry_unix",
        "entry_et",
        "entry_time_et",
        "entry_price",
        "entry_session_id",
        "exit_unix",
        "exit_et",
        "exit_time_et",
        "exit_price",
        "exit_session_id",
        "direction",
        "bars_held",
        "minutes_held",
        "stop_price_at_entry",
        "max_adverse_excursion_usd",
        "point_value_usd",
        "loss_cap_usd",
        "exit_reason",
        "pnl_pct",
        "trade_profit_pct",
        "pnl_usd",
        "trade_profit_usd",
        "cumulative_pnl_pct",
        "cumulative_pnl_usd",
        "result",
        "updated_at_utc",
    ]
    with TRADE_HISTORY_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in trimmed:
            writer.writerow(row)


def _upsert_trade_history(new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _load_trade_history_rows()
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("trade_id", "")).strip()
        if key:
            by_id[key] = row
    for row in new_rows:
        key = str(row.get("trade_id", "")).strip()
        if not key:
            continue
        by_id[key] = row

    merged = sorted(by_id.values(), key=lambda r: int(r.get("entry_unix", 0)))
    running_pct = 0.0
    running_usd = 0.0
    enriched: list[dict[str, Any]] = []
    for row in merged:
        item = dict(row)
        pnl_pct = float(item.get("pnl_pct", item.get("trade_profit_pct", 0.0)) or 0.0)
        pnl_usd = float(item.get("pnl_usd", item.get("trade_profit_usd", (ACCOUNT_SIZE * pnl_pct / 100.0))) or 0.0)
        item["pnl_pct"] = float(pnl_pct)
        item["trade_profit_pct"] = float(pnl_pct)
        item["pnl_usd"] = float(pnl_usd)
        item["trade_profit_usd"] = float(pnl_usd)

        entry_unix = 0
        try:
            entry_unix = int(float(item.get("entry_unix", item.get("entry_time_unix", 0)) or 0))
        except (TypeError, ValueError):
            entry_unix = 0
        if entry_unix > 0:
            entry_label = _format_et_trade(entry_unix)
            item["entry_unix"] = entry_unix
            item["entry_et"] = entry_label
            item["entry_time_et"] = entry_label
        else:
            item["entry_time_et"] = item.get("entry_time_et", item.get("entry_et"))

        exit_unix = 0
        try:
            exit_unix = int(float(item.get("exit_unix", item.get("exit_time_unix", 0)) or 0))
        except (TypeError, ValueError):
            exit_unix = 0
        if exit_unix > 0:
            exit_label = _format_et_trade(exit_unix)
            item["exit_unix"] = exit_unix
            item["exit_et"] = exit_label
            item["exit_time_et"] = exit_label
        else:
            item["exit_time_et"] = item.get("exit_time_et", item.get("exit_et"))

        if entry_unix > 0 and exit_unix > 0:
            entry_dt_et = datetime.fromtimestamp(entry_unix, tz=timezone.utc).astimezone(ET_TZ)
            exit_dt_et = datetime.fromtimestamp(exit_unix, tz=timezone.utc).astimezone(ET_TZ)
            if _trading_session_id(entry_dt_et) != _trading_session_id(exit_dt_et):
                # Enforce same trading-session day trades only.
                continue

        running_pct += pnl_pct
        running_usd += pnl_usd
        item["cumulative_pnl_pct"] = float(running_pct)
        item["cumulative_pnl_usd"] = float(running_usd)
        enriched.append(item)

    merged = enriched
    _write_trade_history(merged)
    return merged


def _summarize_trade_history(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "avg_pnl_pct": 0.0,
            "total_pnl_pct": 0.0,
            "avg_hold_minutes": 0.0,
            "max_hold_minutes": 0.0,
            "exit_reason_distribution": {},
            "first_entry_et": None,
            "last_exit_et": None,
        }
    pnl = [float(r.get("pnl_pct", 0.0) or 0.0) for r in rows]
    pnl_usd = [float(r.get("pnl_usd", (ACCOUNT_SIZE * float(r.get("pnl_pct", 0.0) or 0.0) / 100.0)) or 0.0) for r in rows]
    hold_minutes = [float(r.get("minutes_held", 0.0) or 0.0) for r in rows]
    wins = sum(1 for x in pnl if x > 0)
    losses = sum(1 for x in pnl if x <= 0)
    exit_reason_distribution: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("exit_reason", "unknown"))
        exit_reason_distribution[reason] = exit_reason_distribution.get(reason, 0) + 1
    return {
        "total_trades": int(len(rows)),
        "winning_trades": int(wins),
        "losing_trades": int(losses),
        "win_rate_pct": float((wins / len(rows)) * 100.0),
        "avg_pnl_pct": float(sum(pnl) / len(pnl)),
        "total_pnl_pct": float(sum(pnl)),
        "avg_pnl_usd": float(sum(pnl_usd) / len(pnl_usd)),
        "total_pnl_usd": float(sum(pnl_usd)),
        "avg_hold_minutes": float(sum(hold_minutes) / len(hold_minutes)) if hold_minutes else 0.0,
        "max_hold_minutes": float(max(hold_minutes)) if hold_minutes else 0.0,
        "exit_reason_distribution": exit_reason_distribution,
        "first_entry_et": rows[0].get("entry_et"),
        "last_exit_et": rows[-1].get("exit_et"),
    }


def _fetch_json(url: str, timeout: float = 12.0) -> dict[str, Any]:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (dashboard-local)"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_scalar(value: str) -> Any:
    raw = value.strip().strip('"').strip("'")
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+\.\d+", raw):
        return float(raw)
    return raw


def _parse_kx_backtest_config() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "cost_bps": 5.0,
        "min_signals": 10,
        "hold_period": "1D",
        "entry_slippage_bps": 0.0,
        "exit_slippage_bps": 0.0,
        "commission_bps": 0.0,
        "cvar_alpha": 0.95,
        "cvar_risk_aversion": 3.0,
        "cvar_max_sleeve_weight": 0.8,
        "cvar_include_cash_sleeve": True,
        "cvar_grid_step": 0.05,
    }
    if not KX_CONFIG_FILE.exists():
        return defaults

    in_backtest = False
    for line in KX_CONFIG_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" ") and stripped.endswith(":"):
            in_backtest = stripped[:-1] == "backtest_config"
            continue
        if not in_backtest:
            continue
        if line.startswith("  ") and ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            if key in defaults:
                defaults[key] = _parse_scalar(value.split("#", 1)[0])
        elif not line.startswith("  "):
            break
    return defaults


_BUY_KEYWORDS = [
    "BUY",
    "BULLISH",
    "LONG",
    "UPGRADE",
    "BEAT",
    "OUTPERFORM",
    "OVERWEIGHT",
    "POSITIVE",
    "ACCUMULATE",
]
_SELL_KEYWORDS = [
    "SELL",
    "BEARISH",
    "SHORT",
    "DOWNGRADE",
    "MISS",
    "UNDERPERFORM",
    "UNDERWEIGHT",
    "NEGATIVE",
    "REDUCE",
    "CRASH",
    "DECLINE",
]


def _parse_direction_from_text(text: str) -> str:
    upper = text.upper()
    for kw in _BUY_KEYWORDS:
        if kw in upper:
            return "BUY"
    for kw in _SELL_KEYWORDS:
        if kw in upper:
            return "SELL"
    return "HOLD"


def _extract_sym_from_text(text: str) -> str:
    match = re.search(r"\$([A-Z]{1,5})\b", text)
    if match:
        return match.group(1)
    return "NQ"


def _load_kx_records_from_fallback_snapshot(limit: int) -> dict[str, Any] | None:
    fallback_path = BASE_DIR / "dashboard-fallback.json"
    if not fallback_path.exists():
        return None
    try:
        payload = json.loads(fallback_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    stats = payload.get("distillation_stats", {})
    if not isinstance(stats, dict):
        return None

    direction_mix_raw = stats.get("direction_mix", {})
    if not isinstance(direction_mix_raw, dict):
        return None

    buy_n = max(0, int(direction_mix_raw.get("BUY", 0) or 0))
    sell_n = max(0, int(direction_mix_raw.get("SELL", 0) or 0))
    hold_n = max(0, int(direction_mix_raw.get("HOLD", 0) or 0))
    total = buy_n + sell_n + hold_n
    if total <= 0:
        return None

    counts = {"BUY": buy_n, "SELL": sell_n, "HOLD": hold_n}
    directions: list[str] = []
    target = min(limit, total)
    # Interleave classes so simulated returns are not clustered by class.
    while len(directions) < target:
        progressed = False
        for key in ("BUY", "HOLD", "SELL"):
            if counts[key] > 0 and len(directions) < target:
                directions.append(key)
                counts[key] -= 1
                progressed = True
        if not progressed:
            break

    label_mix_raw = stats.get("label_mix", {})
    if not isinstance(label_mix_raw, dict):
        label_mix_raw = {}
    label_mix = {
        "positive": int(label_mix_raw.get("positive", buy_n) or 0),
        "negative": int(label_mix_raw.get("negative", sell_n) or 0),
        "neutral": int(label_mix_raw.get("neutral", hold_n) or 0),
    }

    top_symbols = stats.get("top_symbols", [])
    if not isinstance(top_symbols, list):
        top_symbols = []
    samples = stats.get("samples", [])
    if not isinstance(samples, list):
        samples = []

    return {
        "n_records": len(directions),
        "n_seed_records": int(stats.get("seed_records", 0) or 0),
        "label_mix": label_mix,
        "direction_mix": {
            "BUY": sum(1 for d in directions if d == "BUY"),
            "SELL": sum(1 for d in directions if d == "SELL"),
            "HOLD": sum(1 for d in directions if d == "HOLD"),
        },
        "top_symbols": top_symbols[:6],
        "directions": directions,
        "samples": samples[:6],
    }


def _effective_cost_bps(
    cost_bps: float,
    entry_slippage_bps: float,
    exit_slippage_bps: float,
    commission_bps: float,
) -> float:
    return max(
        0.0,
        float(cost_bps)
        + float(entry_slippage_bps)
        + float(exit_slippage_bps)
        + float(commission_bps),
    )


def _portfolio_metrics_array(returns: np.ndarray) -> dict[str, float]:
    if returns.size == 0:
        return {"sharpe": 0.0, "max_drawdown": 0.0, "total_return": 0.0, "win_rate": 0.0}
    mean_ret = float(np.mean(returns))
    stdev = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    sharpe = mean_ret / stdev if stdev > 0 else 0.0
    equity = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(equity)
    drawdown = equity / running_peak - 1.0
    return {
        "sharpe": sharpe,
        "max_drawdown": float(np.min(drawdown)) if drawdown.size else 0.0,
        "total_return": float(np.prod(1.0 + returns) - 1.0),
        "win_rate": float(np.mean(returns > 0.0)),
    }


def _integer_compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    out: list[tuple[int, ...]] = []
    for i in range(total + 1):
        for tail in _integer_compositions(total - i, parts - 1):
            out.append((i, *tail))
    return out


def _generate_weight_candidates(
    n_assets: int,
    max_weight: float,
    grid_step: float,
    random_samples: int = 2500,
) -> np.ndarray:
    max_weight = float(max_weight)
    if n_assets <= 0:
        return np.empty((0, 0))
    if n_assets == 1:
        return np.array([[1.0]])

    candidates: list[np.ndarray] = []
    step = min(max(float(grid_step), 0.01), 0.5)
    units = max(int(round(1.0 / step)), 2)

    if n_assets <= 4:
        for comp in _integer_compositions(units, n_assets):
            w = np.array(comp, dtype=float) / units
            if np.max(w) <= max_weight + 1e-12:
                candidates.append(w)
    else:
        rng = np.random.default_rng(42)
        for w in rng.dirichlet(np.ones(n_assets), size=random_samples):
            if np.max(w) <= max_weight + 1e-12:
                candidates.append(w)

    equal = np.full(n_assets, 1.0 / n_assets)
    if np.max(equal) <= max_weight + 1e-12:
        candidates.append(equal)

    for i in range(n_assets):
        w = np.zeros(n_assets, dtype=float)
        w[i] = 1.0
        if np.max(w) <= max_weight + 1e-12:
            candidates.append(w)

    if not candidates:
        return np.array([equal], dtype=float)
    return np.unique(np.round(np.vstack(candidates), 8), axis=0)


def _project_scenarios(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Numerically stable portfolio projection that avoids BLAS matmul warnings."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    if weights.ndim != 1 or matrix.shape[1] != weights.shape[0]:
        raise ValueError("shape mismatch in scenario projection")
    projected = np.sum(matrix * weights, axis=1, dtype=np.float64)
    return np.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)


def _load_kx_direction_records(limit: int = 500) -> dict[str, Any]:
    label_mix = {"positive": 0, "negative": 0, "neutral": 0}
    direction_mix = {"BUY": 0, "SELL": 0, "HOLD": 0}
    symbol_mix: dict[str, int] = {}
    directions: list[str] = []
    samples: list[dict[str, Any]] = []

    if KX_FINGPT_FILE.exists():
        with KX_FINGPT_FILE.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= limit:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                messages = row.get("messages", [])
                if len(messages) < 2:
                    continue
                user_text = str(messages[0].get("content", ""))
                assistant_text = str(messages[1].get("content", "")).strip().lower()

                if assistant_text.startswith("positive"):
                    direction = "BUY"
                    label_mix["positive"] += 1
                elif assistant_text.startswith("negative"):
                    direction = "SELL"
                    label_mix["negative"] += 1
                else:
                    direction = "HOLD"
                    label_mix["neutral"] += 1

                sym = _extract_sym_from_text(user_text)
                symbol_mix[sym] = symbol_mix.get(sym, 0) + 1
                direction_mix[direction] += 1
                directions.append(direction)

                if len(samples) < 6:
                    samples.append(
                        {
                            "sym": sym,
                            "direction": direction,
                            "text": user_text.replace("\n", " ")[:96],
                        }
                    )

    synthetic_records = 0
    if KX_TEST_FILE.exists():
        with KX_TEST_FILE.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= 200:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                messages = row.get("messages", [])
                if len(messages) < 2:
                    continue
                synthetic_records += 1
                assistant_text = str(messages[1].get("content", ""))
                _ = _parse_direction_from_text(assistant_text)

    if not directions:
        fallback = _load_kx_records_from_fallback_snapshot(limit=limit)
        if fallback:
            return fallback

    top_symbols = sorted(symbol_mix.items(), key=lambda x: x[1], reverse=True)[:6]
    return {
        "n_records": len(directions),
        "n_seed_records": synthetic_records,
        "label_mix": label_mix,
        "direction_mix": direction_mix,
        "top_symbols": [{"symbol": sym, "count": count} for sym, count in top_symbols],
        "directions": directions,
        "samples": samples,
    }


def _parse_hold_period_to_bars(hold_period: str) -> int:
    if not hold_period:
        return BARS_PER_SESSION
    text = str(hold_period).strip().lower()
    if text.endswith("d"):
        return max(1, int(float(text[:-1])) * BARS_PER_SESSION)
    if text.endswith("h"):
        return max(1, int(float(text[:-1]) * 60 // BAR_INTERVAL_MIN))
    if text.endswith("min"):
        return max(1, int(float(text[:-3]) // BAR_INTERVAL_MIN))
    return BARS_PER_SESSION


def _fetch_symbol_bars_ohlc(symbol: str, interval: str = "5m", period: str = "5d") -> dict[str, list[float] | list[int]]:
    sym = quote(symbol)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval={interval}&range={period}"
    payload = _fetch_json(url)
    result = payload.get("chart", {}).get("result", [None])[0]
    if not result:
        raise RuntimeError(f"No result from Yahoo chart API for {symbol}")

    timestamps = result.get("timestamp", [])
    indicators = result.get("indicators", {}).get("quote", [{}])[0]
    opens = indicators.get("open", [])
    highs = indicators.get("high", [])
    lows = indicators.get("low", [])
    closes = indicators.get("close", [])
    volumes = indicators.get("volume", [])
    rows: list[tuple[int, float, float, float, float, float]] = []
    for ts, o, h, l, c, v in zip(timestamps, opens, highs, lows, closes, volumes):
        if o is None or h is None or l is None or c is None:
            continue
        rows.append((int(ts), float(o), float(h), float(l), float(c), float(v or 0.0)))
    if len(rows) < 120:
        raise RuntimeError(f"Insufficient bars for {symbol}")
    return {
        "timestamp": [r[0] for r in rows],
        "open": [r[1] for r in rows],
        "high": [r[2] for r in rows],
        "low": [r[3] for r in rows],
        "close": [r[4] for r in rows],
        "volume": [r[5] for r in rows],
    }


def _fetch_symbol_bars(symbol: str, interval: str = "5m", period: str = "5d") -> tuple[list[int], list[float]]:
    bars = _fetch_symbol_bars_ohlc(symbol=symbol, interval=interval, period=period)
    return list(bars["timestamp"]), list(bars["close"])


def _fetch_multi_asset_prices(symbols: list[str]) -> pd.DataFrame:
    series_list: list[pd.Series] = []
    for sym in symbols:
        ts, close = _fetch_symbol_bars(sym, interval="5m", period=QPO_RANGE_5M)
        idx = pd.to_datetime(ts, unit="s", utc=True)
        series_list.append(pd.Series(close, index=idx, name=sym))
    df = pd.concat(series_list, axis=1).dropna()
    if len(df) < 120:
        raise RuntimeError("Insufficient aligned multi-asset bars for QPO overlay")
    return df


def _run_qpo_overlay() -> dict[str, Any]:
    symbols = ["NQ=F", "ES=F", "RTY=F", "^VIX"]
    prices = _fetch_multi_asset_prices(symbols)
    safe_prices = prices.clip(lower=1e-6)
    log_returns_df = np.log(safe_prices).diff().dropna()
    matrix = log_returns_df.to_numpy(dtype=float)
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    if len(matrix) < 50:
        raise RuntimeError("QPO overlay has insufficient finite return rows")
    matrix = np.clip(matrix, -0.25, 0.25)
    mean_vec = np.nan_to_num(matrix.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    std_vec = np.nan_to_num(matrix.std(axis=0, ddof=1), nan=0.001, posinf=0.001, neginf=0.001)
    std_vec = np.maximum(std_vec, 1e-4)
    cov = np.diag(std_vec**2)

    scenario_count = 600
    rng = np.random.default_rng(123)
    scenarios = rng.normal(loc=mean_vec, scale=std_vec, size=(scenario_count, len(symbols)))
    scenarios = np.nan_to_num(scenarios, nan=0.0, posinf=0.0, neginf=0.0)

    alpha = 0.95
    max_weight = 0.8
    candidates = _generate_weight_candidates(len(symbols), max_weight=max_weight, grid_step=0.1)
    risk_levels = [0.5, 1.0, 2.0, 3.0, 4.0]

    frontier: list[dict[str, Any]] = []
    best_weights = candidates[0]
    best_obj = float("-inf")
    for risk_aversion in risk_levels:
        local_best_obj = float("-inf")
        local_best_w = candidates[0]
        local_best_mean = 0.0
        local_best_cvar = 0.0
        for w in candidates:
            rets = _project_scenarios(scenarios, w)
            mean_ret = float(np.mean(rets))
            cvar = _cvar_loss(rets.tolist(), alpha)
            obj = mean_ret - risk_aversion * cvar
            if obj > local_best_obj:
                local_best_obj = obj
                local_best_w = w
                local_best_mean = mean_ret
                local_best_cvar = cvar
        frontier.append(
            {
                "risk_aversion": risk_aversion,
                "expected_return_pct": local_best_mean * BARS_PER_SESSION * 100,
                "cvar_loss_pct": local_best_cvar * 100,
                "objective": local_best_obj,
            }
        )
        if local_best_obj > best_obj:
            best_obj = local_best_obj
            best_weights = local_best_w

    label_map = {"NQ=F": "NQ", "ES=F": "ES", "RTY=F": "RTY", "^VIX": "VX"}
    weights = [
        {"symbol": label_map.get(sym, sym), "raw_symbol": sym, "weight": float(w)}
        for sym, w in zip(symbols, best_weights)
    ]
    weights.sort(key=lambda x: x["weight"], reverse=True)

    return {
        "assets": symbols,
        "return_type": "LOG",
        "fit_type": "gaussian",
        "confidence": alpha,
        "scenario_count": scenario_count,
        "covariance_trace": float(np.trace(cov)),
        "optimal_weights": weights,
        "frontier": frontier,
        "window_start": prices.index.min().isoformat(),
        "window_end": prices.index.max().isoformat(),
    }


def _run_kx_overlay(close: list[float], returns: list[float]) -> dict[str, Any]:
    cfg = _parse_kx_backtest_config()
    ds = _load_kx_direction_records(limit=500)
    directions = ds["directions"] or ["HOLD"] * 60

    effective_cost = _effective_cost_bps(
        cost_bps=float(cfg["cost_bps"]),
        entry_slippage_bps=float(cfg["entry_slippage_bps"]),
        exit_slippage_bps=float(cfg["exit_slippage_bps"]),
        commission_bps=float(cfg["commission_bps"]),
    )
    hold_bars = _parse_hold_period_to_bars(str(cfg["hold_period"]))

    sim_returns: list[float] = []
    long_sleeve: list[float] = []
    short_sleeve: list[float] = []
    for i, d in enumerate(directions):
        if i + hold_bars < len(close) and abs(close[i]) > 1e-9:
            base_ret = (close[i + hold_bars] - close[i]) / close[i]
        else:
            base_ret = returns[i % len(returns)] if returns else 0.0
        base_ret = float(np.clip(np.nan_to_num(base_ret, nan=0.0, posinf=0.0, neginf=0.0), -0.25, 0.25))

        if d == "BUY":
            mult = 1.0
        elif d == "SELL":
            mult = -1.0
        else:
            mult = 0.0

        net = mult * base_ret - (effective_cost / 10000.0 if mult != 0 else 0.0)
        sim_returns.append(net)
        long_sleeve.append(base_ret if d == "BUY" else 0.0)
        short_sleeve.append(-base_ret if d == "SELL" else 0.0)

    ret_array = np.nan_to_num(np.array(sim_returns, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ret_array = np.clip(ret_array, -0.25, 0.25)
    core_metrics = _portfolio_metrics_array(ret_array)
    core_metrics["n_trades"] = int(np.sum(np.array(directions) != "HOLD"))

    alpha = float(cfg["cvar_alpha"])
    risk_aversion = float(cfg["cvar_risk_aversion"])
    max_weight = float(cfg["cvar_max_sleeve_weight"])
    grid_step = float(cfg["cvar_grid_step"])
    include_cash = bool(cfg["cvar_include_cash_sleeve"])

    sleeves = [np.array(long_sleeve, dtype=float), np.array(short_sleeve, dtype=float)]
    sleeve_names = ["long_signal", "short_signal"]
    if include_cash:
        sleeves.append(np.zeros_like(ret_array))
        sleeve_names.append("cash")
    sleeve_matrix = np.vstack(sleeves).T
    sleeve_matrix = np.nan_to_num(sleeve_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    sleeve_matrix = np.clip(sleeve_matrix, -0.25, 0.25)

    candidates = _generate_weight_candidates(
        n_assets=sleeve_matrix.shape[1],
        max_weight=max_weight,
        grid_step=grid_step,
        random_samples=2000,
    )

    best_w = candidates[0]
    best_obj = float("-inf")
    best_mean = 0.0
    best_cvar = 0.0
    for w in candidates:
        p = _project_scenarios(sleeve_matrix, w)
        mean_ret = float(np.mean(p))
        cvar_loss = _cvar_loss(p.tolist(), alpha)
        obj = mean_ret - risk_aversion * cvar_loss
        if obj > best_obj:
            best_obj = obj
            best_w = w
            best_mean = mean_ret
            best_cvar = cvar_loss

    sized_returns = _project_scenarios(sleeve_matrix, best_w)
    sized_metrics = _portfolio_metrics_array(np.array(sized_returns, dtype=float))
    sized_metrics["n_trades"] = int(np.sum(np.array(directions) != "HOLD"))

    return {
        "dataset_records": ds["n_records"],
        "seed_records": ds["n_seed_records"],
        "label_mix": ds["label_mix"],
        "direction_mix": ds["direction_mix"],
        "top_symbols": ds["top_symbols"],
        "samples": ds["samples"],
        "config": {
            "cost_bps": float(cfg["cost_bps"]),
            "effective_cost_bps": float(effective_cost),
            "min_signals": int(cfg["min_signals"]),
            "hold_period": str(cfg["hold_period"]),
            "hold_bars_5m": hold_bars,
            "alpha": alpha,
            "risk_aversion": risk_aversion,
            "max_sleeve_weight": max_weight,
        },
        "backtest": {
            "sharpe": float(core_metrics["sharpe"]),
            "max_drawdown": float(core_metrics["max_drawdown"]),
            "total_return": float(core_metrics["total_return"]),
            "win_rate": float(core_metrics["win_rate"]),
            "n_trades": int(core_metrics["n_trades"]),
        },
        "cvar_sized": {
            "weights": {name: float(best_w[idx]) for idx, name in enumerate(sleeve_names)},
            "objective": float(best_obj),
            "expected_return": float(best_mean),
            "cvar_loss": float(best_cvar),
            "sharpe": float(sized_metrics["sharpe"]),
            "max_drawdown": float(sized_metrics["max_drawdown"]),
            "total_return": float(sized_metrics["total_return"]),
            "win_rate": float(sized_metrics["win_rate"]),
            "n_trades": int(sized_metrics["n_trades"]),
        },
    }


def _empty_kx_overlay(error: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset_records": 0,
        "seed_records": 0,
        "label_mix": {"positive": 0, "negative": 0, "neutral": 0},
        "direction_mix": {"BUY": 0, "SELL": 0, "HOLD": 0},
        "top_symbols": [],
        "samples": [],
        "config": {
            "cost_bps": 5.0,
            "effective_cost_bps": 5.0,
            "min_signals": 10,
            "hold_period": "1D",
            "hold_bars_5m": BARS_PER_SESSION,
            "alpha": 0.95,
            "risk_aversion": 3.0,
            "max_sleeve_weight": 0.8,
        },
        "backtest": {
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "win_rate": 0.0,
            "n_trades": 0,
        },
        "cvar_sized": {
            "weights": {"long_signal": 0.0, "short_signal": 0.0, "cash": 1.0},
            "objective": 0.0,
            "expected_return": 0.0,
            "cvar_loss": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "win_rate": 0.0,
            "n_trades": 0,
        },
    }
    if error:
        payload["error"] = error
    return payload


def _derive_kx_confluence(kx_overlay: dict[str, Any]) -> dict[str, Any]:
    direction_mix = kx_overlay.get("direction_mix", {})
    buy_n = float(direction_mix.get("BUY", 0))
    sell_n = float(direction_mix.get("SELL", 0))
    directional = buy_n + sell_n
    sentiment_bias = ((buy_n - sell_n) / directional) if directional > 1e-9 else 0.0

    backtest = kx_overlay.get("backtest", {})
    cvar_sized = kx_overlay.get("cvar_sized", {})
    cvar_sharpe = float(cvar_sized.get("sharpe", 0.0))
    core_sharpe = float(backtest.get("sharpe", 0.0))
    cvar_total = float(cvar_sized.get("total_return", 0.0))

    quality = 0.70 * max(0.0, cvar_sharpe / 0.15) + 0.30 * max(0.0, core_sharpe / 0.25)
    if cvar_total <= 0 and cvar_sharpe <= 0:
        quality = 0.0
    quality = max(0.0, min(1.4, quality))

    long_relax = max(0.68, min(1.0, 1.0 - max(0.0, sentiment_bias) * 0.85 * quality))
    short_boost = max(1.0, min(1.35, 1.0 + max(0.0, -sentiment_bias) * 0.55 * quality))

    return {
        "enabled": bool(quality > 0.01),
        "quality": float(quality),
        "sentiment_bias": float(sentiment_bias),
        "long_relax": float(long_relax),
        "short_boost": float(short_boost),
        "cvar_sharpe": float(cvar_sharpe),
        "backtest_sharpe": float(core_sharpe),
        "cvar_total_return": float(cvar_total),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)

    data = sorted(values)
    idx = (len(data) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return data[lo]
    frac = idx - lo
    return data[lo] * (1 - frac) + data[hi] * frac


def _cvar_loss(returns: list[float], alpha: float = 0.95) -> float:
    if not returns:
        return 0.0
    losses = [-r for r in returns]
    var_cutoff = _quantile(losses, alpha)
    tail = [x for x in losses if x >= var_cutoff - 1e-12]
    return _mean(tail) if tail else var_cutoff


def _rolling_mean(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(_mean(values[start : i + 1]))
    return out


def _rolling_stdev(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(_stdev(values[start : i + 1]))
    return out


def _cum_equity(returns: list[float]) -> list[float]:
    eq = [1.0]
    for r in returns:
        eq.append(eq[-1] * (1.0 + r))
    return eq


def _max_drawdown(returns: list[float]) -> float:
    eq = _cum_equity(returns)
    peak = eq[0]
    worst = 0.0
    for value in eq:
        if value > peak:
            peak = value
        dd = value / peak - 1.0
        if dd < worst:
            worst = dd
    return worst


def _fetch_nq_bars() -> tuple[list[int], list[float]]:
    return _fetch_symbol_bars("NQ=F", interval="5m", period=BACKTEST_RANGE_5M)


def _fetch_nq_ohlc_bars() -> dict[str, list[float] | list[int]]:
    return _fetch_symbol_bars_ohlc("NQ=F", interval="5m", period=BACKTEST_RANGE_5M)


def _fetch_watchlist() -> list[dict[str, Any]]:
    symbols = ["NQ=F", "ES=F", "RTY=F", "^VIX"]
    params = urlencode({"symbols": ",".join(symbols), "range": "1d", "interval": "5m"})
    url = f"https://query1.finance.yahoo.com/v7/finance/spark?{params}"
    label_map = {"NQ=F": "NQ", "ES=F": "ES", "RTY=F": "RTY", "^VIX": "VX"}
    out_map: dict[str, float] = {}

    try:
        payload = _fetch_json(url)
        result = payload.get("spark", {}).get("result", [])
        for row in result:
            symbol = row.get("symbol")
            response = row.get("response", [{}])[0]
            closes = [c for c in response.get("close", []) if c is not None]
            if not closes or symbol not in label_map:
                continue
            start = closes[0]
            end = closes[-1]
            change_pct = ((end - start) / start) * 100 if start else 0.0
            out_map[label_map[symbol]] = round(change_pct, 2)
    except Exception:
        out_map = {}

    # Weekend/holiday fallback: derive change from last 5-day history.
    for sym in symbols:
        label = label_map[sym]
        if label in out_map and abs(out_map[label]) >= 0.01:
            continue
        try:
            _, hist_close = _fetch_symbol_bars(sym, interval="5m", period="5d")
            if len(hist_close) >= 2 and abs(hist_close[0]) > 1e-9:
                hist_change = ((hist_close[-1] - hist_close[0]) / hist_close[0]) * 100.0
                out_map[label] = round(hist_change, 2)
        except Exception:
            if label not in out_map:
                out_map[label] = 0.0

    out = [{"symbol": k, "change_pct": out_map.get(k, 0.0)} for k in ["NQ", "ES", "RTY", "VX"]]

    if out:
        order = {"NQ": 0, "ES": 1, "RTY": 2, "VX": 3}
        out.sort(key=lambda x: order.get(x["symbol"], 99))
        return out

    return [
        {"symbol": "NQ", "change_pct": 0.0},
        {"symbol": "ES", "change_pct": 0.0},
        {"symbol": "RTY", "change_pct": 0.0},
        {"symbol": "VX", "change_pct": 0.0},
    ]


def _generate_live_signals_with_config(
    returns: list[float],
    cfg: dict[str, Any],
    kx_confluence: dict[str, Any] | None = None,
    macro_bias_override: int | None = None,
) -> list[int]:
    if not returns:
        return []

    momentum = _rolling_mean(returns, int(cfg["momentum_window"]))
    volatility = _rolling_stdev(returns, int(cfg["volatility_window"]))
    trend = _rolling_mean(returns, int(cfg["trend_window"]))

    mthr = float(cfg["momentum_threshold"])
    vol_cap = _quantile(volatility, float(cfg["volatility_quantile_cap"]))
    counter = float(cfg["countertrend_multiplier"])
    min_hold = int(cfg["min_hold_bars"])
    macro_allow = float(cfg["macro_countertrend_allow_multiplier"])
    trend_k = float(cfg.get("trend_regime_multiplier", 0.6))
    neutral_k = float(cfg.get("neutral_regime_multiplier", 1.3))
    high_vol_k = float(cfg.get("high_volatility_multiplier", 1.0))
    short_entry_k = float(cfg.get("short_entry_multiplier", 1.0))
    disable_longs_macro_short = bool(cfg.get("disable_longs_when_macro_short", False))
    use_kx_confluence = bool(cfg.get("use_kx_confluence", True))
    kx_strength = float(cfg.get("kx_confluence_strength", 1.0))
    kx_quality_floor = float(cfg.get("kx_quality_floor", 0.05))

    long_relax = 1.0
    short_boost = 1.0
    if use_kx_confluence and kx_confluence:
        quality = float(kx_confluence.get("quality", 0.0))
        if bool(kx_confluence.get("enabled", True)) and quality >= kx_quality_floor:
            raw_long_relax = float(kx_confluence.get("long_relax", 1.0))
            raw_short_boost = float(kx_confluence.get("short_boost", 1.0))
            long_relax = 1.0 - (1.0 - raw_long_relax) * _clamp(kx_strength, 0.0, 1.5)
            short_boost = 1.0 + (raw_short_boost - 1.0) * _clamp(kx_strength, 0.0, 1.5)

    if macro_bias_override is not None:
        macro_bias = int(macro_bias_override)
    else:
        macro_bias = 0
        try:
            macro_bias = int(_pdf_daily_trend_context()["bias"])
        except Exception:
            macro_bias = 0

    signals: list[int] = []
    position = 0
    held_bars = 0

    for m, v, t in zip(momentum, volatility, trend):
        signal = 0
        if v <= vol_cap * high_vol_k:
            if t <= -mthr * trend_k:
                if m <= -mthr * short_entry_k * short_boost:
                    signal = -1
                elif m >= mthr * counter:
                    signal = 1
            elif t >= mthr * trend_k:
                if m >= mthr:
                    signal = 1
                elif m <= -mthr * counter:
                    signal = -1
            elif abs(m) >= mthr * neutral_k:
                signal = 1 if m > 0 else -1

        # PDF-style medium-term trend overlay:
        # block weak countertrend intraday signals against the 63d daily bias.
        if macro_bias < 0:
            long_gate = mthr * macro_allow * long_relax
            if disable_longs_macro_short and signal > 0:
                signal = 0
            elif signal > 0 and m < long_gate:
                signal = 0
        elif macro_bias > 0 and signal < 0 and m > -mthr * macro_allow:
            signal = 0

        if position != 0 and held_bars < min_hold and signal != -position:
            signal = position
            held_bars += 1
        elif signal != position:
            position = signal
            held_bars = 1 if signal != 0 else 0
        else:
            held_bars = held_bars + 1 if signal != 0 else 0

        signals.append(signal)

    return signals


def _generate_live_signals(returns: list[float], kx_confluence: dict[str, Any] | None = None) -> list[int]:
    return _generate_live_signals_with_config(
        returns=returns,
        cfg=LIVE_SIGNAL_CONFIG,
        kx_confluence=kx_confluence,
    )


def _apply_execution_controls_with_config(
    exec_signal: list[int],
    returns: list[float],
    timestamps: list[int],
    cfg: dict[str, Any],
) -> list[int]:
    if not exec_signal:
        return []

    use_daily_guard = bool(cfg.get("use_daily_loss_guard", True))
    daily_limit_pct = float(cfg.get("daily_loss_limit_pct", 0.75))
    max_trades_day = int(cfg.get("max_trades_per_day", PROP_EXECUTION_CONFIG["max_trades_per_day"]))
    cooldown_bars = int(cfg.get("trade_cooldown_bars", 0))
    enforce_same_day = bool(cfg.get("enforce_same_day_trades", True))
    switch_cost = 2.5 / 10000.0

    out: list[int] = []
    prev = 0
    timer = 0
    active_day = ""
    day_pnl = 0.0
    day_locked = False
    trades_today = 0

    for i, desired in enumerate(exec_signal):
        ts = timestamps[i + 1]
        bar_dt_et = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(ET_TZ)
        session_id = _trading_session_id(bar_dt_et)
        if session_id != active_day:
            active_day = session_id
            day_pnl = 0.0
            day_locked = False
            trades_today = 0
            timer = 0

        desired_sig = int(desired)
        if enforce_same_day and not _is_within_trading_session_window(bar_dt_et):
            desired_sig = 0
        if use_daily_guard and day_locked:
            desired_sig = 0

        if cooldown_bars > 0 and timer > 0 and desired_sig != prev:
            desired_sig = prev

        switched = desired_sig != prev
        if switched and trades_today >= max_trades_day:
            desired_sig = prev
            switched = False

        if switched:
            trades_today += 1
            prev = desired_sig
            timer = cooldown_bars
        elif timer > 0:
            timer -= 1

        out.append(desired_sig)
        bar_ret = desired_sig * float(returns[i]) - (switch_cost if switched else 0.0)
        day_pnl += bar_ret
        if use_daily_guard and day_pnl <= -abs(daily_limit_pct) / 100.0:
            day_locked = True

    return out


def _apply_execution_controls(exec_signal: list[int], returns: list[float], timestamps: list[int]) -> list[int]:
    return _apply_execution_controls_with_config(
        exec_signal=exec_signal,
        returns=returns,
        timestamps=timestamps,
        cfg=LIVE_SIGNAL_CONFIG,
    )


def _finalize_trade(
    trade: dict[str, Any],
    *,
    exit_unix: int,
    exit_price: float,
    exit_reason: str,
    loss_cap_usd: float,
) -> dict[str, Any]:
    direction = int(trade["direction_sign"])
    entry_price = float(trade["entry_price"])
    point_value_usd = float(trade["point_value_usd"])
    pnl_points = (float(exit_price) - entry_price) * direction
    pnl_usd = pnl_points * point_value_usd
    pnl_pct = (pnl_usd / ACCOUNT_SIZE) * 100.0 if ACCOUNT_SIZE > 0 else 0.0
    entry_unix = int(trade["entry_unix"])
    minutes_held = max(0.0, (float(exit_unix) - float(entry_unix)) / 60.0)
    entry_dt = datetime.fromtimestamp(entry_unix, tz=timezone.utc).astimezone(ET_TZ)
    exit_dt = datetime.fromtimestamp(exit_unix, tz=timezone.utc).astimezone(ET_TZ)
    max_adverse_usd = float(trade.get("max_adverse_excursion_usd", 0.0))

    return {
        "trade_id": f"{entry_unix}|{'LONG' if direction > 0 else 'SHORT'}",
        "entry_unix": entry_unix,
        "entry_et": _format_et_trade(entry_unix),
        "entry_time_et": _format_et_trade(entry_unix),
        "entry_price": round(entry_price, 2),
        "exit_unix": int(exit_unix),
        "exit_et": _format_et_trade(exit_unix),
        "exit_time_et": _format_et_trade(exit_unix),
        "exit_price": round(float(exit_price), 2),
        "direction": "LONG" if direction > 0 else "SHORT",
        "entry_session_id": _trading_session_id(entry_dt),
        "exit_session_id": _trading_session_id(exit_dt),
        "bars_held": int(max(1, int(trade.get("bars_held", 1)))),
        "minutes_held": float(minutes_held),
        "stop_price_at_entry": round(float(trade["stop_price_at_entry"]), 2),
        "max_adverse_excursion_usd": float(max_adverse_usd),
        "point_value_usd": float(point_value_usd),
        "loss_cap_usd": float(loss_cap_usd),
        "exit_reason": str(exit_reason),
        "pnl_pct": float(pnl_pct),
        "trade_profit_pct": float(pnl_pct),
        "pnl_usd": float(pnl_usd),
        "trade_profit_usd": float(pnl_usd),
        "result": "WIN" if pnl_pct > 0 else "LOSS",
        "updated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
    }


def _simulate_exec_with_constraints(
    desired_exec_signal: list[int],
    *,
    timestamps: list[int],
    open_px: list[float],
    high_px: list[float],
    low_px: list[float],
    close_px: list[float],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    n = min(len(desired_exec_signal), max(0, len(close_px) - 1))
    if n <= 0:
        return {
            "exec_signal": [],
            "strat_returns": [],
            "trades": [],
            "constraint_checks": {
                "max_hold_violation_count": 0,
                "loss_cap_violation_count": 0,
                "cross_session_violation_count": 0,
                "exit_reason_distribution": {},
            },
        }

    switch_cost = 2.5 / 10000.0
    max_hold_minutes = max(1, int(cfg.get("max_hold_minutes", 90)))
    max_hold_seconds = max_hold_minutes * 60
    loss_cap_usd = abs(float(cfg.get("max_active_trade_loss_usd", 500.0)))
    intrabar_stop_touch = bool(cfg.get("intrabar_stop_touch", True))
    enforce_same_day = bool(cfg.get("enforce_same_day_trades", True))
    point_value_usd = float(NQ_POINT_VALUE_USD)
    stop_distance_points = (loss_cap_usd / point_value_usd) if point_value_usd > 1e-9 else 0.0

    exec_signal: list[int] = []
    strat_returns: list[float] = []
    trades: list[dict[str, Any]] = []
    current_pos = 0
    open_trade: dict[str, Any] | None = None

    for i in range(n):
        bar_idx = i + 1
        bar_start_ts = int(timestamps[i])
        bar_end_ts = int(timestamps[bar_idx])
        bar_end_dt_et = datetime.fromtimestamp(bar_end_ts, tz=timezone.utc).astimezone(ET_TZ)
        desired = int(desired_exec_signal[i])

        if enforce_same_day and not _is_within_trading_session_window(bar_end_dt_et):
            desired = 0

        start_pos = current_pos
        bar_open = float(open_px[bar_idx])
        bar_high = float(high_px[bar_idx])
        bar_low = float(low_px[bar_idx])
        bar_close = float(close_px[bar_idx])

        if desired != current_pos:
            if current_pos != 0 and open_trade is not None:
                reason = "signal_flip" if desired != 0 else "signal_flat"
                trades.append(
                    _finalize_trade(
                        open_trade,
                        exit_unix=bar_start_ts,
                        exit_price=bar_open,
                        exit_reason=reason,
                        loss_cap_usd=loss_cap_usd,
                    )
                )
                open_trade = None
                current_pos = 0

            if desired != 0:
                entry_dt_et = datetime.fromtimestamp(bar_start_ts, tz=timezone.utc).astimezone(ET_TZ)
                can_open = True
                if enforce_same_day and not _is_within_trading_session_window(entry_dt_et):
                    can_open = False
                if can_open:
                    stop_price = bar_open - stop_distance_points if desired > 0 else bar_open + stop_distance_points
                    open_trade = {
                        "entry_unix": int(bar_start_ts),
                        "entry_price": float(bar_open),
                        "direction_sign": int(desired),
                        "point_value_usd": float(point_value_usd),
                        "stop_price_at_entry": float(stop_price),
                        "max_adverse_excursion_usd": 0.0,
                        "bars_held": 0,
                    }
                    current_pos = int(desired)

        bar_return = 0.0
        if current_pos != 0 and open_trade is not None:
            entry_price = float(open_trade["entry_price"])
            direction = int(open_trade["direction_sign"])
            stop_price = float(open_trade["stop_price_at_entry"])

            if direction > 0:
                adverse_points = bar_low - entry_price
            else:
                adverse_points = entry_price - bar_high
            adverse_usd = adverse_points * point_value_usd
            open_trade["max_adverse_excursion_usd"] = min(
                float(open_trade.get("max_adverse_excursion_usd", 0.0)),
                float(adverse_usd),
            )
            open_trade["bars_held"] = int(open_trade.get("bars_held", 0)) + 1

            stop_hit = False
            if intrabar_stop_touch:
                if direction > 0 and bar_low <= stop_price:
                    stop_hit = True
                if direction < 0 and bar_high >= stop_price:
                    stop_hit = True

            elapsed_seconds = int(bar_end_ts - int(open_trade["entry_unix"]))
            exit_reason: str | None = None
            exit_price = float(bar_close)

            if stop_hit:
                exit_reason = "loss_cap"
                exit_price = float(stop_price)
            elif elapsed_seconds >= max_hold_seconds:
                exit_reason = "time_cap"
            elif enforce_same_day:
                entry_dt_et = datetime.fromtimestamp(int(open_trade["entry_unix"]), tz=timezone.utc).astimezone(ET_TZ)
                if (
                    not _is_within_trading_session_window(bar_end_dt_et)
                    or _trading_session_id(entry_dt_et) != _trading_session_id(bar_end_dt_et)
                ):
                    exit_reason = "session_flatten"

            if direction > 0:
                bar_return = (exit_price - float(close_px[i])) / float(close_px[i])
            else:
                bar_return = (float(close_px[i]) - exit_price) / float(close_px[i])

            if exit_reason is not None:
                trades.append(
                    _finalize_trade(
                        open_trade,
                        exit_unix=bar_end_ts,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        loss_cap_usd=loss_cap_usd,
                    )
                )
                open_trade = None
                current_pos = 0

        switched = current_pos != start_pos
        if switched:
            bar_return -= switch_cost
        strat_returns.append(float(bar_return))
        exec_signal.append(int(current_pos))

    if current_pos != 0 and open_trade is not None:
        final_ts = int(timestamps[n])
        final_close = float(close_px[n])
        trades.append(
            _finalize_trade(
                open_trade,
                exit_unix=final_ts,
                exit_price=final_close,
                exit_reason="end_of_data",
                loss_cap_usd=loss_cap_usd,
            )
        )
        exec_signal[-1] = 0

    exit_reason_distribution: dict[str, int] = {}
    max_hold_violations = 0
    loss_cap_violations = 0
    cross_session_violations = 0
    for t in trades:
        reason = str(t.get("exit_reason", "unknown"))
        exit_reason_distribution[reason] = exit_reason_distribution.get(reason, 0) + 1
        if float(t.get("minutes_held", 0.0)) > float(max_hold_minutes) + 1e-9:
            max_hold_violations += 1
        if float(t.get("pnl_usd", 0.0)) < -loss_cap_usd - 1e-6:
            loss_cap_violations += 1
        if str(t.get("entry_session_id")) != str(t.get("exit_session_id")):
            cross_session_violations += 1

    return {
        "exec_signal": exec_signal,
        "strat_returns": strat_returns,
        "trades": trades,
        "constraint_checks": {
            "max_hold_violation_count": int(max_hold_violations),
            "loss_cap_violation_count": int(loss_cap_violations),
            "cross_session_violation_count": int(cross_session_violations),
            "exit_reason_distribution": exit_reason_distribution,
            "max_hold_minutes": int(max_hold_minutes),
            "max_active_trade_loss_usd": float(loss_cap_usd),
        },
    }


def _pdf_daily_trend_context() -> dict[str, Any]:
    cfg = PDF_TREND_CONFIG
    lookback = int(cfg["momentum_lookback_days"])
    vol_window = int(cfg["vol_lookback_days"])
    target_ann_vol = float(cfg["target_ann_vol"])
    max_lev = float(cfg["max_leverage"])

    _, close = _fetch_symbol_bars("NQ=F", interval="1d", period="3y")
    if len(close) < 5:
        return {
            "bias": 0,
            "momentum_lookback_days": lookback,
            "momentum_return_pct": 0.0,
            "realized_ann_vol_pct": 0.0,
            "target_ann_vol_pct": target_ann_vol * 100.0,
            "target_leverage": 0.0,
        }

    if len(close) > lookback:
        mom = close[-1] / close[-(lookback + 1)] - 1.0
    else:
        mom = close[-1] / close[0] - 1.0

    daily_returns = [(close[i] - close[i - 1]) / close[i - 1] for i in range(1, len(close))]
    vol_slice = daily_returns[-vol_window:] if len(daily_returns) >= vol_window else daily_returns
    sigma_daily = _stdev(vol_slice)
    realized_ann = sigma_daily * math.sqrt(252) if sigma_daily > 0 else 0.0
    target_daily = target_ann_vol / math.sqrt(252)
    target_leverage = target_daily / sigma_daily if sigma_daily > 1e-9 else 0.0
    target_leverage = _clamp(target_leverage, 0.0, max_lev)

    bias = 1 if mom > 0 else -1 if mom < 0 else 0
    return {
        "bias": bias,
        "momentum_lookback_days": lookback,
        "momentum_return_pct": mom * 100.0,
        "realized_ann_vol_pct": realized_ann * 100.0,
        "target_ann_vol_pct": target_ann_vol * 100.0,
        "target_leverage": target_leverage,
    }


def _build_payload() -> dict[str, Any]:
    bars_all = _fetch_nq_ohlc_bars()
    timestamps_all = list(bars_all["timestamp"])
    open_all = list(bars_all["open"])
    high_all = list(bars_all["high"])
    low_all = list(bars_all["low"])
    close_all = list(bars_all["close"])
    eval_end_idx = _resolve_evaluation_end_index(timestamps_all)
    if eval_end_idx < 2:
        eval_end_idx = len(timestamps_all) - 1

    timestamps = timestamps_all[: eval_end_idx + 1]
    open_px = open_all[: eval_end_idx + 1]
    high_px = high_all[: eval_end_idx + 1]
    low_px = low_all[: eval_end_idx + 1]
    close = close_all[: eval_end_idx + 1]
    live_latest_ts = timestamps_all[-1]

    returns = [(close[i] - close[i - 1]) / close[i - 1] for i in range(1, len(close))]
    try:
        kx_overlay = _run_kx_overlay(close=close, returns=returns)
        kx_status = "Integrated"
    except Exception as e:
        kx_overlay = _empty_kx_overlay(str(e))
        kx_status = "Degraded"
    kx_confluence = _derive_kx_confluence(kx_overlay)

    raw_signal = _generate_live_signals(returns, kx_confluence=kx_confluence)
    desired_exec_signal = _apply_execution_controls([0] + raw_signal[:-1], returns, timestamps)
    sim = _simulate_exec_with_constraints(
        desired_exec_signal,
        timestamps=timestamps,
        open_px=open_px,
        high_px=high_px,
        low_px=low_px,
        close_px=close,
        cfg=LIVE_SIGNAL_CONFIG,
    )
    exec_signal = list(sim["exec_signal"])
    strat_returns = list(sim["strat_returns"])
    sim_trade_rows = list(sim["trades"])
    constraint_checks = dict(sim["constraint_checks"])
    n_trades = int(len(sim_trade_rows))
    trade_pnls = [float(t.get("pnl_usd", 0.0) or 0.0) for t in sim_trade_rows]

    mean_ret = _mean(strat_returns)
    std_ret = _stdev(strat_returns)
    annual_factor = math.sqrt(252 * BARS_PER_SESSION)
    sharpe = (mean_ret / std_ret * annual_factor) if std_ret > 1e-9 else 0.0

    total_return_pct = (_cum_equity(strat_returns)[-1] - 1.0) * 100
    max_drawdown_pct = _max_drawdown(strat_returns) * 100
    win_rate_pct = ((sum(1 for x in trade_pnls if x > 0) / len(trade_pnls)) * 100.0) if trade_pnls else 0.0

    recent_trend = _mean(returns[-12:])
    recent_vol = _stdev(returns[-24:])
    expected_session_return_pct = recent_trend * BARS_PER_SESSION * 100

    confidence = 58 + recent_trend * 10000 + sharpe * 3 - recent_vol * 1200
    forecast_confidence_pct = _clamp(confidence, 32, 93)

    projected_max_drawdown_pct = -abs((recent_vol * math.sqrt(BARS_PER_SESSION) * 2.2) * 100)

    # Regime map from last 8 windows
    regimes: list[dict[str, str]] = []
    recent_returns = returns[-64:]
    labels_bull = ["Impulse", "Trend Hold", "Reclaim", "Drive"]
    labels_neutral = ["Absorb", "Balance", "Pause", "Range"]
    labels_bear = ["Sweep", "Fade", "Liquidation", "Stress"]

    for i in range(8):
        chunk = recent_returns[i * 8 : (i + 1) * 8]
        m = _mean(chunk)
        v = _stdev(chunk)
        if m > 0.00025 and v < 0.0048:
            kind = "bull"
            label = labels_bull[i % len(labels_bull)]
        elif m < -0.00025:
            kind = "bear"
            label = labels_bear[i % len(labels_bear)]
        else:
            kind = "neutral"
            label = labels_neutral[i % len(labels_neutral)]
        regimes.append({"kind": kind, "label": label})

    # Signal mix from configurable lookback so users can inspect broader directional context.
    recent_signals = raw_signal[-SIGNAL_MIX_LOOKBACK_BARS:]
    signal_mix = {
        "buy": sum(1 for s in recent_signals if s == 1),
        "sell": sum(1 for s in recent_signals if s == -1),
        "hold": sum(1 for s in recent_signals if s == 0),
    }

    # CVaR sleeve sizing
    long_sleeve = [r if s == 1 else 0.0 for s, r in zip(exec_signal, returns)]
    short_sleeve = [-r if s == -1 else 0.0 for s, r in zip(exec_signal, returns)]
    cash_sleeve = [0.0 for _ in returns]

    best_obj = -10**9
    best = (0.5, 0.3, 0.2)
    alpha = 0.95
    risk_aversion = 3.0

    for wl_i in range(0, 17):
        for ws_i in range(0, 17):
            wl = wl_i * 0.05
            ws = ws_i * 0.05
            if wl > 0.8 or ws > 0.8 or wl + ws > 1.0:
                continue
            wc = 1.0 - wl - ws
            portfolio = [wl * l + ws * s + wc * c for l, s, c in zip(long_sleeve, short_sleeve, cash_sleeve)]
            cvar = _cvar_loss(portfolio, alpha)
            obj = _mean(portfolio) - risk_aversion * cvar
            if obj > best_obj:
                best_obj = obj
                best = (wl, ws, wc)

    wl, ws, wc = best
    # If optimization collapses to full cash, enforce a small directional floor
    # derived from live signal mix so the dashboard reflects active bias.
    if wl + ws < 0.05:
        directional = signal_mix["buy"] + signal_mix["sell"]
        if directional > 0:
            active_ratio = directional / max(len(recent_signals), 1)
            target_alloc = _clamp(0.2 + 0.35 * active_ratio, 0.2, 0.55)
            buy_share = signal_mix["buy"] / directional
            sell_share = signal_mix["sell"] / directional
            wl = round(target_alloc * buy_share, 2)
            ws = round(target_alloc * sell_share, 2)
            if wl + ws <= 1e-9:
                ws = target_alloc
            if wl + ws > 0.8:
                scale = 0.8 / (wl + ws)
                wl = round(wl * scale, 2)
                ws = round(ws * scale, 2)
            wc = max(0.0, round(1.0 - wl - ws, 2))

    def sleeve_row(name: str, weight: float, sleeve: list[float], status: str) -> dict[str, Any]:
        return {
            "sleeve": name,
            "weight": weight,
            "expected_return_pct": _mean(sleeve) * BARS_PER_SESSION * 100,
            "cvar_loss_pct": _cvar_loss(sleeve, alpha) * 100,
            "daily_var_pct": -_cvar_loss([weight * x for x in sleeve], alpha) * 100,
            "status": status,
        }

    allocation_rows = [
        sleeve_row("Long Signal", wl, long_sleeve, "Active"),
        sleeve_row("Short Signal", ws, short_sleeve, "Hedge"),
        sleeve_row("Cash Buffer", wc, cash_sleeve, "Reserve"),
    ]

    # Forecast line: recent closes
    forecast_points = close[-56:]

    # Recent signal table
    signals = []
    for i in range(max(0, len(returns) - 30), len(returns)):
        label = "BUY" if exec_signal[i] == 1 else "SELL" if exec_signal[i] == -1 else "HOLD"
        signals.append(
            {
                "time": _format_et_short(timestamps[i + 1]),
                "close": round(close[i + 1], 2),
                "signal": label,
                "bar_return_pct": returns[i] * 100,
                "strategy_return_pct": strat_returns[i] * 100,
            }
        )

    latest_ts = timestamps[-1]
    next_bar_ts = latest_ts + BAR_INTERVAL_MIN * 60
    now_et = _now_et().strftime("%H:%M:%S ET")
    now_et_full = _now_et().strftime("%Y-%m-%d %H:%M:%S ET")

    def _sig_label(sig: int) -> str:
        if sig > 0:
            return "BUY"
        if sig < 0:
            return "SELL"
        return "HOLD"

    next_signal = raw_signal[-1] if raw_signal else 0
    current_exec_signal = exec_signal[-1] if exec_signal else 0
    prev_raw_signal = raw_signal[-2] if len(raw_signal) > 1 else 0
    signal_changed = next_signal != prev_raw_signal

    stop_points = _round_to_tick(_clamp(close[-1] * max(recent_vol, 0.00035) * 1.35, 8.0, 80.0))
    target_points = _round_to_tick(stop_points * float(PROP_EXECUTION_CONFIG["target_rr"]))
    risk_per_trade_usd = ACCOUNT_SIZE * float(PROP_EXECUTION_CONFIG["risk_per_trade_pct"])
    risk_per_nq_usd = stop_points * NQ_POINT_VALUE_USD
    risk_per_mnq_usd = stop_points * MNQ_POINT_VALUE_USD

    sleeve_weight = wl if next_signal > 0 else ws if next_signal < 0 else 0.0
    risk_sized_nq = int(risk_per_trade_usd // risk_per_nq_usd) if risk_per_nq_usd > 1e-9 else 0
    risk_sized_mnq = int(risk_per_trade_usd // risk_per_mnq_usd) if risk_per_mnq_usd > 1e-9 else 0
    nq_contracts = min(
        int(PROP_EXECUTION_CONFIG["max_nq_contracts"]),
        max(0, int(round(risk_sized_nq * sleeve_weight))),
    )
    mnq_contracts = min(
        int(PROP_EXECUTION_CONFIG["max_mnq_contracts"]),
        max(0, int(round(risk_sized_mnq * sleeve_weight))),
    )
    if next_signal != 0 and mnq_contracts == 0 and risk_sized_mnq > 0:
        mnq_contracts = 1

    next_bar_dt_et = datetime.fromtimestamp(next_bar_ts, tz=timezone.utc).astimezone(ET_TZ)
    within_same_day_window = _is_within_trading_session_window(next_bar_dt_et)
    if not within_same_day_window:
        nq_contracts = 0
        mnq_contracts = 0

    entry_ref = close[-1]
    stop_price = None
    target_price = None
    if next_signal > 0:
        stop_price = _round_to_tick(entry_ref - stop_points)
        target_price = _round_to_tick(entry_ref + target_points)
    elif next_signal < 0:
        stop_price = _round_to_tick(entry_ref + stop_points)
        target_price = _round_to_tick(entry_ref - target_points)

    latest_session_id = _trading_session_id(
        datetime.fromtimestamp(latest_ts, tz=timezone.utc).astimezone(ET_TZ)
    )
    trades_today = 0
    today_return = 0.0
    for i, sig in enumerate(exec_signal):
        ts = timestamps[i + 1]
        bar_session_id = _trading_session_id(datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(ET_TZ))
        if bar_session_id != latest_session_id:
            continue
        prev_sig = exec_signal[i - 1] if i > 0 else 0
        if sig != prev_sig:
            trades_today += 1
        today_return += strat_returns[i]

    daily_model_pnl_usd = today_return * ACCOUNT_SIZE
    daily_loss_limit_usd = ACCOUNT_SIZE * float(PROP_EXECUTION_CONFIG["daily_loss_limit_pct"])
    under_trade_cap = trades_today < int(PROP_EXECUTION_CONFIG["max_trades_per_day"])
    within_daily_limit = daily_model_pnl_usd > -daily_loss_limit_usd
    has_direction = next_signal != 0
    has_size = nq_contracts > 0 or mnq_contracts > 0
    eligible = bool(under_trade_cap and within_daily_limit and has_direction and has_size and within_same_day_window)

    checks = [
        {
            "name": "Signal Is Directional",
            "pass": has_direction,
            "detail": f"Next-bar signal is {_sig_label(next_signal)}.",
        },
        {
            "name": "Size Is Tradable",
            "pass": has_size,
            "detail": f"Suggested size NQ {nq_contracts} / MNQ {mnq_contracts}.",
        },
        {
            "name": "Same-Day Session Window",
            "pass": within_same_day_window,
            "detail": (
                f"Next bar {next_bar_dt_et.strftime('%Y-%m-%d %H:%M:%S ET')} must be within the NQ session "
                f"window ({TRADING_SESSION_OPEN_ET.strftime('%H:%M')} -> "
                f"{TRADING_SESSION_FLAT_TIME_ET.strftime('%H:%M')} ET pre-close flatten)."
            ),
        },
        {
            "name": "Daily Trade Cap",
            "pass": under_trade_cap,
            "detail": (
                f"{trades_today}/{int(PROP_EXECUTION_CONFIG['max_trades_per_day'])} "
                "signal changes today."
            ),
        },
        {
            "name": "Daily Loss Guard",
            "pass": within_daily_limit,
            "detail": (
                f"Model day PnL ${daily_model_pnl_usd:,.2f} vs "
                f"limit -${daily_loss_limit_usd:,.2f}."
            ),
        },
    ]

    backtest_window_start = (
        datetime.fromtimestamp(timestamps[0], tz=timezone.utc)
        .astimezone(ET_TZ)
        .strftime("%Y-%m-%d %H:%M:%S ET")
    )
    backtest_window_end = (
        datetime.fromtimestamp(latest_ts, tz=timezone.utc).astimezone(ET_TZ).strftime("%Y-%m-%d %H:%M:%S ET")
    )
    approx_sessions = len(returns) / BARS_PER_SESSION if BARS_PER_SESSION > 0 else 0.0
    trading_session_days = len(
        {
            _trading_session_id(datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(ET_TZ))
            for ts in timestamps
        }
    )
    trades_per_trading_session = (n_trades / trading_session_days) if trading_session_days > 0 else 0.0
    trades_per_equiv_session = (n_trades / approx_sessions) if approx_sessions > 1e-9 else 0.0

    execution_plan = {
        "eligible": eligible,
        "action_next_bar": _sig_label(next_signal),
        "current_position": _sig_label(current_exec_signal),
        "signal_changed": signal_changed,
        "as_of_et": _format_et(latest_ts),
        "next_bar_et": _format_et(next_bar_ts),
        "entry_reference": round(entry_ref, 2),
        "stop_price": round(stop_price, 2) if stop_price is not None else None,
        "target_price": round(target_price, 2) if target_price is not None else None,
        "stop_distance_points": float(stop_points),
        "target_distance_points": float(target_points),
        "risk_per_trade_usd": round(risk_per_trade_usd, 2),
        "contract_plan": {
            "nq": int(nq_contracts if next_signal != 0 else 0),
            "mnq": int(mnq_contracts if next_signal != 0 else 0),
            "risk_per_nq_usd": round(risk_per_nq_usd, 2),
            "risk_per_mnq_usd": round(risk_per_mnq_usd, 2),
            "sleeve_weight": round(float(sleeve_weight), 2),
        },
        "prop_rules": {
            "max_trades_per_day": int(PROP_EXECUTION_CONFIG["max_trades_per_day"]),
            "trades_today": int(trades_today),
            "daily_model_pnl_usd": round(daily_model_pnl_usd, 2),
            "daily_loss_limit_usd": round(daily_loss_limit_usd, 2),
            "checks": checks,
        },
        "backtest_window": {
            "start_et": backtest_window_start,
            "end_et": backtest_window_end,
            "range_5m": BACKTEST_RANGE_5M,
            "bars": len(returns),
            "trading_session_days": int(trading_session_days),
            "sessions_equiv_78bar": round(approx_sessions, 2),
            "sessions_equiv": round(approx_sessions, 2),
            "trades": int(n_trades),
            "trades_per_session": round(trades_per_trading_session, 2),
            "trades_per_session_equiv_78bar": round(trades_per_equiv_session, 2),
            "avg_return_per_trade_pct": round((total_return_pct / n_trades) if n_trades > 0 else 0.0, 4),
        },
        "constraint_checks": {
            "max_hold_minutes": int(constraint_checks.get("max_hold_minutes", LIVE_SIGNAL_CONFIG.get("max_hold_minutes", 90))),
            "max_active_trade_loss_usd": float(
                constraint_checks.get("max_active_trade_loss_usd", LIVE_SIGNAL_CONFIG.get("max_active_trade_loss_usd", 500.0))
            ),
            "max_hold_violation_count": int(constraint_checks.get("max_hold_violation_count", 0)),
            "loss_cap_violation_count": int(constraint_checks.get("loss_cap_violation_count", 0)),
            "cross_session_violation_count": int(constraint_checks.get("cross_session_violation_count", 0)),
            "exit_reason_distribution": dict(constraint_checks.get("exit_reason_distribution", {})),
        },
        "notes": (
            "Signal is generated on 5m bar close; execute on next 5m bar open to mirror model timing. "
            "Intrabar stop-touch exits enforce the $500 active-loss cap, max hold is 90 minutes, "
            "and overnight holds are blocked (same-session-day trades only)."
        ),
    }

    trade_entry_candidate = bool(eligible and signal_changed and has_direction and has_size)
    journal_entry: dict[str, Any] | None = None
    if trade_entry_candidate:
        journal_entry = {
            "event_id": (
                f"{int(next_bar_ts)}|{_sig_label(next_signal)}|"
                f"{round(entry_ref, 2)}|{int(nq_contracts)}|{int(mnq_contracts)}"
            ),
            "logged_at_et": now_et_full,
            "logged_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "signal_time_et": _format_et_trade(latest_ts),
            "execute_at_et": _format_et_trade(next_bar_ts),
            "execute_at_unix": int(next_bar_ts),
            "action": _sig_label(next_signal),
            "entry_reference": round(entry_ref, 2),
            "stop_price": round(stop_price, 2) if stop_price is not None else None,
            "target_price": round(target_price, 2) if target_price is not None else None,
            "nq_contracts": int(nq_contracts),
            "mnq_contracts": int(mnq_contracts),
            "risk_per_trade_usd": round(risk_per_trade_usd, 2),
            "sleeve_weight": round(float(sleeve_weight), 2),
            "eligible": bool(eligible),
            "signal_changed": bool(signal_changed),
            "notes": "Auto-logged on directional signal change with tradable size.",
        }
        journal_logged, trade_journal_rows = _append_trade_journal(journal_entry)
    else:
        journal_logged = False
        trade_journal_rows = _load_trade_journal_rows()

    trade_journal_summary = _summarize_trade_journal(trade_journal_rows)
    trade_history_rows = _upsert_trade_history(sim_trade_rows)
    trade_history_summary = _summarize_trade_history(trade_history_rows)

    try:
        qpo_overlay = _run_qpo_overlay()
        qpo_status = "Integrated"
    except Exception as e:
        qpo_overlay = {
            "assets": [],
            "return_type": "LOG",
            "fit_type": "gaussian",
            "confidence": 0.95,
            "scenario_count": 0,
            "covariance_trace": 0.0,
            "optimal_weights": [],
            "frontier": [],
            "window_start": "",
            "window_end": "",
            "error": str(e),
        }
        qpo_status = "Degraded"

    try:
        pdf_trend = _pdf_daily_trend_context()
    except Exception as e:
        pdf_trend = {
            "bias": 0,
            "momentum_lookback_days": int(PDF_TREND_CONFIG["momentum_lookback_days"]),
            "momentum_return_pct": 0.0,
            "realized_ann_vol_pct": 0.0,
            "target_ann_vol_pct": float(PDF_TREND_CONFIG["target_ann_vol"]) * 100.0,
            "target_leverage": 0.0,
            "error": str(e),
        }

    events = [
        {
            "time": _format_et_short(latest_ts),
            "message": f"NQ close synced at {close[-1]:,.2f} from Yahoo 5m feed.",
        },
        {
            "time": now_et.replace(" ET", ""),
            "message": f"Backtest updated: Sharpe {sharpe:.2f}, win rate {win_rate_pct:.1f}%.",
        },
        {
            "time": now_et.replace(" ET", ""),
            "message": f"CVaR sizing refreshed: long {wl:.2f}, short {ws:.2f}, cash {wc:.2f}.",
        },
        {
            "time": now_et.replace(" ET", ""),
            "message": (
                f"KX distillation overlay: {kx_overlay['dataset_records']} FinGPT records, "
                f"Sharpe {kx_overlay['backtest']['sharpe']:.2f}, "
                f"CVaR-sized Sharpe {kx_overlay['cvar_sized']['sharpe']:.2f}."
            ),
        },
        {
            "time": now_et.replace(" ET", ""),
            "message": (
                f"QPO overlay: {qpo_overlay['scenario_count']} Gaussian scenarios, "
                f"{qpo_overlay['fit_type']} fit, top sleeve "
                f"{(qpo_overlay['optimal_weights'][0]['symbol'] if qpo_overlay['optimal_weights'] else 'N/A')}."
            ),
        },
        {
            "time": now_et.replace(" ET", ""),
            "message": (
                f"PDF trend overlay: {pdf_trend['momentum_lookback_days']}d momentum "
                f"{pdf_trend['momentum_return_pct']:+.2f}%, bias "
                f"{'LONG' if pdf_trend['bias'] > 0 else 'SHORT' if pdf_trend['bias'] < 0 else 'FLAT'}."
            ),
        },
    ]
    if journal_logged and journal_entry is not None:
        events.insert(
            1,
            {
                "time": now_et.replace(" ET", ""),
                "message": (
                    f"Trade journal logged {_sig_label(next_signal)} "
                    f"@ {journal_entry['entry_reference']:.2f} for {journal_entry['execute_at_et']}."
                ),
            },
        )

    watchlist = _fetch_watchlist()

    health = {
        "model": "Healthy" if kx_status == "Integrated" else "Degraded",
        "feed": "Synced",
        "drift": f"{(total_return_pct - expected_session_return_pct):+.2f}%",
    }

    filters = [
        {"label": "US Session", "active": True},
        {"label": "News Lock", "active": False},
        {"label": "Volatility Gate", "active": True},
    ]

    headline = (
        "Short-term continuation favored while volatility compresses."
        if expected_session_return_pct >= 0
        else "Momentum pressure rising as downside volatility expands."
    )

    subheadline = (
        "Live NQ feed blended with NVIDIA QPO scenario optimization and KX distillation backtest conventions."
    )

    return {
        "meta": {
            "symbol": "NQ=F",
            "as_of_et": _format_et(latest_ts),
            "live_last_bar_et": _format_et(live_latest_ts),
            "evaluation_mode": "last_closed_session" if FREEZE_TO_LAST_CLOSED_SESSION else "live",
            "range_5m": BACKTEST_RANGE_5M,
            "n_bars": int(len(returns)),
            "window_start_et": backtest_window_start,
            "window_end_et": backtest_window_end,
            "last_price": round(close[-1], 2),
        },
        "headline": headline,
        "subheadline": subheadline,
        "watchlist": watchlist,
        "filters": filters,
        "health": health,
        "risk": {"account_size": ACCOUNT_SIZE, "allocated_pct": round((wl + ws) * 100, 1)},
        "execution_plan": execution_plan,
        "kpis": {
            "forecast_confidence_pct": float(forecast_confidence_pct),
            "expected_session_return_pct": float(expected_session_return_pct),
            "projected_max_drawdown_pct": float(projected_max_drawdown_pct),
            "sharpe_rolling_20": float(sharpe),
            "max_drawdown_pct": float(max_drawdown_pct),
            "total_return_pct": float(total_return_pct),
            "win_rate_pct": float(win_rate_pct),
            "n_trades": int(n_trades),
            "trades_per_day": float(trades_per_trading_session),
            "avg_hold_minutes": float(trade_history_summary.get("avg_hold_minutes", 0.0)),
            "max_hold_minutes_observed": float(trade_history_summary.get("max_hold_minutes", 0.0)),
            "max_hold_violation_count": int(constraint_checks.get("max_hold_violation_count", 0)),
            "loss_cap_violation_count": int(constraint_checks.get("loss_cap_violation_count", 0)),
            "cross_session_violation_count": int(constraint_checks.get("cross_session_violation_count", 0)),
            "winning_trades_lifetime": int(trade_history_summary["winning_trades"]),
            "losing_trades_lifetime": int(trade_history_summary["losing_trades"]),
            "total_pnl_lifetime_pct": float(trade_history_summary["total_pnl_pct"]),
            "total_pnl_lifetime_usd": float(trade_history_summary["total_pnl_usd"]),
        },
        "forecast": {
            "points": forecast_points,
            "horizon_label": "5m horizon • live feed • cost-adjusted",
        },
        "regimes": regimes,
        "allocation": {"rows": allocation_rows},
        "events": events,
        "signals": signals,
        "trade_journal": {
            "path": str(TRADE_JOURNAL_JSONL),
            "csv_path": str(TRADE_JOURNAL_CSV),
            "summary": trade_journal_summary,
            "all": trade_journal_rows,
            "recent": trade_journal_rows,
        },
        "trade_history": {
            "path": str(TRADE_HISTORY_JSONL),
            "csv_path": str(TRADE_HISTORY_CSV),
            "summary": trade_history_summary,
            "all": trade_history_rows,
            "recent": trade_history_rows,
        },
        "constraints": {
            "max_hold_minutes": int(constraint_checks.get("max_hold_minutes", LIVE_SIGNAL_CONFIG.get("max_hold_minutes", 90))),
            "max_active_trade_loss_usd": float(
                constraint_checks.get("max_active_trade_loss_usd", LIVE_SIGNAL_CONFIG.get("max_active_trade_loss_usd", 500.0))
            ),
            "max_hold_violation_count": int(constraint_checks.get("max_hold_violation_count", 0)),
            "loss_cap_violation_count": int(constraint_checks.get("loss_cap_violation_count", 0)),
            "cross_session_violation_count": int(constraint_checks.get("cross_session_violation_count", 0)),
            "exit_reason_distribution": dict(constraint_checks.get("exit_reason_distribution", {})),
        },
        "signal_mix": signal_mix,
        "signal_mix_lookback_bars": int(SIGNAL_MIX_LOOKBACK_BARS),
        "config": {"cvar_alpha": alpha, "risk_aversion": risk_aversion},
        "pdf_trend": pdf_trend,
        "repo_sources": [
            {
                "name": "NVIDIA QPO",
                "status": qpo_status,
                "path": str(QPO_DIR),
                "detail": "LOG returns + Gaussian scenarios + CVaR objective",
            },
            {
                "name": "KX Distillation",
                "status": kx_status,
                "path": str(KX_FINGPT_FILE.parent),
                "detail": "FinGPT labels + backtest_config + BUY/SELL/HOLD mapping",
            },
        ],
        "distillation_stats": kx_overlay,
        "kx_confluence": kx_confluence,
        "portfolio_optimization_stats": qpo_overlay,
    }


@lru_cache(maxsize=1)
def _cached_payload_30s(bucket: int) -> dict[str, Any]:
    del bucket
    return _build_payload()


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/dashboard"):
            try:
                bucket = int(time.time() // 30)
                payload = _cached_payload_30s(bucket)
                self._json(payload)
            except (HTTPError, URLError) as e:
                self._json({"error": f"data_source_error: {e}"}, status=502)
            except Exception as e:
                self._json({"error": f"internal_error: {e}"}, status=500)
            return

        if self.path in {"/", "/index.html"}:
            self.path = "/index.html"
        return super().do_GET()


def main() -> None:
    server = ThreadingHTTPServer(("", PORT), DashboardHandler)
    print(f"Serving dashboard + API at http://localhost:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
