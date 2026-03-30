#!/usr/bin/env python3
"""Build static dashboard snapshot + persistent trade history for GitHub Pages.

This script runs the existing strategy engine in server.py, writes a static
`data/dashboard.json`, and materializes closed-trade PnL history so the data pool
can keep growing over time when committed by GitHub Actions.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRADE_LOG_DIR = ROOT / "trade_logs"
ENTRY_JSONL = TRADE_LOG_DIR / "nq_trade_journal.jsonl"
ENTRY_CSV = TRADE_LOG_DIR / "nq_trade_journal.csv"
DASHBOARD_JSON = DATA_DIR / "dashboard.json"
FALLBACK_JSON = ROOT / "dashboard-fallback.json"
TRADE_HISTORY_JSON = DATA_DIR / "trade_history.json"
TRADE_HISTORY_CSV = DATA_DIR / "trade_history.csv"
ENTRY_COPY_CSV = DATA_DIR / "trade_entry_journal.csv"

NQ_POINT_VALUE_USD = 20.0
MNQ_POINT_VALUE_USD = 2.0

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep published snapshots pinned to the validated benchmark setup.
os.environ.setdefault("BACKTEST_RANGE_5M", "60d")
os.environ.setdefault("QPO_RANGE_5M", "60d")

import server  # noqa: E402


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _et_iso_from_unix(unix_ts: int | None) -> str | None:
    if not unix_ts:
        return None
    try:
        dt = datetime.fromtimestamp(int(unix_ts), tz=timezone.utc).astimezone(server.ET_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S ET")
    except Exception:
        return None


def _normalize_journal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        execute_unix = _safe_int(item.get("execute_at_unix"), 0)
        if execute_unix > 0:
            item["execute_at_et"] = _et_iso_from_unix(execute_unix) or item.get("execute_at_et")
        logged_utc = str(item.get("logged_at_utc", "") or "").strip()
        if logged_utc:
            try:
                dt = datetime.fromisoformat(logged_utc.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                item["logged_at_et"] = dt.astimezone(server.ET_TZ).strftime("%Y-%m-%d %H:%M:%S ET")
            except Exception:
                pass
        out.append(item)
    return out


def _normalize_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        entry_unix = _safe_int(item.get("entry_unix", item.get("entry_time_unix")), 0)
        if entry_unix > 0:
            entry_label = _et_iso_from_unix(entry_unix)
            if entry_label:
                item["entry_unix"] = entry_unix
                item["entry_time_et"] = entry_label
                item["entry_et"] = entry_label
        exit_unix = _safe_int(item.get("exit_unix", item.get("exit_time_unix")), 0)
        if exit_unix > 0:
            exit_label = _et_iso_from_unix(exit_unix)
            if exit_label:
                item["exit_unix"] = exit_unix
                item["exit_time_et"] = exit_label
                item["exit_et"] = exit_label
        out.append(item)
    return out


def _build_closed_trades(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Normalize directional events first.
    events: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    for row in entries:
        action = str(row.get("action", "")).upper().strip()
        if action not in {"BUY", "SELL"}:
            continue

        event_id = str(row.get("event_id", "")).strip()
        if not event_id:
            continue
        if event_id in seen_event_ids:
            continue
        seen_event_ids.add(event_id)

        execute_unix = _safe_int(row.get("execute_at_unix"), 0)
        entry_price = _safe_float(row.get("entry_reference"), 0.0)
        if execute_unix <= 0 or entry_price <= 0:
            continue

        nq_contracts = max(0, _safe_int(row.get("nq_contracts"), 0))
        mnq_contracts = max(0, _safe_int(row.get("mnq_contracts"), 0))

        events.append(
            {
                "event_id": event_id,
                "action": action,
                "side": 1 if action == "BUY" else -1,
                "execute_at_unix": execute_unix,
                "execute_at_et": _et_iso_from_unix(execute_unix) or row.get("execute_at_et"),
                "entry_price": entry_price,
                "nq_contracts": nq_contracts,
                "mnq_contracts": mnq_contracts,
            }
        )

    events.sort(key=lambda x: (x["execute_at_unix"], x["event_id"]))

    closed: list[dict[str, Any]] = []
    open_trade: dict[str, Any] | None = None
    cumulative_pnl = 0.0

    for ev in events:
        point_value = (ev["nq_contracts"] * NQ_POINT_VALUE_USD) + (ev["mnq_contracts"] * MNQ_POINT_VALUE_USD)
        # If sizing is unavailable, keep conservative fallback to 1 MNQ for continuity.
        if point_value <= 0:
            point_value = MNQ_POINT_VALUE_USD

        if open_trade is None:
            open_trade = {
                "entry_time_unix": ev["execute_at_unix"],
                "entry_time_et": ev["execute_at_et"],
                "entry_side": ev["action"],
                "entry_direction": ev["side"],
                "entry_price": ev["entry_price"],
                "nq_contracts": ev["nq_contracts"],
                "mnq_contracts": ev["mnq_contracts"],
                "point_value_usd": point_value,
                "entry_event_id": ev["event_id"],
            }
            continue

        # Same-side signal refresh: keep current open position unchanged.
        if ev["side"] == open_trade["entry_direction"]:
            continue

        pnl_points = (ev["entry_price"] - open_trade["entry_price"]) * open_trade["entry_direction"]
        trade_pnl = pnl_points * open_trade["point_value_usd"]
        cumulative_pnl += trade_pnl

        trade_id = f"{open_trade['entry_event_id']}->{ev['event_id']}"
        closed_row = {
            "trade_id": trade_id,
            "entry_time_et": open_trade["entry_time_et"],
            "entry_time_unix": open_trade["entry_time_unix"],
            "exit_time_et": ev["execute_at_et"],
            "exit_time_unix": ev["execute_at_unix"],
            "side": open_trade["entry_side"],
            "entry_price": round(open_trade["entry_price"], 2),
            "exit_price": round(ev["entry_price"], 2),
            "nq_contracts": int(open_trade["nq_contracts"]),
            "mnq_contracts": int(open_trade["mnq_contracts"]),
            "point_value_usd": round(open_trade["point_value_usd"], 2),
            "pnl_points": round(pnl_points, 2),
            "trade_pnl_usd": round(trade_pnl, 2),
            "total_pnl_usd": round(cumulative_pnl, 2),
        }
        closed.append(closed_row)

        open_trade = {
            "entry_time_unix": ev["execute_at_unix"],
            "entry_time_et": ev["execute_at_et"],
            "entry_side": ev["action"],
            "entry_direction": ev["side"],
            "entry_price": ev["entry_price"],
            "nq_contracts": ev["nq_contracts"],
            "mnq_contracts": ev["mnq_contracts"],
            "point_value_usd": point_value,
            "entry_event_id": ev["event_id"],
        }

    wins = sum(1 for row in closed if row["trade_pnl_usd"] > 0)
    losses = sum(1 for row in closed if row["trade_pnl_usd"] < 0)
    summary = {
        "total_closed": len(closed),
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate_pct": round((wins / len(closed) * 100.0), 2) if closed else 0.0,
        "total_pnl_usd": round(sum(row["trade_pnl_usd"] for row in closed), 2),
        "latest_total_pnl_usd": round(closed[-1]["total_pnl_usd"], 2) if closed else 0.0,
        "open_trade": open_trade,
    }

    return closed, summary


def _write_trade_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trade_id",
        "entry_time_unix",
        "entry_unix",
        "entry_time_et",
        "entry_et",
        "exit_time_et",
        "exit_time_unix",
        "exit_unix",
        "exit_et",
        "side",
        "direction",
        "entry_price",
        "exit_price",
        "nq_contracts",
        "mnq_contracts",
        "bars_held",
        "point_value_usd",
        "pnl_points",
        "pnl_pct",
        "trade_pnl_usd",
        "pnl_usd",
        "trade_profit_pct",
        "trade_profit_usd",
        "total_pnl_usd",
        "cumulative_pnl_pct",
        "cumulative_pnl_usd",
        "result",
        "updated_at_utc",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize_trade_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "avg_pnl_pct": 0.0,
            "total_pnl_pct": 0.0,
            "avg_pnl_usd": 0.0,
            "total_pnl_usd": 0.0,
            "first_entry_et": None,
            "last_exit_et": None,
        }

    pnl_pct_vals: list[float] = []
    pnl_usd_vals: list[float] = []
    for row in rows:
        pnl_pct = _safe_float(row.get("pnl_pct", row.get("trade_profit_pct", 0.0)), 0.0)
        pnl_usd = _safe_float(row.get("pnl_usd", row.get("trade_profit_usd", 0.0)), 0.0)
        pnl_pct_vals.append(pnl_pct)
        pnl_usd_vals.append(pnl_usd)

    wins = sum(1 for x in pnl_pct_vals if x > 0)
    losses = len(pnl_pct_vals) - wins

    return {
        "total_trades": len(rows),
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate_pct": round((wins / len(rows)) * 100.0, 4) if rows else 0.0,
        "avg_pnl_pct": round(sum(pnl_pct_vals) / len(pnl_pct_vals), 8),
        "total_pnl_pct": round(sum(pnl_pct_vals), 8),
        "avg_pnl_usd": round(sum(pnl_usd_vals) / len(pnl_usd_vals), 4),
        "total_pnl_usd": round(sum(pnl_usd_vals), 4),
        "first_entry_et": rows[0].get("entry_time_et", rows[0].get("entry_et")),
        "last_exit_et": rows[-1].get("exit_time_et", rows[-1].get("exit_et")),
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRADE_LOG_DIR.mkdir(parents=True, exist_ok=True)

    payload = server._build_payload()

    trade_journal = payload.setdefault("trade_journal", {})
    journal_rows = trade_journal.get("all")
    if not isinstance(journal_rows, list):
        journal_rows = _load_jsonl(ENTRY_JSONL)
    journal_rows = _normalize_journal_rows(journal_rows)
    trade_journal["path"] = "./trade_logs/nq_trade_journal.jsonl"
    trade_journal["csv_path"] = "./trade_logs/nq_trade_journal.csv"
    trade_journal["all"] = journal_rows
    trade_journal["recent"] = journal_rows

    trade_history = payload.setdefault("trade_history", {})
    history_rows = trade_history.get("all")
    if not isinstance(history_rows, list):
        history_rows = trade_history.get("recent")
    if not isinstance(history_rows, list) or not history_rows:
        # Fallback path if upstream payload does not expose trade_history yet.
        closed_rows, _ = _build_closed_trades(journal_rows)
        history_rows = closed_rows
    history_rows = _normalize_history_rows(history_rows)
    history_summary = trade_history.get("summary")
    if not isinstance(history_summary, dict) or not history_summary:
        history_summary = _summarize_trade_rows(history_rows)

    trade_history["path"] = "./trade_logs/nq_trade_history.jsonl"
    trade_history["csv_path"] = "./trade_logs/nq_trade_history.csv"
    trade_history["all"] = history_rows
    trade_history["recent"] = history_rows
    trade_history["summary"] = history_summary

    payload.setdefault("meta", {})["snapshot_generated_utc"] = datetime.now(tz=timezone.utc).isoformat()

    DASHBOARD_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    FALLBACK_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    trade_history_payload = {
        "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
        "summary": history_summary,
        "trades": history_rows,
    }
    TRADE_HISTORY_JSON.write_text(json.dumps(trade_history_payload, indent=2), encoding="utf-8")
    _write_trade_history_csv(TRADE_HISTORY_CSV, history_rows)

    if ENTRY_CSV.exists():
        ENTRY_COPY_CSV.write_text(ENTRY_CSV.read_text(encoding="utf-8"), encoding="utf-8")

    print(
        "snapshot_ok",
        f"journal_entries={len(journal_rows)}",
        f"history_trades={len(history_rows)}",
        f"wins={history_summary.get('winning_trades', 0)}",
        f"losses={history_summary.get('losing_trades', 0)}",
        f"total_pnl_usd={float(history_summary.get('total_pnl_usd', 0.0)):.2f}",
    )


if __name__ == "__main__":
    main()
