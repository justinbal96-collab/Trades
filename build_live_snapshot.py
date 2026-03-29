#!/usr/bin/env python3
"""Build static dashboard snapshot + persistent trade history for GitHub Pages.

This script runs the existing strategy engine in server.py, writes a static
`data/dashboard.json`, and materializes closed-trade PnL history so the data pool
can keep growing over time when committed by GitHub Actions.
"""

from __future__ import annotations

import csv
import json
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
        return dt.isoformat()
    except Exception:
        return None


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
                "execute_at_et": row.get("execute_at_et") or _et_iso_from_unix(execute_unix),
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


def _write_closed_trade_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trade_id",
        "entry_time_et",
        "entry_time_unix",
        "exit_time_et",
        "exit_time_unix",
        "side",
        "entry_price",
        "exit_price",
        "nq_contracts",
        "mnq_contracts",
        "point_value_usd",
        "pnl_points",
        "trade_pnl_usd",
        "total_pnl_usd",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRADE_LOG_DIR.mkdir(parents=True, exist_ok=True)

    payload = server._build_payload()

    entry_rows = _load_jsonl(ENTRY_JSONL)
    closed_rows, closed_summary = _build_closed_trades(entry_rows)

    # Expose persistent lifecycle stats in payload (used by UI / downstream tools).
    trade_journal = payload.setdefault("trade_journal", {})
    trade_journal["path"] = "./trade_logs/nq_trade_journal.jsonl"
    trade_journal["csv_path"] = "./trade_logs/nq_trade_journal.csv"
    trade_journal["closed_summary"] = closed_summary
    trade_journal["closed_recent"] = closed_rows[-120:]
    trade_journal["history_csv_path"] = "./data/trade_history.csv"

    payload.setdefault("meta", {})["snapshot_generated_utc"] = datetime.now(tz=timezone.utc).isoformat()

    DASHBOARD_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    FALLBACK_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    history_payload = {
        "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
        "summary": closed_summary,
        "trades": closed_rows,
    }
    TRADE_HISTORY_JSON.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    _write_closed_trade_csv(TRADE_HISTORY_CSV, closed_rows)

    if ENTRY_CSV.exists():
        ENTRY_COPY_CSV.write_text(ENTRY_CSV.read_text(encoding="utf-8"), encoding="utf-8")

    print(
        "snapshot_ok",
        f"entries={len(entry_rows)}",
        f"closed={closed_summary['total_closed']}",
        f"wins={closed_summary['winning_trades']}",
        f"losses={closed_summary['losing_trades']}",
        f"total_pnl={closed_summary['total_pnl_usd']:.2f}",
    )


if __name__ == "__main__":
    main()
