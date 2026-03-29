#!/usr/bin/env python3
"""Send NQ entry alerts via SMS, WhatsApp (Twilio), or ntfy push from dashboard API.

Usage:
  python3 scripts/sms_trade_notifier.py --once --dry-run
  python3 scripts/sms_trade_notifier.py

Required env vars for Twilio sends:
  TWILIO_ACCOUNT_SID
  TWILIO_AUTH_TOKEN
  TWILIO_FROM_NUMBER (for SMS)
  TWILIO_TO_NUMBER

Optional env vars:
  NQ_ALERT_CHANNEL=sms|whatsapp|ntfy (default: sms)
  TWILIO_WHATSAPP_FROM=+1... (or already prefixed with whatsapp:)
  NTFY_SERVER=https://ntfy.sh
  NTFY_TOPIC=<your-topic>
  NTFY_TITLE=<optional title>
  NTFY_PRIORITY=1..5
  NTFY_TAGS=chart_with_upwards_trend,rotating_light
  NTFY_TOKEN=<optional bearer token for protected topic>
  NQ_ALERT_FORMAT=compact|full (default: compact)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from zoneinfo import ZoneInfo

    NY_TZ = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    NY_TZ = timezone.utc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _fetch_dashboard(api_url: str, timeout: float = 10.0) -> dict[str, Any]:
    req = Request(api_url, headers={"User-Agent": "nq-sms-notifier/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _to_nyc_label(value: Any) -> str:
    text = str(value).strip()
    if not text or text == "None":
        return "--"

    # Incoming values are typically like "16:55:00 ET"; normalize to explicit NY tz.
    clock = text.split()[0]
    try:
        parts = [int(p) for p in clock.split(":")]
        if len(parts) < 2:
            raise ValueError("missing minute field")
        sec = parts[2] if len(parts) > 2 else 0
        now_ny = datetime.now(tz=timezone.utc).astimezone(NY_TZ)
        stamp = now_ny.replace(hour=parts[0], minute=parts[1], second=sec, microsecond=0)
        return stamp.strftime("%H:%M:%S %Z")
    except Exception:
        if text.endswith(" ET"):
            zone = datetime.now(tz=timezone.utc).astimezone(NY_TZ).strftime("%Z")
            return f"{text[:-3]} {zone}"
        return text


def _latest_trade_journal_entry(payload: dict[str, Any]) -> dict[str, Any] | None:
    journal = payload.get("trade_journal", {})
    recent = journal.get("recent", [])
    if not isinstance(recent, list) or not recent:
        return None
    last = recent[-1]
    if isinstance(last, dict):
        return last
    return None


def _build_alert_text(
    payload: dict[str, Any],
    *,
    fmt: str = "compact",
    journal_entry: dict[str, Any] | None = None,
) -> tuple[str, str]:
    plan = payload.get("execution_plan", {})
    checks = plan.get("prop_rules", {}).get("checks", [])
    trades_today = int(plan.get("prop_rules", {}).get("trades_today", 0))
    max_trades = int(plan.get("prop_rules", {}).get("max_trades_per_day", 0))

    use_journal = bool(journal_entry and str(journal_entry.get("event_id", "")).strip())

    if use_journal:
        action = str(journal_entry.get("action", "HOLD"))
        next_bar = _to_nyc_label(journal_entry.get("execute_at_et", "--"))
        as_of = _to_nyc_label(journal_entry.get("signal_time_et", "--"))
        entry = journal_entry.get("entry_reference")
        stop = journal_entry.get("stop_price")
        target = journal_entry.get("target_price")
        risk_usd = journal_entry.get("risk_per_trade_usd")
        nq = int(journal_entry.get("nq_contracts", 0) or 0)
        mnq = int(journal_entry.get("mnq_contracts", 0) or 0)
        model_ok = bool(journal_entry.get("eligible", False))
    else:
        action = str(plan.get("action_next_bar", "HOLD"))
        next_bar = _to_nyc_label(plan.get("next_bar_et", "--"))
        as_of = _to_nyc_label(plan.get("as_of_et", "--"))
        entry = plan.get("entry_reference")
        stop = plan.get("stop_price")
        target = plan.get("target_price")
        risk_usd = plan.get("risk_per_trade_usd")
        contracts = plan.get("contract_plan", {})
        nq = int(contracts.get("nq", 0))
        mnq = int(contracts.get("mnq", 0))
        model_ok = bool(plan.get("eligible", False))

    if action == "BUY":
        action_title = "ACTION: BUY (LONG)"
        plan_text = "Plan: Buy next bar open"
        action_short = "BUY"
    elif action == "SELL":
        action_title = "ACTION: SELL (SHORT)"
        plan_text = "Plan: Sell next bar open"
        action_short = "SELL"
    else:
        action_title = "ACTION: HOLD"
        plan_text = "Plan: No new trade"
        action_short = "HOLD"

    signal_detail = ""
    if checks:
        first = checks[0]
        signal_detail = str(first.get("detail", "")).strip()

    if str(fmt).strip().lower() == "full":
        lines = [
            f"{action_title} ({next_bar})",
            f"As of {as_of}",
            plan_text,
            f"Entry {entry} | SL {stop} | TP {target}",
            f"Size NQ {nq} / MNQ {mnq}",
            f"Risk ${risk_usd} | Eligible {'YES' if model_ok else 'NO'}",
            f"Trades today {trades_today}/{max_trades}",
        ]
        if use_journal:
            lines.append("Source: trade_journal event")
        if signal_detail:
            lines.append(signal_detail)
    else:
        lines = [
            f"{action_short} {next_bar}",
            f"E {entry} | SL {stop} | TP {target}",
            f"NQ {nq} / MNQ {mnq} | Risk ${risk_usd}",
            f"Eligible {'YES' if model_ok else 'NO'} | Trades {trades_today}/{max_trades}",
        ]
        if use_journal:
            lines.append("Source trade_journal")

    message = "\n".join(lines)
    if use_journal:
        alert_id = f"journal|{journal_entry.get('event_id')}"
    else:
        alert_id = f"{action}|{next_bar}|{entry}|{stop}|{target}|{nq}|{mnq}"
    return alert_id, message


def _as_whatsapp_address(number_or_address: str) -> str:
    text = str(number_or_address).strip()
    if text.lower().startswith("whatsapp:"):
        return text
    return f"whatsapp:{text}"


def _twilio_send_message(body: str, sid: str, token: str, from_number: str, to_number: str) -> dict[str, Any]:
    form = urlencode({"From": from_number, "To": to_number, "Body": body}).encode("utf-8")
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    auth = base64.b64encode(f"{sid}:{token}".encode("utf-8")).decode("ascii")
    req = Request(
        url,
        data=form,
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "nq-sms-notifier/1.0",
        },
        method="POST",
    )
    with urlopen(req, timeout=15.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ntfy_publish_message(
    body: str,
    *,
    server: str,
    topic: str,
    title: str | None = None,
    priority: str | None = None,
    tags: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    clean_server = str(server or "https://ntfy.sh").strip().rstrip("/")
    clean_topic = str(topic or "").strip().strip("/")
    if not clean_topic:
        raise RuntimeError("Missing NTFY_TOPIC for ntfy channel")

    url = f"{clean_server}/{clean_topic}"
    headers: dict[str, str] = {"User-Agent": "nq-sms-notifier/1.0"}
    if title:
        headers["Title"] = title
    if priority:
        headers["Priority"] = str(priority).strip()
    if tags:
        headers["Tags"] = str(tags).strip()
    if token:
        headers["Authorization"] = f"Bearer {str(token).strip()}"

    req = Request(
        url,
        data=body.encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(req, timeout=15.0) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"status": "ok", "raw": raw}


def _should_send(
    payload: dict[str, Any],
    state: dict[str, Any],
    require_eligible: bool,
    send_every_directional: bool,
) -> tuple[bool, str, dict[str, Any] | None]:
    latest_journal = _latest_trade_journal_entry(payload)
    if latest_journal is not None:
        journal_event_id = str(latest_journal.get("event_id", "")).strip()
        if journal_event_id:
            last_journal_event_id = str(state.get("last_journal_event_id", "")).strip()
            if not last_journal_event_id:
                # Backward compatibility with older state files that only tracked alert_id.
                last_alert_id = str(state.get("last_alert_id", "")).strip()
                if last_alert_id.startswith("journal|"):
                    last_journal_event_id = last_alert_id.split("journal|", 1)[1]
            journal_action = str(latest_journal.get("action", "")).upper()
            if require_eligible and not bool(latest_journal.get("eligible", False)):
                return False, "latest journal setup not eligible", latest_journal
            if journal_action not in {"BUY", "SELL"}:
                return False, "latest journal event is non-directional", latest_journal
            if journal_event_id != last_journal_event_id:
                return True, "new trade_journal event", latest_journal
            return False, "trade_journal event already alerted", latest_journal

    plan = payload.get("execution_plan", {})
    action = str(plan.get("action_next_bar", "HOLD"))
    signal_changed = bool(plan.get("signal_changed", False))
    eligible = bool(plan.get("eligible", False))

    if action not in {"BUY", "SELL"}:
        return False, "action is HOLD", None
    if require_eligible and not eligible:
        return False, "setup not eligible", None

    if send_every_directional:
        return True, "directional setup", None
    if signal_changed:
        return True, "signal changed", None
    if not state.get("last_action"):
        return True, "first run directional setup", None
    if state.get("last_action") != action:
        return True, "action changed", None
    return False, "no new setup", None


def main() -> None:
    parser = argparse.ArgumentParser(description="Send NQ entry alerts by SMS, WhatsApp, or ntfy.")
    parser.add_argument("--api-url", default=os.environ.get("NQ_DASHBOARD_API_URL", "http://localhost:8080/api/dashboard"))
    parser.add_argument("--poll-sec", type=int, default=int(os.environ.get("NQ_ALERT_POLL_SEC", "20")))
    parser.add_argument("--once", action="store_true", help="Run one poll cycle and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print alert text without sending SMS.")
    parser.add_argument(
        "--state-file",
        default=os.environ.get("NQ_ALERT_STATE_FILE", "/tmp/nq_sms_alert_state.json"),
        help="Path for dedupe state.",
    )
    parser.add_argument(
        "--send-every-directional",
        action="store_true",
        default=_env_bool("NQ_SEND_EVERY_DIRECTIONAL", False),
        help="Send each directional setup, not just changes.",
    )
    parser.add_argument(
        "--allow-ineligible",
        action="store_true",
        default=not _env_bool("NQ_REQUIRE_ELIGIBLE", True),
        help="Send alerts even when eligibility checks fail.",
    )
    parser.add_argument(
        "--channel",
        choices=["sms", "whatsapp", "ntfy"],
        default=os.environ.get("NQ_ALERT_CHANNEL", "sms").strip().lower(),
        help="Delivery channel.",
    )
    parser.add_argument(
        "--format",
        choices=["compact", "full"],
        default=os.environ.get("NQ_ALERT_FORMAT", "compact").strip().lower(),
        help="Alert text format.",
    )
    args = parser.parse_args()

    require_eligible = not args.allow_ineligible
    state_path = Path(args.state_file)
    state = _load_state(state_path)

    sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    from_number = os.environ.get("TWILIO_FROM_NUMBER", "")
    to_number = os.environ.get("TWILIO_TO_NUMBER", "")
    whatsapp_from = os.environ.get("TWILIO_WHATSAPP_FROM", "")
    ntfy_server = os.environ.get("NTFY_SERVER", "https://ntfy.sh")
    ntfy_topic = os.environ.get("NTFY_TOPIC", "")
    ntfy_title = os.environ.get("NTFY_TITLE", "NQ Trade Alert")
    ntfy_priority = os.environ.get("NTFY_PRIORITY", "4")
    ntfy_tags = os.environ.get("NTFY_TAGS", "chart_with_upwards_trend")
    ntfy_token = os.environ.get("NTFY_TOKEN", "")

    while True:
        try:
            payload = _fetch_dashboard(args.api_url)
            should_send, reason, latest_journal = _should_send(
                payload=payload,
                state=state,
                require_eligible=require_eligible,
                send_every_directional=bool(args.send_every_directional),
            )
            alert_id, message = _build_alert_text(payload, fmt=args.format, journal_entry=latest_journal)
            action = str(payload.get("execution_plan", {}).get("action_next_bar", "HOLD"))

            if should_send and state.get("last_alert_id") != alert_id:
                if args.dry_run:
                    print(f"[DRY-RUN] would send {args.channel.upper()} message:")
                    print(message)
                else:
                    if args.channel == "ntfy":
                        resp = _ntfy_publish_message(
                            message,
                            server=ntfy_server,
                            topic=ntfy_topic,
                            title=ntfy_title,
                            priority=ntfy_priority,
                            tags=ntfy_tags,
                            token=ntfy_token,
                        )
                        print(
                            "[SENT] ntfy"
                            f" id={resp.get('id', 'unknown')}"
                            f" event={resp.get('event', 'message')}"
                        )
                    else:
                        send_from = from_number
                        send_to = to_number
                        if args.channel == "whatsapp":
                            from_base = whatsapp_from or from_number
                            send_from = _as_whatsapp_address(from_base)
                            send_to = _as_whatsapp_address(to_number)

                        required = {
                            "TWILIO_ACCOUNT_SID": sid,
                            "TWILIO_AUTH_TOKEN": token,
                            "TWILIO_TO_NUMBER": to_number,
                        }
                        if args.channel == "sms":
                            required["TWILIO_FROM_NUMBER"] = from_number
                        else:
                            required["TWILIO_WHATSAPP_FROM or TWILIO_FROM_NUMBER"] = (whatsapp_from or from_number)

                        missing = [k for k, v in required.items() if not v]
                        if missing:
                            raise RuntimeError(f"Missing env vars for Twilio send: {', '.join(missing)}")
                        resp = _twilio_send_message(
                            body=message,
                            sid=sid,
                            token=token,
                            from_number=send_from,
                            to_number=send_to,
                        )
                        print(f"[SENT] sid={resp.get('sid', 'unknown')} status={resp.get('status', 'unknown')}")

                state["last_alert_id"] = alert_id
                state["last_alert_reason"] = reason
                state["last_sent_at_unix"] = int(time.time())
                if latest_journal is not None:
                    journal_event_id = str(latest_journal.get("event_id", "")).strip()
                    if journal_event_id:
                        state["last_journal_event_id"] = journal_event_id
            else:
                print(f"[SKIP] {reason}")

            state["last_action"] = action
            state["last_seen_at_unix"] = int(time.time())
            _save_state(state_path, state)
        except Exception as exc:
            print(f"[ERROR] {exc}")

        if args.once:
            break
        time.sleep(max(5, int(args.poll_sec)))


if __name__ == "__main__":
    main()
