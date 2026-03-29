#!/usr/bin/env bash
set -euo pipefail

NOTIFIER_PID_FILE="/tmp/nq_ntfy_notifier.pid"
API_PID_FILE="/tmp/nq_local_api.pid"

stop_pid_file() {
  local file="$1"
  local label="$2"
  if [[ ! -f "$file" ]]; then
    echo "$label: not running (no pid file)."
    return
  fi
  local pid
  pid="$(cat "$file" 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    rm -f "$file"
    echo "$label: empty pid file removed."
    return
  fi
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
    echo "$label: stopped pid $pid."
  else
    echo "$label: pid $pid not running."
  fi
  rm -f "$file"
}

stop_pid_file "$NOTIFIER_PID_FILE" "Notifier"
stop_pid_file "$API_PID_FILE" "Local API"
