#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT/qwen-tts.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "Qwen3-TTS is not running."
  exit 0
fi

PID="$(cat "$PID_FILE")"
if [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
  kill -TERM "$PID"
  for _ in {1..150}; do
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.2
  done
  if kill -0 "$PID" 2>/dev/null; then
    echo "Graceful shutdown timed out; terminating owned worker processes." >&2
    CHILDREN="$(pgrep -P "$PID" || true)"
    if [[ -n "$CHILDREN" ]]; then
      # shellcheck disable=SC2086
      kill -TERM $CHILDREN 2>/dev/null || true
    fi
    sleep 2
    kill -KILL "$PID" 2>/dev/null || true
    if [[ -n "$CHILDREN" ]]; then
      # shellcheck disable=SC2086
      kill -KILL $CHILDREN 2>/dev/null || true
    fi
  fi
fi
rm -f "$PID_FILE"
echo "Qwen3-TTS stopped."
