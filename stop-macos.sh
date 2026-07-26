#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT/qwen-tts.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "Qwen3-TTS is not running."
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  for _ in {1..50}; do
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.2
  done
fi
rm -f "$PID_FILE"
echo "Qwen3-TTS stopped."
