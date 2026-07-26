#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT/qwen-tts.pid"
LOG_DIR="$ROOT/logs"
LIB_DIR="$ROOT/.runtime/qwentts.cpp/build-metal"
LIBQWEN="$LIB_DIR/libqwen.dylib"

if [[ -f "$ROOT/.env.macos" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env.macos"
  set +a
fi

if [[ ! -x "$ROOT/.venv/bin/python" || ! -f "$LIBQWEN" ]]; then
  echo "Runtime is missing. Run ./setup-macos.sh first." >&2
  exit 1
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Qwen3-TTS is already running with PID $(cat "$PID_FILE")."
  exit 0
fi

mkdir -p "$LOG_DIR"
export QWEN_TTS_PYTHON="$ROOT/.venv/bin/python"
export QWEN_TTS_BACKEND="ggml"
export QWENTTS_CPP_LIBRARY="$LIBQWEN"
export DYLD_LIBRARY_PATH="$LIB_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

nohup "$ROOT/.venv/bin/python" "$ROOT/clis/qwen_tts_app.py" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-7861}" \
  --qwen-backend ggml \
  --qwen-quant "${QWEN_TTS_QUANT:-Q4_K_M}" \
  --qwentts-library "$LIBQWEN" \
  >"$LOG_DIR/service.out.log" \
  2>"$LOG_DIR/service.err.log" \
  </dev/null &

echo "$!" >"$PID_FILE"
echo "Qwen3-TTS started with PID $! at http://${HOST:-0.0.0.0}:${PORT:-7861}"
