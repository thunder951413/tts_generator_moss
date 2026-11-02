#!/usr/bin/env bash
# One-shot local verification: Python service tests, reader behaviour tests,
# native Swift behaviour tests and both macOS app builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
echo "== Python service tests"
"$PY" -m pytest tests/ -q
echo "== Reader behaviour tests"
node --test tests/test_reader_playback.cjs tests/test_reader_player.cjs tests/test_reader_library.cjs
echo "== Native macOS behaviour tests"
./scripts/test_native_macos.sh
if [[ "${1:-}" != "--no-build" ]]; then
  echo "== Build QwenTTS.app and QwenReader.app"
  ./scripts/build_macos_app.sh
  ./scripts/build_macos_reader.sh
fi
echo "All checks passed."
