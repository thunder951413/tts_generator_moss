#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$ROOT/scripts/build_macos_app.sh"
open "$ROOT/dist/QwenTTS.app" --args --repo-root "$ROOT"
