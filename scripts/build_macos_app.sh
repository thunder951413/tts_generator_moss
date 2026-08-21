#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$ROOT/dist/QwenTTS.app"
MACOS_DIR="$APP/Contents/MacOS"
RESOURCES_DIR="$APP/Contents/Resources"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "The macOS desktop app can only be built on macOS." >&2
  exit 1
fi

mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"
cp "$ROOT/macos/Info.plist" "$APP/Contents/Info.plist"
printf '%s\n' "$ROOT" > "$RESOURCES_DIR/repository-root.txt"
xcrun swiftc \
  "$ROOT/macos/QwenTTSApp.swift" \
  "$ROOT/macos/StudioDesignSystem.swift" \
  "$ROOT/macos/NativeStudio.swift" \
  "$ROOT/macos/STTWorkbench.swift" \
  -o "$MACOS_DIR/QwenTTS" \
  -parse-as-library \
  -framework AppKit \
  -framework AVFoundation \
  -framework CoreMedia \
  -framework CryptoKit \
  -framework ScreenCaptureKit

# A local ad-hoc signature prevents Gatekeeper from treating a rebuilt bundle
# as an incomplete app while keeping distribution under the user's own account.
codesign --force --deep --sign - "$APP" >/dev/null
echo "Built $APP"
