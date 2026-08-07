#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$ROOT/dist/QwenReader.app"
MACOS_DIR="$APP/Contents/MacOS"
RESOURCES_DIR="$APP/Contents/Resources"
ICON_SOURCE="$ROOT/web/novel_reader/assets/qwen-shengyue-icon.png"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "The macOS reader app can only be built on macOS." >&2
  exit 1
fi

mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"
cp "$ROOT/macos/QwenReaderInfo.plist" "$APP/Contents/Info.plist"
printf '%s\n' "$ROOT" > "$RESOURCES_DIR/repository-root.txt"

xcrun swiftc \
  "$ROOT/macos/QwenReaderApp.swift" \
  -o "$MACOS_DIR/QwenReader" \
  -parse-as-library \
  -target arm64-apple-macos13.0 \
  -framework AppKit \
  -framework CryptoKit \
  -framework SwiftUI \
  -framework WebKit

if [[ -f "$ICON_SOURCE" ]]; then
  ICONSET="$(mktemp -d)/QwenReader.iconset"
  mkdir -p "$ICONSET"
  sips -z 16 16 "$ICON_SOURCE" --out "$ICONSET/icon_16x16.png" >/dev/null
  sips -z 32 32 "$ICON_SOURCE" --out "$ICONSET/icon_16x16@2x.png" >/dev/null
  sips -z 32 32 "$ICON_SOURCE" --out "$ICONSET/icon_32x32.png" >/dev/null
  sips -z 64 64 "$ICON_SOURCE" --out "$ICONSET/icon_32x32@2x.png" >/dev/null
  sips -z 128 128 "$ICON_SOURCE" --out "$ICONSET/icon_128x128.png" >/dev/null
  sips -z 256 256 "$ICON_SOURCE" --out "$ICONSET/icon_128x128@2x.png" >/dev/null
  sips -z 256 256 "$ICON_SOURCE" --out "$ICONSET/icon_256x256.png" >/dev/null
  sips -z 512 512 "$ICON_SOURCE" --out "$ICONSET/icon_256x256@2x.png" >/dev/null
  sips -z 512 512 "$ICON_SOURCE" --out "$ICONSET/icon_512x512.png" >/dev/null
  sips -z 1024 1024 "$ICON_SOURCE" --out "$ICONSET/icon_512x512@2x.png" >/dev/null
  iconutil -c icns "$ICONSET" -o "$RESOURCES_DIR/QwenReader.icns"
fi

codesign --force --deep --sign - "$APP" >/dev/null
echo "Built $APP"
