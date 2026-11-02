#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$(mktemp -d)"
trap 'rm -rf "$TEST_DIR"' EXIT
SWIFT_SOURCES=()
for source in "$TASK_ROOT"/macos/*.swift; do
  [[ "$(basename "$source")" == "QwenReaderApp.swift" ]] && continue
  SWIFT_SOURCES+=("$source")
done
xcrun swiftc "${SWIFT_SOURCES[@]}" \
  -D QWEN_STUDIO_TESTS "$TASK_ROOT/tests/native_reference_rename.swift" \
  -parse-as-library -target "$(uname -m)-apple-macos14.0" \
  -o "$TEST_DIR/native-tests" \
  -framework AppKit -framework AVFoundation -framework CoreMedia \
  -framework CryptoKit -framework ScreenCaptureKit
"$TEST_DIR/native-tests"
xcrun swiftc "$TASK_ROOT/macos/QwenReaderApp.swift" \
  -parse-as-library -typecheck -target "$(uname -m)-apple-macos13.0" \
  -framework AppKit -framework CryptoKit -framework SwiftUI -framework WebKit
