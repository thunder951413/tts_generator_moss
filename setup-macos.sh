#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="$ROOT/.runtime"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
FASTER_REF="a70afc0f81f7f5f8801c3227968f1102f43f211c"
WRAPPER_REF="b0b2da11293fb5a3f84fafc0a4c64524d7635b88"
QWENTTS_REF="7df559a8ca25f66fee02970514ebe5f01dee9055"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This installer requires an Apple Silicon Mac (Darwin arm64)." >&2
  exit 1
fi

for command in git cmake ninja ffmpeg "$PYTHON_BIN"; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing $command. Install prerequisites with:" >&2
    echo "  brew install python@3.12 cmake ninja ffmpeg libsndfile" >&2
    exit 1
  fi
done

mkdir -p "$RUNTIME"
"$PYTHON_BIN" -m venv "$ROOT/.venv"
PYTHON="$ROOT/.venv/bin/python"
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install -r "$ROOT/requirements-macos.txt"

clone_at_ref() {
  local url="$1"
  local directory="$2"
  local ref="$3"
  if [[ ! -d "$directory/.git" ]]; then
    git clone --recursive "$url" "$directory"
  fi
  git -C "$directory" fetch --tags origin
  git -C "$directory" checkout --detach "$ref"
  git -C "$directory" submodule update --init --recursive
}

clone_at_ref \
  "https://github.com/ServeurpersoCom/qwentts.cpp.git" \
  "$RUNTIME/qwentts.cpp" \
  "$QWENTTS_REF"
clone_at_ref \
  "https://github.com/andimarafioti/qwentts-cpp-python.git" \
  "$RUNTIME/qwentts-cpp-python" \
  "$WRAPPER_REF"
clone_at_ref \
  "https://github.com/andimarafioti/faster-qwen3-tts.git" \
  "$RUNTIME/faster-qwen3-tts" \
  "$FASTER_REF"

cmake \
  -S "$RUNTIME/qwentts.cpp" \
  -B "$RUNTIME/qwentts.cpp/build-metal" \
  -G Ninja \
  -DQWEN_SHARED=ON \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_ACCELERATE=ON \
  -DCMAKE_BUILD_TYPE=Release \
  '-DCMAKE_BUILD_RPATH=@loader_path' \
  '-DCMAKE_INSTALL_RPATH=@loader_path'
cmake --build "$RUNTIME/qwentts.cpp/build-metal" --target qwen -j "$(sysctl -n hw.logicalcpu)"

LIBQWEN="$RUNTIME/qwentts.cpp/build-metal/libqwen.dylib"
if [[ ! -f "$LIBQWEN" ]]; then
  echo "Metal build completed but libqwen.dylib was not found at $LIBQWEN" >&2
  exit 1
fi

"$PYTHON" -m pip install -e "$RUNTIME/qwentts-cpp-python"
"$PYTHON" -m pip install -e "$RUNTIME/faster-qwen3-tts"

QWENTTS_CPP_LIBRARY="$LIBQWEN" "$PYTHON" - <<'PY'
from qwentts_cpp import QwenLibrary

library = QwenLibrary()
print("qwentts.cpp ABI:", library.version())
PY

if [[ ! -f "$ROOT/.env.macos" ]]; then
  cp "$ROOT/.env.macos.example" "$ROOT/.env.macos"
  echo "Created .env.macos. Change QWEN_TTS_ACCESS_PASSWORD before remote access."
fi

echo
echo "Metal runtime is ready. Model weights are not stored in this repository."
echo "Start with: ./start-macos.sh"
