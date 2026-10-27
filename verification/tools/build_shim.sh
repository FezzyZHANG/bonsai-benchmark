#!/bin/bash
# Build the hidden calibration ABI shim (llama.cpp behind libbonsai.so).
set -euo pipefail
BENCH="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
LLAMA_SRC="${LLAMA_SRC:-$BENCH/../bonsai-investigation/llama.cpp-prism}"
BUILD="$BENCH/verification/build/refshim"
NPROC="${NPROC:-$(nproc)}"
[ -d "$LLAMA_SRC" ] || { echo "LLAMA_SRC not found: $LLAMA_SRC" >&2; exit 1; }

cmake -S "$BENCH/verification/tools/refshim" -B "$BUILD" \
  -DLLAMA_SRC="$LLAMA_SRC" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --target bonsai -j "$NPROC"
# Hidden calibration artifact: never install this under deps/, which is mounted
# into the agent sandbox.
mkdir -p "$BENCH/verification/bin"
cp "$BUILD/libbonsai.so" "$BENCH/verification/bin/libbonsai_ref.so"
strip --strip-all "$BENCH/verification/bin/libbonsai_ref.so"
ls -lh "$BENCH/verification/bin/libbonsai_ref.so"
