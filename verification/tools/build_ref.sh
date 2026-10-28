#!/bin/bash
# Build the stripped reference binaries and the line-protocol oracle.
#
#   LLAMA_SRC=/path/to/prism/llama.cpp verification/tools/build_ref.sh
#
# Outputs (gitignored): deps/ref/bin/{llama-cli,llama-debug,llama-perplexity,bonsai-ref}
set -euo pipefail
BENCH="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
LLAMA_SRC="${LLAMA_SRC:-$BENCH/../bonsai-investigation/llama.cpp-prism}"
BUILD="$BENCH/verification/build/llama-ref"
OUT="$BENCH/deps/ref/bin"
NPROC="${NPROC:-$(nproc)}"

[ -d "$LLAMA_SRC" ] || { echo "LLAMA_SRC not found: $LLAMA_SRC" >&2; exit 1; }
mkdir -p "$OUT"

# Fast path: reuse an existing static build directory if it is configured.
cmake -S "$LLAMA_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_VULKAN=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_COMMON=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_APP=OFF
cmake --build "$BUILD" --target llama-cli llama-debug llama-perplexity -j "$NPROC"

for b in llama-cli llama-debug llama-perplexity; do
  cp "$BUILD/bin/$b" "$OUT/$b"
  strip --strip-all "$OUT/$b"
done

# Oracle executable, linked against the same static llama/ggml build.
cmake -S "$BENCH/verification/tools/oracle" -B "$BUILD/oracle" \
  -DLLAMA_SRC="$LLAMA_SRC" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD/oracle" -j "$NPROC"
cp "$BUILD/oracle/bonsai-ref" "$OUT/bonsai-ref"
strip --strip-all "$OUT/bonsai-ref"

# Pure-CPU oracle for golden values (no Vulkan backend at all).
cmake -S "$BENCH/verification/tools/oracle" -B "$BUILD/oracle-cpu" \
  -DLLAMA_SRC="$LLAMA_SRC" \
  -DORACLE_CPU=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD/oracle-cpu" --target bonsai-ref -j "$NPROC"
# CPU-only oracle: shipped to agents (same runner, no Vulkan backend) and kept
# hidden for golden generation.
strip --strip-all "$BUILD/oracle-cpu/bonsai-ref"
cp "$BUILD/oracle-cpu/bonsai-ref" "$OUT/bonsai-ref-cpu"
mkdir -p "$BENCH/verification/bin"
cp "$OUT/bonsai-ref-cpu" "$BENCH/verification/bin/bonsai-ref-cpu"

# Calibration ABI shim (hidden; built with PIC static libs). It is a complete
# ABI implementation on top of llama.cpp, so it must never be visible to the
# agent: it lives outside deps/ (which is mounted into the sandbox).
cmake -S "$BENCH/verification/tools/refshim" -B "$BUILD/refshim" \
  -DLLAMA_SRC="$LLAMA_SRC" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD/refshim" --target bonsai -j "$NPROC"
mkdir -p "$BENCH/verification/bin"
cp "$BUILD/refshim/libbonsai.so" "$BENCH/verification/bin/libbonsai_ref.so"
strip --strip-all "$BENCH/verification/bin/libbonsai_ref.so"

cp "$BENCH/agent/examples/reference/PROTOCOL.md" "$BENCH/deps/ref/PROTOCOL.md"
echo "reference binaries installed in $OUT"
ls -lh "$OUT"
