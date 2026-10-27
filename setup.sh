#!/bin/bash
# Fetch everything the benchmark needs that is not tracked in git.
#
#   ./setup.sh [--skip-images] [--skip-ref] [--skip-data]
#
# Idempotent. Needs network, git, curl, tar, xz, cmake, g++, python3, podman.
set -euo pipefail
BENCH="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$BENCH"
INV="${BONSAI_INVESTIGATION:-$BENCH/../bonsai-investigation}"

SKIP_IMAGES=0; SKIP_REF=0; SKIP_DATA=0
for arg in "$@"; do
  case "$arg" in
    --skip-images) SKIP_IMAGES=1 ;;
    --skip-ref) SKIP_REF=1 ;;
    --skip-data) SKIP_DATA=1 ;;
    *) echo "unknown option $arg" >&2; exit 1 ;;
  esac
done

log() { echo "==> $*"; }
need() { command -v "$1" >/dev/null || { echo "missing required tool: $1" >&2; exit 1; }; }
for t in git curl tar xz cmake g++ python3 sha256sum; do need "$t"; done

DL=deps/downloads
TC=deps/toolchains
DOCS=deps/docs
DATA=deps/data
mkdir -p "$DL" "$TC" "$DOCS" "$DATA"

# --- toolchain archives -------------------------------------------------------
fetch() { # file url
  if [ ! -f "$DL/$1" ]; then
    log "download $1"
    curl -fL --retry 3 -o "$DL/$1.part" "$2"
    mv "$DL/$1.part" "$DL/$1"
  fi
  (cd "$DL" && awk -v f="$1" '$2 == f' SHA256SUMS | sha256sum -c --quiet -) \
    || { echo "checksum mismatch for $1" >&2; exit 1; }
}

ZIG_A=zig-x86_64-linux-0.17.0.tar.xz
VK_A=vulkansdk-linux-x86_64-1.4.363.0.tar.xz
fetch "$ZIG_A" "https://ziglang.org/download/0.17.0/$ZIG_A"
fetch "$VK_A"  "https://sdk.lunarg.com/sdk/download/1.4.363.0/linux/$VK_A"

if [ ! -x "$TC/zig-x86_64-linux-0.17.0/zig" ]; then
  log "extract Zig"
  mkdir -p "$TC/zig-x86_64-linux-0.17.0"
  tar -xJf "$DL/$ZIG_A" -C "$TC/zig-x86_64-linux-0.17.0" --strip-components=1
fi
VK_ROOT="$TC/vulkansdk-1.4.363.0"
VK="$VK_ROOT/1.4.363.0/x86_64"
if [ ! -x "$VK/bin/vulkaninfo" ]; then
  log "extract Vulkan SDK"
  mkdir -p "$VK_ROOT"
  tar -xJf "$DL/$VK_A" -C "$VK_ROOT"
fi

# --- docs --------------------------------------------------------------------
dl_doc() { [ -s "$DOCS/$1" ] || { log "download docs/$1"; curl -fL --retry 3 -o "$DOCS/$1" "$2"; }; }
dl_doc vkspec.html "https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html"
dl_doc SPIRV.html "https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html"
dl_doc GLSLangSpec.4.60.pdf "https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.pdf"
dl_doc roadmap.adoc "https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs/main/appendices/roadmap.adoc"
dl_doc Roadmap-2026.adoc "https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs/main/appendices/roadmap/Roadmap-2026.adoc"
dl_doc VK_KHR_cooperative_matrix.adoc "https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs/main/proposals/VK_KHR_cooperative_matrix.adoc"
dl_doc gguf.md "https://raw.githubusercontent.com/ggml-org/ggml/master/docs/gguf.md"
dl_doc Vulkan-Profiles-OVERVIEW.md "https://raw.githubusercontent.com/KhronosGroup/Vulkan-Profiles/main/OVERVIEW.md"
mkdir -p "$DOCS/device"
cp -f "$VK/share/vulkan/registry/profiles/VP_KHR_roadmap.json" "$DOCS/device/VP_KHR_roadmap.json"
cp -f "$DOCS/Roadmap-2026.adoc" "$DOCS/device/Roadmap-2026.adoc"
cp -f "$TC/zig-x86_64-linux-0.17.0/doc/langref.html" "$DOCS/zig-langref.html" 2>/dev/null || true
if [ ! -s "$DOCS/device/vulkaninfo.json" ]; then
  tmp=$(mktemp -d)
  VK_ABS="$BENCH/$VK"
  (cd "$tmp" && LD_LIBRARY_PATH="$VK_ABS/lib/VulkanLoader/lib" \
      VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
      "$VK_ABS/bin/vulkaninfo" -j >/dev/null 2>&1) || true
  f=$(find "$tmp" -name 'VP_VULKANINFO_*.json' -print -quit)
  [ -n "$f" ] && cp -f "$f" "$DOCS/device/vulkaninfo.json" || \
    echo "warning: could not generate docs/device/vulkaninfo.json" >&2
  rm -rf "$tmp"
fi

# --- Python data environment -------------------------------------------------
if [ ! -x deps/python/bin/python ]; then
  log "create deps/python"
  python3 -m venv deps/python
  deps/python/bin/pip install --no-cache-dir numpy safetensors
fi

# --- data --------------------------------------------------------------------
if [ "$SKIP_DATA" = 0 ]; then
  GGUF=deps/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf
  ST=deps/data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors
  SRC_GGUF="$INV/models/Ternary-Bonsai-2-4B-PTQ1_0/Ternary-Bonsai-2-4B-PTQ1_0.gguf"
  SRC_ST="$INV/exports/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors"
  if [ ! -f "$GGUF" ]; then
    if [ -f "$SRC_GGUF" ]; then
      log "link GGUF from investigation tree"
      ln "$SRC_GGUF" "$GGUF" 2>/dev/null || cp --reflink=auto "$SRC_GGUF" "$GGUF"
    else
      log "download GGUF"
      curl -fL --retry 3 -o "$GGUF.part" \
        "https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf/resolve/main/Ternary-Bonsai-2-4B-PTQ1_0.gguf"
      mv "$GGUF.part" "$GGUF"
    fi
  fi
  if [ ! -f "$ST" ]; then
    if [ -f "$SRC_ST" ]; then
      log "link safetensors from investigation tree"
      ln "$SRC_ST" "$ST" 2>/dev/null || cp --reflink=auto "$SRC_ST" "$ST"
    else
      log "generate safetensors from GGUF (a few minutes)"
      deps/python/bin/python -I verification/tools/gguf_to_safetensors.py \
        "$GGUF" "$ST" --gguf-py deps/gguf-py \
        --manifest deps/data/Ternary-Bonsai-2-4B-PTQ1_0-f16.manifest.json
    fi
  fi
  mkdir -p deps/data/samples
  if [ -f "$INV/data/agreement_corpus.txt" ]; then
    cp -f "$INV/data/agreement_corpus.txt" deps/data/samples/corpus.txt
    cp -f "$INV/data/p2k.txt" deps/data/samples/short.txt 2>/dev/null || true
  fi
  if [ ! -s deps/data/samples/corpus.txt ]; then
    deps/python/bin/python -I - <<'PY'
from pathlib import Path
out = []
for p in ["deps/docs/gguf.md", "deps/docs/Vulkan-Profiles-OVERVIEW.md"]:
    try: out.append(Path(p).read_text(errors="ignore"))
    except OSError: pass
Path("deps/data/samples/corpus.txt").write_text("\n\n".join(out) or "sample corpus\n")
Path("deps/data/samples/short.txt").write_text("The quick brown fox jumps over the lazy dog.\n")
PY
  fi
fi

# --- reference binaries ------------------------------------------------------
if [ "$SKIP_REF" = 0 ]; then
  LLAMA_SRC="${LLAMA_SRC:-$INV/llama.cpp-prism}" verification/tools/build_ref.sh
fi

# --- container images --------------------------------------------------------
if [ "$SKIP_IMAGES" = 0 ]; then
  command -v podman >/dev/null || { echo "podman missing; cannot build images" >&2; exit 1; }
  sandbox/build_images.sh
fi

log "setup complete"
echo "Zig:        $TC/zig-x86_64-linux-0.17.0/zig"
echo "Vulkan SDK: $VK"
echo "Reference:  deps/ref/bin"
echo "Data:       deps/data"
