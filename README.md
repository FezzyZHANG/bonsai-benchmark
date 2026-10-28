# bonsai-vk benchmark (operator documentation)

Benchmark task for AI coding agents running in Claude Code: implement a Vulkan
inference engine for PrismML's `Ternary-Bonsai-2-4B-PTQ1_0.gguf` in **Zig
0.17.0** with **Vulkan SDK 1.4.363.0**, targeting Vulkan 1.4 and the
compute-relevant Vulkan Roadmap 2026 feature set.

The agent gets the GGUF, the dequantized safetensors, the fork's `gguf-py`
reader, the Vulkan/SPIR-V/GLSL/GGUF specs, and symbol-stripped llama.cpp
reference binaries. It implements the engine and exposes a C ABI. Correctness
is the acceptance criterion; performance is the optimization target; peak VRAM
is a secondary gate to prevent pre-dequantization.

## Layout

| path | visible to agent | content |
|---|---|---|
| `agent/` | yes (copied into each run) | workspace template |
| `agent/PROMPT.md` | yes | prompt |
| `agent/TASK.md` | yes | task description |
| `agent/ABI.md` | yes | C ABI contract |
| `agent/ACCEPTANCE.md` | yes | acceptance and scoring |
| `agent/DEVICE.md` | yes | GPU and Roadmap 2026 compute subset |
| `agent/MODEL.md` | yes | architecture specification |
| `agent/TOKENIZER.md` | yes | byte-level BPE / qwen35 specification |
| `agent/TOKENIZER-VECTORS.json` | yes | visible reference token ids |
| `agent/examples/` | yes | ABI stub, reference protocol, sample prompts |
| `deps/docs/` | yes (read-only mount) | Vulkan/SPIR-V/GLSL/GGUF/Zig docs |
| `deps/data/` | yes (read-only mount) | GGUF, safetensors, sample text |
| `deps/ref/` | yes (read-only mount) | stripped llama.cpp binaries + protocol |
| `deps/toolchains/` | image only | Zig, Vulkan SDK (in the container image) |
| `deps/python/` | image only | host/verifier Python env |
| `verification/` | **no** | harness, hidden tests, golden data, verifier |
| `verification/tools/oracle/` | **no** | `bonsai-ref` source |
| `verification/tools/refshim/` | **no** | ABI shim around llama.cpp for calibration |
| `sandbox/` | **no** | Containerfiles, podman run scripts, proxy |
| `reports/` | no | investigation/calibration output |
| `scripts/` | no | investigation helpers |

`deps/downloads/`, `deps/toolchains/`, `deps/docs/`, `deps/data/`, `deps/ref/`,
`deps/python/`, `verification/build/`, `verification/tests/golden/`,
`sandbox/generated/` and `runs/` are gitignored.

## Setup

```bash
./setup.sh                 # toolchains, docs, data, ref binaries, container images
./setup.sh --skip-images   # no podman image build
```

`setup.sh` is idempotent. It verifies the Zig and Vulkan SDK archives against
`deps/downloads/SHA256SUMS`, populates the docs, links/copies the GGUF and the
F16 safetensors, builds the static stripped llama.cpp reference binaries and
the `bonsai-ref` oracle, and builds:

- `bonsai-agent:1.0` — Ubuntu 26.04, Mesa 26.0.8/RADV, Zig 0.17.0, Vulkan SDK
  1.4.363.0, Python env, Claude Code 2.1.310
- `bonsai-proxy:1.0` — Deno 2.9.7 with the vendored credential-injecting proxy

## Running an agent

```bash
sandbox/prepare_run.sh run01
ANTHROPIC_BASE_URL=https://... ANTHROPIC_AUTH_TOKEN=... sandbox/run_agent.sh run01
```

Set `BENCH_MODEL=<model-id>` to pin the model through the proxy (it exports
`ANTHROPIC_MODEL` and the `ANTHROPIC_DEFAULT_*_MODEL` variables inside the
container). `RUN_TIMEOUT` bounds the run (default 4 h).

- The run directory is outside this repo (default `/mnt/workspace/bonsai-runs`).
- The agent container has **no internet route**. It is attached only to the
  internal podman network and can reach only the proxy container. The proxy
  holds the credential and forwards only the Messages API paths
  (`/v1/messages`, `/v1/messages/count_tokens`, `/v1/models`) configured by
  `ALLOWED_PATHS`; `MAX_REQUESTS` bounds the budget.
- Claude Code runs inside the container with `sandbox.enabled = false`
  (`sandbox/settings.template.json`); podman is the sandbox. `WebFetch` and
  `WebSearch` are denied at the Claude permission level; the container network
  blocks everything except the proxy.
- GPU: `--device /dev/dri --group-add keep-groups`; the image carries Mesa
  RADV. Use the host's `render` group/ACL as in the investigation.
- The transcript is `RUNS_ROOT/RUN_NAME.transcript.jsonl`; proxy JSONL is
  `RUNS_ROOT/RUN_NAME/proxy.jsonl`.

## Grading

```bash
python3 verification/verify.py /mnt/workspace/bonsai-runs/run01 --out .../run01-report
```

The verifier builds the submission offline in a clean container, then runs it
in a GPU container with hidden tests. It reports:

1. **Correctness (primary)** — tokenizer ids, dequantized tensors, last-token
   logits, per-position agreement, greedy token ids, all against the CPU
   llama.cpp golden. Weighted score; acceptance ≥ 0.90.
2. **Performance** — pp (512/2048/4096/8192) and tg (depths 0/512/2048/4096/8192)
   vs the stripped llama.cpp Vulkan reference, geomean speedup.
3. **VRAM (secondary gate)** — peak process VRAM via `/proc/<pid>/fdinfo`
   `drm-memory-vram` and `VmHWM`, sampled across the whole run (largest
   context 16384); the reference is sampled during decode with the same
   workload. The cap is `max(1.5 × reference, reference + 512 MiB)` for VRAM
   and `reference RSS + 2 GiB` for host RSS. Failing the gate zeroes the
   performance score.

Use `--no-container` for host-side debugging and `--library PATH` to grade a
prebuilt library without rebuilding.

See `agent/ACCEPTANCE.md` for the exact metrics and provisional thresholds.

## Calibration

`verification/tools/refshim/` implements the ABI on top of llama.cpp, built with
PIC static libs into `verification/bin/libbonsai_ref.so` (hidden; never under
`deps/`, which the agent can see). `BONSAI_SHIM_BACKEND=cpu|vulkan` selects the
backend. It is used to calibrate the harness: the shim should score ~1.0
correctness on CPU (`vulkan` scores like the CPU-vs-Vulkan agreement study),
~1.0× performance on Vulkan, and reference VRAM; the ABI stub scores 0.
Thresholds are adjusted after calibration and pilot runs.

## Operator-only solution outline

Do **not** put this in the agent prompt; it is the expected workflow used to
design the tests and calibration:

1. Parse the GGUF with `gguf-py`.
2. Reverse-engineer PTQ1_0 from the GGUF/safetensors pair.
3. Implement the qwen35/GDN architecture and Vulkan kernels in Zig.
4. Make it match CPU llama.cpp (tokenizer, logits, greedy).
5. Optimize pp/tg while staying within the VRAM gate.

## Status

Investigation complete; benchmark scaffold, reference toolchain, container
image, ABI, harness and calibration are being implemented. See `reports/` for
the Roadmap 2026 support notes and the initial performance/agreement study.
