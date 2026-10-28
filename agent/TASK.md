# Task: a Vulkan inference engine for Ternary-Bonsai-2-4B

Implement **`bonsai-vk`**, a Vulkan inference engine for the model in
`data/Ternary-Bonsai-2-4B-PTQ1_0.gguf`, written in **Zig 0.17.0** using
**Vulkan SDK 1.4.363.0**.

The engine must load the GGUF file at runtime, reproduce the model's logits and
greedy token sequence, and be as fast as possible on the provided AMD GPU.
`ref/` contains symbol-stripped llama.cpp binaries with CPU and Vulkan backends
and a reference runner; use them as the correctness oracle. `data/` also
contains the dequantized weights as a safetensors file for development.

## Deliverable

A Zig project at the root of this directory that builds offline with:

```
zig build -Doptimize=ReleaseFast
```

and produces **`zig-out/lib/libbonsai.so`** exporting exactly the C ABI in
`ABI.md` (`extern "C"`, no C++ mangling, no extra required runtime files).

The library must be self-contained:

- It reads only the GGUF path passed by the harness. It must not read the
  safetensors file, a converted sidecar, or any other model file at runtime.
- It must not embed model weights or precomputed results in the submission.
- It must not link to or execute llama.cpp/ggml, or any prebuilt inference
  engine. All model computation must go through Vulkan compute shaders; the
  definition of model computation and the CPU allowlist are under Target and
  constraints.
- No network access is available. The build must not fetch dependencies.
- The grading build runs in a clean container containing only the files you
  submit. `data/`, `docs/` and `ref/` are not present at build time; do not
  read them from `build.zig` or embed them, and do not hard-code `/workspace`
  paths. Anything the build needs must live in the submission.

## Environment

Everything is offline. Available tools and data:

- **Zig 0.17.0** on `PATH` (`zig`). Note that `@cImport` is not available in
  this Zig version; `zig translate-c` is.
- **Vulkan SDK 1.4.363.0**: headers, loader, `glslc`, `glslangValidator`,
  `slangc`, `spirv-*`, `vulkaninfo`, validation layers, `vkprofiles`.
- **Python** with `numpy`, `safetensors`, and the fork's `gguf-py` reader. It is
  importable as `import gguf` from the default `python`; `PYTHONPATH` is preset
  (`python -I` ignores it and will not find the module). `gguf-py` can parse
  this GGUF and expose the raw tensor bytes and metadata; it has no PTQ1_0
  dequantizer and no runtime tokenizer.
- **Data**: `data/Ternary-Bonsai-2-4B-PTQ1_0.gguf` (the model),
  `data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors` (dequantized weights),
  and `data/samples/` (sample prompts/corpus). `TOKENIZER-VECTORS.json` gives
  reference token ids for visible strings.
- **Reference**: `ref/bin/{llama-cli,llama-debug,llama-perplexity,bonsai-ref,bonsai-ref-cpu}`
  are stripped llama.cpp builds. `-ngl 0` / `-ngl 99` or `ctx ... cpu|vulkan`
  select the backend; `bonsai-ref-cpu` has no Vulkan backend. The Vulkan backend
  runs at roughly 130–180 token/s generation and 2,600–3,500 token/s prefill on
  this GPU; the CPU backend is memory-bandwidth-bound at roughly 10–35 token/s,
  so reserve it for tokenizer checks and short spot checks. `ref/PROTOCOL.md`
  describes the line protocol.
- **Docs**: `docs/` contains the Vulkan 1.4 spec, Roadmap 2026 appendix and
  profile, SPIR-V/GLSL specs, GGUF spec, Vulkan Profiles overview, and Zig docs.

## Bash command classification

Claude Code may deny a safe Bash command when it is compound. Prefer one simple
command per call, do not set environment variables to paths outside the
workspace, and if a command is still denied, put it in a workspace script and
run the script.

## Time budget

The session is terminated at a fixed wall-clock deadline. `BENCH_DEADLINE_EPOCH`
holds the deadline as Unix time, and `time-left` prints the remaining time. The
default budget is 2 hours; trust the environment variable if it differs.

Only files in this workspace at the deadline are graded. The conversation is
not saved for grading, so write code, run tests and update `NOTES.md`
continuously rather than leaving everything to the end. Plan the budget: the
Vulkan reference is fast enough for frequent checks, the CPU reference is slow,
and hidden grading uses precomputed CPU values, so CPU inference is never
required.

## Target and constraints

- Target **Vulkan 1.4** and the compute-relevant Vulkan Roadmap 2026 feature
  subset in `DEVICE.md`. The GPU does not support three non-compute Roadmap 2026
  items (presentation/copy/host-image); those are not required.
- The engine must create a Vulkan device from the provided GPU and run the
  model's forward pass on it. **Model computation** means every arithmetic
  operation on model weights or activations in the forward pass, including
  normalization, activations (silu/sigmoid), softmax, RoPE, the GDN
  recurrence, conv1d, KV-cache updates and projections. All of it must be
  issued as Vulkan compute work. CPU code is limited to tokenization, Vulkan
  command/memory orchestration and I/O; CPU tensor arithmetic is a violation,
  including a CPU fallback used when Vulkan initialization fails (fail
  instead). A submission that performs model computation on the CPU is
  non-compliant even if it passes correctness and the VRAM residency gate.
- Support at least `n_ctx = 16384`; the harness creates contexts up to 16384
  (the long greedy case and the VRAM measurement). The model's training context
  is 262144.
- The harness calls the ABI single-threaded; the engine may use its own worker
  threads. Calls must not crash on malformed input; return a negative error and
  set `bonsai_last_error()`.
- Model execution must be deterministic for a fixed token sequence (greedy,
  no sampling randomness).

## Correctness and scoring

Correctness is the primary acceptance criterion; it is measured against
precomputed CPU llama.cpp golden values, using the same methodology as the
reference CPU-vs-Vulkan comparison (tokenizer IDs, dequantized tensors, logits,
per-position agreement, greedy tokens). Grading does not run CPU inference.
Performance is the optimization target; VRAM is a secondary acceptance gate. The
exact metrics and reference thresholds are in `ACCEPTANCE.md`. Hidden tests use
prompts and corpora not present here.

Do not modify `data/`, `docs/`, or `ref/`.

When done, leave `NOTES.md` with your design, known limitations, and measured
results.
