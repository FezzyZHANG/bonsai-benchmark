# Acceptance criteria

The submission is this directory as you leave it. It is accepted when it builds
offline, implements the ABI, is correct against the CPU llama.cpp oracle, and
passes the VRAM gate. Performance is the optimization target and is reported
separately.

## 1. Build and contract

- `zig build -Doptimize=ReleaseFast` succeeds offline in a clean container and
  produces `zig-out/lib/libbonsai.so`.
- The library exports the ABI in `ABI.md` and loads the GGUF path supplied by
  the harness.
- It does not read the safetensors file or any sidecar model file at runtime,
  does not embed model weights, does not link/execute llama.cpp or another
  inference engine, and does not use the network.
- All model computation uses Vulkan 1.4 with the compute subset of Roadmap 2026
  in `DEVICE.md`. CPU is limited to tokenization/orchestration.
- No prebuilt engine binary; only source and build scripts are submitted.

## 2. Correctness (primary)

Hidden prompts and corpora are different from `data/samples/`. Golden values
are precomputed by the verifier with the CPU llama.cpp backend; grading does
not run CPU inference. The correctness score is a weighted mean:

| suite | weight | metric |
|---|---:|---|
| tokenizer | 0.10 | fraction of hidden strings whose token ids match exactly |
| dequant | 0.15 | fraction of sampled tensors whose f32 values match the reference exactly (PTQ1_0 widened from f16; F32 copied) and whose `tensor_info` type/rank/shape match the GGUF |
| logits | 0.30 | per prompt at n_ctx 512/2048/4096/8192: top-1 match, mean KL, cosine |
| positions | 0.15 | hidden corpus at n_ctx 512/2048: top-1 agreement and mean KL across positions |
| greedy | 0.30 | teacher-forced agreement over 64 steps after 64/2048/8192-token prompts, plus a `generate` smoke test |

Reference points from the CPU-vs-Vulkan llama.cpp comparison on this machine:
logits cosine 0.9998–0.99997, top-1 agreement 97.8–98.8%, mean KLD
0.0007–0.0008 nats. Greedy steps are compared teacher-forced (free-running
sequences diverge after the first flipped token, so sequence equality is not a
useful metric at long context).

Reference (full-credit) operating points:

- tokenizer and dequant: exact;
- top-1 agreement ≥ 95%;
- mean KL ≤ 0.01 nats;
- cosine similarity ≥ 0.995;
- teacher-forced greedy agreement ≥ 95% (and `generate` must pass its smoke
  test).

The verifier converts each component into a [0,1] score and reports the suite
breakdown. **Acceptance requires a correctness score ≥ 0.90.** The CPU-vs-Vulkan
llama.cpp reference scores about 0.99 under this formula. The verifier is
authoritative.

## 3. Performance (optimization target)

Measured only after correctness acceptance:

- pp (prompt processing): 512/2048/4096/8192-token prompts in one eval.
- tg (generation): 128 greedy tokens after context depths 0/512/2048/4096/8192.
- Reference: the stripped llama.cpp **Vulkan** binary on the same GPU, same
  contexts, same greedy policy, same machine.
- Score: geometric mean of `agent tokens/s / reference tokens/s` across the
  pp and tg tests. pp and tg results are reported separately.
- Warmup, repeated runs, and medians are used. Performance is only scored when
  the correctness suites pass; the perf prompts come from the hidden corpus, so
  they cannot be special-cased.

## 4. VRAM (secondary acceptance gate)

The harness polls peak process VRAM from `drm-memory-vram` in
`/proc/<pid>/fdinfo` (deduplicated by DRM client id) across the whole run,
including the largest context (n_ctx 16384), and reads peak host RSS from
`VmHWM`. The reference (llama.cpp Vulkan) is measured with the same workload,
sampling during decode.

- Peak VRAM must satisfy
  `peak ≤ max(1.5 × reference_peak, reference_peak + 512 MiB)`.
- Peak host RSS must satisfy `peak ≤ reference_RSS + 2 GiB`.
- Peak VRAM must be at least ~450 MiB (half the packed model size) to
  demonstrate that the model is resident on the GPU rather than computed on
  the CPU.

The decisive anti-pre-dequant check is the model size: a full F16 copy of the
weights (about 8.4 GiB) or an F32 copy (about 16.8 GiB) cannot fit under the
gate, while the packed weights are about 0.9 GiB. Failing the VRAM gate sets
the performance score to zero; correctness is still reported. The reference
peak/RSS are measured by the verifier on the same GPU.

## 5. Anti-cheat and hygiene

- The engine must derive all weights from the GGUF at the supplied path. A
  submission that embeds weights, reads a generated sidecar, or depends on the
  safetensors file is rejected.
- The reference binaries are not present in the grading container; the engine
  may not spawn them or link llama.cpp/ggml.
- The submission source must be under 32 MiB (excluding `data/`, `docs/`,
  `ref/` and Zig build outputs); embedding model weights, a foreign engine
  binary or generated data is rejected. Linked libraries are checked. No
  network is available.
- The ABI permits worker threads, not child processes. The harness records
  `RUSAGE_CHILDREN` and live children of the grading process; a library that
  spawns a helper process (for example an embedded engine binary) scores zero.
- The engine must be deterministic for a fixed token sequence.
- Do not modify `data/`, `docs/`, or `ref/`.

## 6. Deliverable notes

The session is terminated at the deadline in `BENCH_DEADLINE_EPOCH`; only
workspace files present at that moment are graded. Save code and notes
continuously.

Leave `NOTES.md` with:

- the engine design and which Vulkan features it uses,
- known limitations and unsupported cases,
- measured correctness and pp/tg results,
- peak VRAM at the contexts you tested.
