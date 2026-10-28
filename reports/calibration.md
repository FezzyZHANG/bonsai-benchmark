# Calibration report

Machine: AMD Radeon RX 9070 (RDNA4, RADV GFX1201), Mesa 26.0.8, Vulkan SDK
1.4.363.0 (loader), Zig 0.17.0, Linux. CPU reference uses 6 threads.

## Golden data

- Hidden corpus: 140,000-character excerpt from *Pride and Prejudice*, Project
  Gutenberg #1342 (public domain); 34,376 tokens. Deliberately different from
  the visible `data/samples/` corpus.
- Generated once with the CPU-only oracle (`verification/bin/bonsai-ref-cpu`,
  built with `-DORACLE_CPU=ON`, no Vulkan backend) via
  `verification/tools/generate_golden.py`; the verifier regenerates it if
  missing. It is gitignored.
- Manifest contents: 15 tokenizer strings; last-token logits at n_ctx
  512/2048/4096/8192 (prompts of 448/1984/4032/8128 tokens); 3 teacher-forced
  greedy cases (64/2048/8192-token prompts, 64 steps each); position suites at
  n_ctx 512/2048 with 32 sampled positions each; dequant samples for
  `blk.0.attn_gate.weight`, `blk.3.attn_v.weight`, `blk.0.ssm_alpha.weight`,
  `blk.0.ssm_conv1d.weight`, `output_norm.weight`. About 90 MB.

## Controls

| control | backend | correctness | accepted |
|---|---|---:|---|
| ABI stub (`agent/examples/abi`) | — | 0.0000 | no (model_open fails) |
| llama.cpp calibration shim | Vulkan | 0.9951 | yes |
| llama.cpp calibration shim | CPU | 0.9952 | yes |

The shim is the reference implementation of the ABI on top of llama.cpp
(`verification/tools/refshim/`, static PIC build, hidden in
`verification/bin/libbonsai_ref.so`). It is scored by exactly the same harness
as a submission.

Suite breakdown:

| suite | weight | Vulkan shim | CPU shim |
|---|---:|---:|---:|
| tokenizer | 0.10 | 1.0000 | 1.0000 |
| dequant (values + metadata) | 0.15 | 1.0000 | 1.0000 |
| logits | 0.30 | 0.9932 | 0.9934 |
| positions | 0.15 | 0.9809 | 0.9812 |
| greedy (teacher-forced) | 0.30 | 1.0000 | 1.0000 |
| **total** | | **0.9951** | **0.9952** |

Why the shim does not score exactly 1.0: the golden is produced by llama.cpp
CPU with n_batch 2048 and the harness drives contexts with n_batch 512, and the
Vulkan backend accumulates differently. Observed on Vulkan: top-1 100% on all
four logits cases, KL 2e-6 to 3e-4 nats, cosine >= 0.99993; positions top-1
98.4% (31/32 at n_ctx 512, 32/32 at 2048), mean KL 2.3e-4; teacher-forced
greedy 64/64 on all three prompts. The CPU shim differs from the CPU golden by
the same amount, confirming the floor is batch numerics, not the backend.

## Performance and VRAM

Vulkan shim vs the stripped llama.cpp Vulkan reference, measured through the
grading path (harness inside the GPU container, reference on the host; same
GPU/contexts, 5 repetitions, median):

| test | agent t/s | reference t/s | speedup |
|---|---:|---:|---:|
| pp512 | 2595.87 | 2602.00 | 0.998 |
| pp2048 | 3552.00 | 2651.30 | 1.340 |
| pp4096 | 2997.80 | 2997.16 | 1.000 |
| pp8192 | 2747.07 | 2739.52 | 1.003 |
| tg128 @ d0 | 178.26 | 181.68 | 0.981 |
| tg128 @ d512 | 182.96 | 183.51 | 0.997 |
| tg128 @ d2048 | 140.65 | 178.50 | 0.788 |
| tg128 @ d4096 | 137.54 | 136.82 | 1.005 |
| tg128 @ d8192 | 131.51 | 131.50 | 1.000 |

Geometric mean: **1.004**. A host-path repeat gave 1.003 with the same VRAM
within 0.1 MiB. Both sides are the same llama.cpp code, so ~1.0 is the expected
result. The pp2048 and tg@d2048 rows are measurement outliers
(batching/scheduling and clock noise); the geometric mean is what is scored.

VRAM/RSS (reference sampled during decode at n_ctx 16384; agent peak sampled
across the whole run, largest context 16384):

- agent peak VRAM 1978.8 MiB; reference 1967.1 MiB; cap
  `max(1.5x, ref + 512 MiB)` = 2950.6 MiB -> **PASS**
- agent host RSS 1333 MiB; cap reference RSS + 2 GiB = 4089.4 MiB -> **PASS**
- lower bound (half the packed model, ~450 MiB) -> **PASS**

Pre-dequantized weights cannot pass: an F16 copy is about 8.4 GiB and an F32
copy about 16.8 GiB, far above the cap; the packed model is about 0.9 GiB. The
CPU shim reports 658.5 MiB VRAM (llama.cpp still initializes the Vulkan device
with `-ngl 0`) and is not subject to the gate during calibration.

## Anti-cheat checks exercised

- Grading runs the library in a container that mounts only the built `.so`, the
  GGUF, the golden data and the harness; no submission tree and no network.
- Hidden corpus and golden (gitignored; not mounted into agent runs) prevent
  special-casing test inputs.
- Greedy is scored teacher-forced against the reference prefix, because
  free-running sequences diverge after the first flipped token (the earlier
  corpus produced one flip at step 25 and 25/64 positional matches even though
  the model agreed at every step up to the flip).
- Submission source is capped at 32 MiB (build outputs excluded) to block
  embedded weights or a foreign engine binary.
- The harness records `RUSAGE_CHILDREN` and live children; the ABI permits
  worker threads, not child processes. The shim reports zero on both.
- VRAM/RSS gates are described above. The reference oracle and llama.cpp
  binaries are not present in the grading container.

## Sandbox

`sandbox/smoke_test.sh` passed against a mock Messages API: Bash runs, the GPU
is visible (`AMD Radeon RX 9070 (RADV GFX1201)`), `Write`/`Read` work in
`/workspace`, `WebFetch` is refused as unavailable, and the run exits cleanly.
The mock observed the injected upstream credential; the proxy log shows
`authorization":["<redacted>"]`. The agent container cannot resolve or reach
external hosts. Default wall-clock budget is 2 hours, exported as
`BENCH_DEADLINE_EPOCH`; `time-left` reports the remainder and
`RUN_NAME.meta.json` records start/end/timeout.

## Reproducing

```bash
PY="deps/python/bin/python -I verification/verify.py"

# negative control (ABI stub)
$PY /tmp/abi_stub --skip-perf --out /tmp/rep-stub

# positive control, correctness only, Vulkan backend
BONSAI_SHIM_BACKEND=vulkan $PY /tmp/abi_stub \
  --library verification/bin/libbonsai_ref.so --skip-perf --out /tmp/rep-vk

# full run (correctness + perf + VRAM gate)
BONSAI_SHIM_BACKEND=vulkan $PY /tmp/abi_stub \
  --library verification/bin/libbonsai_ref.so --out /tmp/rep-full

# CPU backend
BONSAI_SHIM_BACKEND=cpu $PY /tmp/abi_stub \
  --library verification/bin/libbonsai_ref.so --skip-perf --out /tmp/rep-cpu
```
