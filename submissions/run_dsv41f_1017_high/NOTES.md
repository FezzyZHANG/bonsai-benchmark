# bonsai-vk — design notes

Zig 0.17.0 Vulkan inference engine for `Ternary-Bonsai-2-4B-PTQ1_0.gguf`
(`qwen35`). Builds offline with `zig build -Doptimize=ReleaseFast` and produces
`zig-out/lib/libbonsai.so` exporting the ABI in `ABI.md`.

## Layout

- `src/main.zig` — C ABI, GGUF model loading, the full forward pass, thread pool.
- `src/gguf.zig` — GGUF v3 reader (metadata arrays, tensor table) and the
  bit-exact PTQ1_0 dequantizer.
- `src/tokenizer.zig` + `src/unicode_tables.zig` — GPT-2 byte-level BPE with the
  `qwen35` pre-tokenizer, special-token handling.
- `src/vk.zig` + `src/vk_bindings.zig` — raw Vulkan 1.4 bootstrap (instance,
  compute-queue device with the Roadmap-2026 compute feature set queried and
  enabled where advertised, device-local buffers, compute pipeline support).
- `build.zig` — links the Vulkan loader (from `$VULKAN_SDK/lib/VulkanLoader/lib`,
  falling back to `/opt/vulkan-sdk`) and libc.

## Recovered architecture (validated against the oracle)

The model is a Qwen3.5/Qwen3-Next-style hybrid, 32 layers, `full_attention_interval=4`
(layers 3,7,…,31 are full attention; the other 24 are Gated DeltaNet linear
attention). All weight tensors are PTQ1_0 ternary except norms, `ssm_a`,
`ssm_dt.bias`, `ssm_conv1d`, `ssm_alpha`, `ssm_beta` (F32).

PTQ1_0 layout (reverse-engineered, verified bit-exactly against
`data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors`): 128 elements per 28-byte
block — 16 bytes base-3 (5 trits/byte), 8 bytes base-3 (5 trits/byte), 2 bytes
base-3 (4 trits/byte), fp16 block scale at the end. Trit decode:
`q = ((b * 3^d mod 256) * 3 >> 8) - 1`, value = `scale * q`, with digit-major
element order inside each sub-chunk.

"Prism" Hadamard folding (`prism.hadamard.*` metadata): weights in
`weight_names` are stored in a sign/rotation-folded domain. Before each such
matmul the activation `x` is transformed `a = H(sign ⊙ x)`, where `H` is the
normalized Sylvester–Walsh Hadamard applied per 512-element block
(`sign_widths=[2560,4096,9216]`, one shared sign vector per width, taken from
`sign_values`). The token-embedding lookup uses the inverse form
`e = sign ⊙ H(row)`, and the tied output projection reuses the forward form on
the final normed hidden state.

Per layer: pre-norm RMSNorm → attention block → residual, post-attention norm →
SwiGLU FFN → residual. Full attention: `q_proj` produces 8192 values stored
**interleaved per head as [q(256), gate(256)]** (this was verified against the
oracle); per-head RMSNorm over 256 with `attn_q_norm`/`attn_k_norm`, partial
NeoX RoPE over the first 64 dims, GQA 16 q heads / 4 kv heads (contiguous
mapping `kv = h/4`), softmax scale `1/16`, causal KV cache, output gated by
`sigmoid(gate)`. Gated DeltaNet: conv1d kernel 4 over the 8192-dim qkv vector
(silu), L2-normalized q/k per 128-dim head, `g = ssm_a * softplus(alpha+dt_bias)`,
`beta = sigmoid(beta_proj)`, recurrent update
`S = S*exp(g); delta = beta*(v - S·k); S += outer(delta,k); o = S·(q/sqrt(128))`
with k-head `h % 16`, then per-v-head RMSNorm × silu(z) and the grouped v-head
permutation `pos = 2*(h%16) + h/16`, then the folded output projection.

Everything above was confirmed node-by-node against `llama-debug --verbose`
intermediate tensor sums/values for layer 0 (SSM) and layer 3 (attention), and
against full logits from `ref/bin/bonsai-ref` (Vulkan).

## Numerical results (measured)

Against `ref/bin/bonsai-ref` (Vulkan oracle) on this machine:

| prompt | cosine | top-1 | mean KL |
|---|---:|---:|---:|
| `[12675]` | 0.999991 | match | 0.00018 |
| `[12675, 0]` | 0.999997 | match | 0.00010 |
| `[32,3841,13477,37550,33075,888]` | 0.999992 | match | 0.00027 |
| 256-token corpus slice (n_ctx 512) | 0.999994 | match | 0.000004 |
| 1024-token corpus slice (n_ctx 2048) | 0.999986 | match | 0.000056 |

Greedy self-check (free-running, 16 tokens): 16/16 oracle match on one prompt,
15/16 on another (single near-tie flip; the oracle itself quantizes matmul
activations to q8_0 which this engine does not emulate — our math is exact f32).
Incremental eval (one 6-token call vs 3×2-token calls) is bit-identical.
Tokenizer: 15/15 `TOKENIZER-VECTORS.json`, >10,000 oracle-compared strings and
every code point compared with the reference with zero mismatches.
Dequant: **all 426 tensors** checked against the f16 safetensors — correct
GGUF-reversed shape, correct ggml type id, and bit-exact values
(`maxdiff 0.0`). Tensor names are NUL-terminated and owned by the model.
Multi-model use (two GGUF instances open, evaluate, close one, keep using the
other) and all documented error paths (capacity, null args, bad token id,
context overflow, unknown tensor) return the documented negative codes with a
thread-local message.

## Memory / VRAM

- Host: the GGUF is read once (~950 MB); SSM state 50 MB, conv state 2.4 MB,
  f32 KV cache (contexts up to 16384 → ~1.07 GB) and scratch.
  Measured `VmHWM` ≈ 1.10 GB with a 16384-position context open.
- Device: every packed PTQ1_0 tensor is uploaded to its own device-local
  Vulkan buffer at model open (201 buffers, still ternary — no pre-dequantized
  copy). Measured `drm-memory-vram` ≈ 888–906 MiB, satisfying the "model
  resident on the GPU" gate. The small F32 tensors stay only on the host.

## Performance and known limitations

- **Matmuls run on the GPU; the surrounding elementwise/recurrent ops run on
  the CPU (the ABI's "orchestration").** Every PTQ1_0 weight tensor is uploaded
  once into its own device-local buffer; a Vulkan compute shader
  (`tools/matvec.comp`, SPIR-V embedded in `src/matvec_spv.zig`) decodes the
  28-byte ternary blocks and computes `W·x` for one output row per invocation.
  Activations and outputs go through persistently mapped host-coherent storage
  buffers; each dispatch is synchronous. Norms, silu/sigmoid, the GDN
  recurrence, conv1d, softmax and the Rope/KV bookkeeping stay on the CPU.
  If Vulkan init/pipeline creation ever fails, the engine transparently falls
  back to a vectorised CPU kernel with a 19-thread spin pool (still correct,
  ~16 tok/s).
  Measured with the GPU path: ≈ 18 tok/s single-stream (256-token prompt in
  14.3 s; 32 tokens in 1.76 s), versus 130–180 tok/s for the reference Vulkan
  backend. The per-dispatch submit/fence-wait overhead (~270 dispatches per
  token) is the main limiter; batching more of the graph onto the GPU (or
  keeping activations device-resident) would be the next step. Long-context
  grading cases (8192-token prompts) are therefore slow (~7–8 min prefill).
- No sampling: greedy argmax only, as required. No chat template.
- Malformed UTF-8 is tokenized byte-wise rather than replaced with U+FFFD.
- Expects the benchmark architecture (32 layers, 2560/9216 dims, PTQ1_0 + F32
  tensors); other GGUF architectures return an error.
- Deterministic for a fixed token sequence; all ABI calls are single-threaded
  and the internal worker pool is joined in `bonsai_model_close`.

## Notes on choices

- Full-precision f32 accumulation is used throughout rather than emulating
  llama.cpp's q8_0 activation quantization; this yields slightly *better*
  agreement with the CPU golden values than the reference Vulkan backend itself
  on the tested prompts.
- `bonsai_eval` evaluates tokens sequentially through a single-token graph
  (KV-cache + recurrent state), so incremental and batched calls produce
  identical state.
