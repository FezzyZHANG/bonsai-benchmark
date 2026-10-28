# Model architecture: Ternary-Bonsai-2-4B (qwen35)

This is the architecture the engine must implement. The GGUF metadata is the
source of truth for dimensions; the tensor names and shapes below are from
`Ternary-Bonsai-2-4B-PTQ1_0.gguf`. The reference binary is the numerical
oracle; where this document and the oracle disagree, the oracle wins.

## Hyperparameters (GGUF keys)

| key | value |
|---|---|
| `general.architecture` | `qwen35` |
| `qwen35.block_count` | 32 |
| `qwen35.context_length` | 262144 |
| `qwen35.embedding_length` | 2560 |
| `qwen35.feed_forward_length` | 9216 |
| `qwen35.attention.head_count` | 16 |
| `qwen35.attention.head_count_kv` | 4 |
| `qwen35.attention.key_length` | 256 |
| `qwen35.attention.value_length` | 256 |
| `qwen35.attention.layer_norm_rms_epsilon` | 1e-6 |
| `qwen35.rope.dimension_count` | 64 |
| `qwen35.rope.dimension_sections` | `[11, 11, 10, 0]` |
| `qwen35.rope.freq_base` | 1e7 |
| `qwen35.full_attention_interval` | 4 |
| `qwen35.ssm.conv_kernel` | 4 |
| `qwen35.ssm.inner_size` | 4096 |
| `qwen35.ssm.state_size` | 128 |
| `qwen35.ssm.time_step_rank` | 32 |
| `qwen35.ssm.group_count` | 16 |

Layer schedule: layer `il` is a **full-attention** layer when
`(il + 1) % 4 == 0`; i.e. layers 3, 7, 11, ..., 31. All other layers are
**linear-attention (gated delta net)** layers: 24 GDN + 8 full attention.

Common primitives:

- `RMSNorm(x, w, eps) = x / sqrt(mean(x^2) + eps) * w`
- `SiLU(x) = x * sigmoid(x)`
- `softplus(x) = log(1 + exp(x))` (the implementation uses the stable form)
- `sigmoid(x) = 1 / (1 + exp(-x))`
- Attention softmax is over the causal context, computed in f32.

## Tensor map

Per full-attention layer (`blk.3`, `blk.7`, ..., `blk.31`):

| tensor | GGUF shape (ne0, ne1) | role |
|---|---|---|
| `attn_q.weight` | 2560 × 8192 | query + gate projection: 16 heads × (256 q + 256 gate) |
| `attn_k.weight` | 2560 × 1024 | K: 4 KV heads × 256 |
| `attn_v.weight` | 2560 × 1024 | V: 4 KV heads × 256 |
| `attn_output.weight` | 4096 × 2560 | output projection |
| `attn_q_norm.weight` | 256 | RMSNorm over each query head |
| `attn_k_norm.weight` | 256 | RMSNorm over each key head |
| `attn_norm.weight` | 2560 | pre-attention RMSNorm |
| `post_attention_norm.weight` | 2560 | post-attention RMSNorm |
| `ffn_gate.weight` | 2560 × 9216 | FFN gate |
| `ffn_up.weight` | 2560 × 9216 | FFN up |
| `ffn_down.weight` | 9216 × 2560 | FFN down |

Per GDN layer (the other 24 layers):

| tensor | GGUF shape | role |
|---|---|---|
| `attn_qkv.weight` | 2560 × 8192 | [Q 2048, K 2048, V 4096] projections |
| `attn_gate.weight` | 2560 × 4096 | z gate for the gated norm |
| `ssm_conv1d.weight` | 4 × 8192 | causal depthwise conv, kernel 4 |
| `ssm_dt.bias` | 32 | alpha time-step bias |
| `ssm_a` | 32 | negative decay (per v head) |
| `ssm_beta.weight` | 2560 × 32 | beta projection |
| `ssm_alpha.weight` | 2560 × 32 | alpha projection |
| `ssm_norm.weight` | 128 | RMSNorm over each v head (head dim) |
| `ssm_out.weight` | 4096 × 2560 | output projection |
| `attn_norm.weight`, `post_attention_norm.weight` | 2560 | RMSNorm |
| `ffn_gate/up/down` | as above | FFN |

Global tensors:

- `token_embd.weight` (2560 × 248320): latent token embedding table.
- `output_norm.weight` (2560).
- No separate `output.weight`; input and output embeddings are tied
  (`prism.hadamard.tied_output = true`).

## Hadamard latent basis (Prism fold)

This model's 2-D weights are stored in a **rotated basis**. The GGUF metadata
`prism.hadamard.*` describes the transform:

- `block_size = 512`
- `transform = normalized-sylvester-walsh-hadamard`
- `axis = input-last-dimension`
- `sign_mode = explicit`; `sign_widths = [2560, 4096, 9216]`; the
  `sign_values` array is sliced in that order (2560, then 4096, then 9216
  values; total 15872).
- `weight_names`: 200 weights (all 2-D attention/SSM/FFN weights).
- `inverse_weight_names = ["token_embd.weight"]`
- `tied_output = true`, `gdn_v_grouped = true`.

The 512×512 matrix is the normalized Sylvester matrix:

```
H[r][c] = (-1)^popcount(r & c) / sqrt(512)
```

It is symmetric and orthogonal (`H H = I`). For a feature vector `x` of width
`w` (w a multiple of 512), the blockwise transform is

```
(H x)[b*512 + r] = sum_c H[r][c] * x[b*512 + c]     for each block b
```

The sign vector `s_w` (width `w`) is applied elementwise; `s_w[i] ∈ {-1,+1}`.
The sign offset for a width is the cumulative offset in
`[2560, 4096, 9216]`.

### Folded weights

For every weight in `weight_names`, the stored matrix is in the latent basis.
The reference transforms the **activation** before the matmul (this is the
order the oracle uses):

```
x' = H_block(s_w * x)        # H applied after sign multiplication
y  = W_latent @ x'           # W_latent is the GGUF/safetensors tensor
```

where `w = W.ne[0]` (the input width) selects `s_w`. Rotating the latent weight
once at load time is mathematically equivalent and is allowed; either form must
match the oracle within the acceptance tolerance.

Special case: `ssm_out.weight` and `gdn_v_grouped`. The GDN output features
(4096 = 128 × 32 v heads) are in tiled order. Before the sign and Hadamard
transform, reshape as `(128, 16, 2)` where the axes are
`[head_dim, n_k_heads, rep]`, permute to `(128, 2, 16)` (i.e. axes
`(0, 2, 1)`), and flatten back to 4096. Then apply `s_4096` and `H_512`.

### Embedding and tied output

`token_embd.weight` is stored as a latent table. For an input token id:

```
z = token_embd_row(id)          # 2560 values, latent
h = s_2560 * H_512(z)           # inverse: Hadamard first, then signs
```

The tied LM head applies the forward fold to the final hidden state and
multiplies by the latent embedding matrix:

```
logits = E_latent @ H_512(s_2560 * h_norm)      # E_latent is [248320, 2560]
```

where `h_norm = RMSNorm(final_hidden, output_norm)`. This is consistent with
`E_latent` rows being the inverse-transformed embeddings.

## Full-attention layer

Input `x` (2560). Let `n = RMSNorm(x, attn_norm)`.

1. **Q + gate projection:** `qg = W_q @ n`, 8192 outputs. The output layout is
   16 groups of 512 values: `[q_head0 (256), gate_head0 (256), q_head1, ...]`.
   `q_head` is split into 16 heads of 256; `gate` is 16 heads of 256.
2. **Q norm:** RMSNorm over each 256-wide query head with `attn_q_norm`.
3. **K/V:** `k = W_k @ n` → 4 heads × 256; `v = W_v @ n` → 4 heads × 256.
   RMSNorm K over each 256-wide head with `attn_k_norm`.
4. **RoPE:** apply rotary embedding to the first `n_rot = 64` dimensions of each
   Q and K head (the remaining 192 dimensions are untouched), with
   `freq_base = 1e7` and `rope.dimension_sections = [11,11,10,0]`. For
   text-only input all MRoPE sections use the same token position; this reduces
   to ordinary RoPE over 64 dimensions.
5. **GQA attention:** 16 Q heads, 4 KV heads (each KV head serves 4 Q heads),
   head dim 256, causal, scale `1/sqrt(256)`. Compute in f32 and softmax over
   the causal positions.
6. **Gate:** `attn = attn * sigmoid(gate)` (gate is the second 256 values of
   each 512-wide QG group).
7. **Output:** `out = W_o @ attn`, where the input width is 4096 and the
   weight is folded (apply `s_4096`, `H_512` to the 4096-wide attention output
   before the matmul).

Then the residual path:

```
x = x + out
x = x + FFN( RMSNorm(x, post_attention_norm) )
```

## FFN

For `u = RMSNorm(x, post_attention_norm)`:

```
g = SiLU(W_gate @ H(s_2560 * u))       # gate and up weights are folded
u2 = W_up @ H(s_2560 * u)
d = W_down @ H(s_9216 * (g * u2))      # down is folded with the 9216 sign vector
x = x + d
```

The gate/up/down weights are all in `weight_names`; apply the fold to their
inputs. `W_gate`/`W_up` input width is 2560, `W_down` input width is 9216.

## Gated delta net (linear attention) layer

Input `x` (2560). Let `n = RMSNorm(x, attn_norm)`.

Dimensions: `d_state=128`, `n_k_heads=16`, `n_v_heads=32`,
`d_inner=4096`, `head_k_dim=head_v_dim=128`, `conv_kernel=4`.

1. **QKV projection:** `qkv = W_qkv @ n` → 8192 values in the order
   `[Q (16×128 = 2048), K (16×128 = 2048), V (32×128 = 4096)]`, each head-major.
2. **Gate projection:** `z = W_gate @ n` → 4096 values (32 v heads × 128).
3. **Causal conv:** apply a depthwise causal convolution with kernel 4 across
   time on the 8192 qkv channels, then SiLU:
   `qkv_conv = SiLU(conv1d(qkv, W_conv1d))`. Keep the last 3 inputs per channel
   as recurrent conv state.
4. **Split** `qkv_conv` into Q (2048), K (2048), V (4096), each head-major.
5. **Q/K L2 norm:** over each 128-wide q/k head:
   `q = q / sqrt(sum(q^2) + eps)`, same for k. (L2, not RMS; eps=1e-6.)
6. **Beta:** `beta = sigmoid(W_beta @ n)` → one scalar per v head (32).
7. **Alpha / decay:** `a = softplus(W_alpha @ n + ssm_dt.bias) * ssm_a` →
   one scalar per v head (32). `ssm_a` is negative.
8. **Repeat q/k to v heads:** v head `h` uses q/k head `h % 16`; the model
   uses the grouped-v layout (`prism.hadamard.gdn_v_grouped = true`).
9. **Recurrent state:** per v head an `S` matrix of shape 128×128, initially
   zero, carried across tokens. Let `q,k ∈ R^128`, `v ∈ R^128`, `β` scalar,
   `g` scalar (the alpha value above). For each token, in order:

   ```
   S = exp(g) * S
   delta = (v - S^T k) * β          # 128 values
   S = S + k ⊗ delta                # outer product
   o = S^T q * (1/sqrt(128))        # 128 values
   ```

   (The implementation stores `S` transposed as `M[j][i] = S[i][j]`; the
   equations are equivalent.) The state is updated before producing `o`.
10. **Gated norm:** per v head, `out_h = SiLU(z_h) * RMSNorm(o_h, ssm_norm_h)`,
    with `ssm_norm` of width 128 shared over heads.
11. **Output projection:** concatenate the 32 heads (4096 values) in head-major
    order, apply the grouped-v permutation described above, then the
    `ssm_out` fold (`s_4096`, `H_512`), and multiply by `W_ssm_out` (4096→2560).

Then the same residual structure:

```
x = x + ssm_out_result
x = x + FFN( RMSNorm(x, post_attention_norm) )
```

## Layer loop and output

```
h = embedding(token_ids)                 # latent lookup + inverse Hadamard
for il in 0..31:
    h_in = h
    n = RMSNorm(h, blk.il.attn_norm)
    a = full_attention(n) if (il+1)%4 == 0 else gated_delta_net(n)
    h = h_in + a
    h = h + FFN(RMSNorm(h, blk.il.post_attention_norm))
h = RMSNorm(h, output_norm)
logits = latent_token_embd @ H(s_2560 * h)
```

## Numerical notes

- GGUF weights are PTQ1_0 (trits × fp16 group scale) or F32. Dequantized PTQ1_0
  values are exactly representable in F16; the activations and logits should be
  computed in f32 (or f16 where the device supports it, with f32 accumulation)
  to match the CPU golden within the acceptance tolerance.
- The Hadamard transform is blockwise over the **input/last dimension**, block
  size 512; it is applied to activations, not to weights.
- The sign vectors are per input width (2560, 4096, 9216), not per weight.
- The model is text-only for this benchmark; the GGUF chat template is not used
  by the ABI.
- The reference `ref/bin/bonsai-ref` is the authoritative check for all layout
  and ordering details (Q/Gate interleaving, GDN grouped-v permutation,
  state layout, RoPE sections, tokenizer, etc.).
