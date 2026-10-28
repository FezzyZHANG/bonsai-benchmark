# C ABI contract

The grader loads `zig-out/lib/libbonsai.so` with `dlopen` and drives it through
the C ABI below. The canonical header is `examples/abi/bonsai.h`; this document
defines the semantics the harness relies on.

```c
typedef struct bonsai_model   bonsai_model;
typedef struct bonsai_context bonsai_context;

uint32_t    bonsai_abi_version(void);       /* returns 1 */
const char *bonsai_last_error(void);        /* thread-local; NULL if none */

int32_t  bonsai_model_open(const char *gguf_path, bonsai_model **out_model);
void     bonsai_model_close(bonsai_model *model);
int32_t  bonsai_n_vocab(const bonsai_model *model);
int32_t  bonsai_n_ctx_train(const bonsai_model *model);

int32_t  bonsai_tokenize(const bonsai_model *model, const char *utf8,
                         int32_t add_bos, int32_t *ids, int32_t capacity);

int32_t  bonsai_tensor_count(const bonsai_model *model);
int32_t  bonsai_tensor_info(const bonsai_model *model, int32_t index,
                            const char **name, int32_t *ggml_type,
                            int32_t *rank, int64_t ne[4]);
int64_t  bonsai_tensor_dequant(const bonsai_model *model, const char *name,
                               float *out, int64_t capacity);

int32_t  bonsai_context_create(bonsai_model *model, int32_t n_ctx,
                               int32_t n_batch, bonsai_context **out_ctx);
void     bonsai_context_close(bonsai_context *ctx);
int32_t  bonsai_context_reset(bonsai_context *ctx);
int32_t  bonsai_eval(bonsai_context *ctx, const int32_t *tokens, int32_t n_tokens,
                     float *logits, int32_t logits_capacity);
int32_t  bonsai_generate_greedy(bonsai_context *ctx,
                                const int32_t *prompt, int32_t n_prompt,
                                int32_t n_predict, uint32_t flags,
                                int32_t *out_ids);
```

## Build artifact

- `zig build -Doptimize=ReleaseFast` at the submission root must produce
  `zig-out/lib/libbonsai.so`.
- Symbols are C linkage (`extern "C"`), no C++ mangling, no extra shared
  libraries except the system Vulkan loader/libc. The grader checks that the
  library does not link llama.cpp/ggml.
- The library must be safe to `dlopen` without environment variables or files
  beside the GGUF path.

## Model

- `bonsai_model_open` parses the GGUF at `gguf_path`. It must not open any other
  model/weight file. The path is a normal filesystem path supplied by the
  harness.
- `bonsai_n_vocab` returns the vocabulary size from the GGUF tokenizer
  metadata.
- `bonsai_n_ctx_train` returns the model context length from the GGUF
  `context_length` metadata.
- Multiple models may be opened, but the tests normally use one.
- Errors: return a negative code and set a human-readable `bonsai_last_error`.

## Tokenizer

- `bonsai_tokenize` implements the model's tokenizer from the GGUF metadata:
  GPT-2 byte-level BPE with the `qwen35` pre-tokenizer described in
  `TOKENIZER.md`.
- `add_bos != 0` requests BOS insertion. This model's GGUF sets
  `tokenizer.ggml.add_bos_token = false`, so the reference (`llama_tokenize`
  with `add_special = add_bos != 0`) does **not** prepend BOS even when
  `add_bos != 0`; the ids are the same as with `add_bos = 0`. The harness
  always calls with `add_bos = 0`.
- Special tokens contained in the GGUF vocabulary are matched atomically as
  their UTF-8 byte sequences. No chat template is applied.
- If `ids == NULL`, return the required number of ids without writing.
- If `ids != NULL` and `capacity < required`, return `BONSAI_ERR_CAPACITY`.
- The token stream must exactly match llama.cpp's tokenizer for the same flags.

## Tensor introspection (staged tests)

- `bonsai_tensor_count` returns the number of tensors in the GGUF.
- `bonsai_tensor_info` returns the tensor name (owned by the model), the numeric
  GGML type id (as reported by `gguf-py`), the rank, and `ne[0..3]` in **GGUF
  order** (`ne[0]` is the contiguous/inner dimension).
- `bonsai_tensor_dequant` writes the tensor to `out` as **f32** in PyTorch /
  safetensors row-major order: the GGUF dimensions are reversed, so `ne[0]`
  varies fastest and element `(i1, i0)` is at `i1*ne[0] + i0`.
- For PTQ1_0 the values must match the reference dequantization bit-exactly
  (no approximation tolerance is applied). For F32 tensors the values are
  copied bit-exactly.
- Returns the element count, or `BONSAI_ERR_CAPACITY` if `capacity` is too
  small.

## Context and evaluation

- `bonsai_context_create` allocates a context with `n_ctx` positions. `n_batch`
  is a processing hint; the engine may choose its own internal batching.
- `bonsai_context_reset` discards all KV/recurrent state and positions.
- `bonsai_eval` appends `n_tokens` token ids (n_tokens >= 1) and evaluates
  them. The token ids are already tokenized; the engine must not re-tokenize
  them.
- If `logits != NULL`, it must point to at least `n_vocab` floats; the engine
  writes the f32 logits for the **last token**. If `capacity < n_vocab`,
  return `BONSAI_ERR_CAPACITY`.
- The engine must support incremental calls: `bonsai_eval` on a prompt followed
  by repeated `bonsai_eval` with one token, producing the same state as one
  large call.
- Position indices advance by the number of tokens appended; reset starts at 0.

## Greedy generation

- `bonsai_generate_greedy` evaluates `prompt[0..n_prompt)` (may be empty), then
  selects the argmax token of the last-token logits, appends it, and repeats up
  to `n_predict` times.
- It writes generated token ids to `out_ids` (capacity at least `n_predict`) and
  returns the count. It stops early at the model's EOS token (from the GGUF
  tokenizer metadata) unless
  `BONSAI_GEN_IGNORE_EOS` is set.
- Determinism is required: same prompt and context size produce the same ids.
- After the call the context state is unspecified; the harness resets it.

## Errors and robustness

- `bonsai_last_error` returns a thread-local NUL-terminated string, valid until
  the next API call on that thread, or NULL.
- The library must not call `exit`, abort, or install signal handlers. Invalid
  arguments and allocation failures return negative codes.
- All API calls are made from one thread at a time. The engine may spawn worker
  threads internally; it must join/free them in `close`.
