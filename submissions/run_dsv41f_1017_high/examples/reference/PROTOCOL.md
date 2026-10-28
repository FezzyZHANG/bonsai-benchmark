# Reference runner protocol

`ref/bin/bonsai-ref` is a symbol-stripped llama.cpp runner (CPU and Vulkan
backends) for the benchmark model. It reads one command per line on stdin and
replies with exactly one line on stdout.

```
load <model.gguf>
info
tokenize <hex-utf8> [add_bos]
ctx <id> <n_ctx> <cpu|vulkan>
reset <id>
eval <id> <n_tokens> <token_id>...
logits <id> <output.bin>
gen <id> <n_predict> [ignore_eos]
quit
```

| command | reply |
|---|---|
| `load <path>` | `ok` / `error <msg>` |
| `info` | `ok arch=<arch> n_layer=<n> n_embd=<n> n_vocab=<n> n_ctx_train=<n>` |
| `tokenize <hex> [1]` | `ok <n> <id>...` / `error <msg>` |
| `ctx <id> <n_ctx> <cpu\|vulkan>` | `ok` / `error <msg>` |
| `reset <id>` | `ok` / `error <msg>` |
| `eval <id> <n> <id>...` | `ok` / `error <msg>` (appends and evaluates) |
| `logits <id> <path>` | `ok <n_vocab>`; writes raw little-endian f32 logits for the last token |
| `gen <id> <n> [1]` | `ok <n_generated> <id>...` (greedy; stops at EOS unless the last argument is `1`) |
| `quit` | exits |

Notes:

- `<hex-utf8>` is the lowercase hex encoding of the UTF-8 text. This avoids
  quoting issues. Example: `tokenize 48656c6c6f` for `Hello`.
- `tokenize` uses the model's `qwen35` tokenizer; `add_bos` defaults to 0.
- `ctx` creates a context with `n_ctx` positions. `cpu` uses `-ngl 0`;
  `vulkan` uses the Vulkan backend with all layers offloaded.
- `eval` token ids are appended to the context in order; the context must have
  room. Call `reset` between independent prompts.
- `logits` writes `n_vocab` float32 values in model-vocabulary order.
- `gen` selects argmax at each step and appends the token; no sampling
  parameters.
- All state is deterministic for a fixed sequence.
- The CPU backend is memory-bandwidth-bound on this machine: expect tens of
  tokens/s, versus hundreds on the Vulkan backend (and thousands for prefill).
  Use `cpu` for tokenizer, metadata and short spot checks; use `vulkan` for
  iteration. The grader compares against precomputed CPU golden values, so you
  do not need to run CPU inference to be graded.
- `ref/bin/bonsai-ref-cpu` is the same runner built without the Vulkan backend;
  it accepts only `cpu` contexts and is useful for pure-CPU checks.

`llama-cli`, `llama-debug` and `llama-perplexity` are also provided for
ad-hoc reference runs; `llama-cli -ngl 0` uses the CPU backend and
`llama-cli -ngl 99` the Vulkan backend.
