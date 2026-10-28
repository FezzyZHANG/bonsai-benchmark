# Model: Ternary-Bonsai-2-4B (`qwen35`)

The model is `data/Ternary-Bonsai-2-4B-PTQ1_0.gguf`. Its GGUF metadata and
tensor inventory are the source of truth for dimensions, tensor names, shapes
and quantization types; parse them with the provided `gguf-py` reader or
directly.

**This document intentionally does not describe the architecture.** The layer
structure, tensor ordering, weight representation and numerical behavior are
part of the task. Recover them from the GGUF, the dequantized development
tensors in `data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors` (development only;
never read at runtime), and the reference runner.

`ref/bin/bonsai-ref` is the numerical oracle; `ref/PROTOCOL.md` documents its
line protocol. Where your interpretation disagrees with the oracle, the oracle
wins.

Contract notes:

- The engine loads the GGUF at runtime and derives every weight from it; it must
  not read the safetensors file, embed weights, or use another engine.
- `bonsai_tensor_dequant` must match the reference dequantization exactly.
  Logits, per-position agreement and greedy tokens are scored against the CPU
  oracle (`ABI.md`, `ACCEPTANCE.md`).
- The ABI and context semantics are in `ABI.md`; the tokenizer is specified in
  `TOKENIZER.md`.
