# Tokenizer specification

The GGUF contains the complete tokenizer dictionary and configuration:

| key | value |
|---|---|
| `tokenizer.ggml.model` | `gpt2` |
| `tokenizer.ggml.pre` | `qwen35` |
| `tokenizer.ggml.tokens` | 248,320 strings |
| `tokenizer.ggml.merges` | 247,587 merge rules |
| `tokenizer.ggml.token_type` | 248,320 type codes |
| `tokenizer.ggml.bos_token_id` | 248044 |
| `tokenizer.ggml.eos_token_id` | 248046 |
| `tokenizer.ggml.padding_token_id` | 248044 |
| `tokenizer.ggml.add_bos_token` | false |
| `tokenizer.chat_template` | Jinja template (not used by the ABI) |

This is a **GPT-2 byte-level BPE** tokenizer, not SentencePiece. The dictionary
and merges come from the GGUF; the missing details are the byte mapping and the
`qwen35` pre-tokenization rule.

## 1. Byte-to-unicode mapping

Map each raw byte to a printable Unicode code point (the standard GPT-2 table):

```python
bs = list(range(ord("!"), ord("~") + 1)) + \
     list(range(ord("¡"), ord("¬") + 1)) + \
     list(range(ord("®"), ord("ÿ") + 1))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
byte_to_unicode = {b: chr(c) for b, c in zip(bs, cs)}
```

Tokens in the GGUF are already strings in this representation (for example a
leading space is `Ġ`). Merges are two such strings separated by a space.

## 2. Pre-tokenization (`qwen35`)

Before BPE, split the UTF-8-decoded text with this regex (equivalent to the
case-insensitive original used by the tokenizer):

```
(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
|\p{N}
| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
|\s*[\r\n]+
|\s+(?!\S)
|\s+
```

This is the `qwen35` rule: letters may include combining marks (`\p{M}`),
unlike the older Qwen2 rule. `clean_spaces = false`: do not strip leading
spaces from the pre-tokenized pieces.

## 3. BPE merges

For each pre-token:

1. Convert its raw UTF-8 bytes to the byte-to-unicode string.
2. Split into individual symbols.
3. Repeatedly merge the adjacent pair whose concatenation has the **lowest
   index** in `tokenizer.ggml.merges` (the list order is the merge rank), until
   no pair is present. Ties are resolved by the leftmost occurrence.
4. Map each resulting symbol to its index in `tokenizer.ggml.tokens`; that index
   is the token id.

The merges are the usual `"left right"` strings, e.g. `Ġ Ġ` or `i n`. There are
247,587 entries; token id 0 is the first entry of the tokens array.

## 4. Special tokens and BOS

- Tokens whose type marks them as control/user-defined/special are matched
  **atomically** in the input text before BPE (e.g. `<|im_start|>`); they are
  emitted as their exact token ids.
- `add_bos_token` is false in the GGUF. The reference passes
  `add_special = (add_bos != 0)` to llama.cpp, which honors this metadata, so
  `add_bos` does **not** prepend BOS for this model; `ids_add_bos` in
  `TOKENIZER-VECTORS.json` is therefore identical to `ids`. The harness uses
  `add_bos = 0`.
- EOS is 248046; padding is 248044.
- The ABI does not apply the chat template. The harness tokenizes raw text and
  compares token ids with the reference tokenizer.

## 5. Validation

`ref/bin/bonsai-ref` and `llama-cli` use the same tokenizer; use them to obtain
reference ids for arbitrary strings. `TOKENIZER-VECTORS.json` in this directory
lists reference ids for 15 visible strings (`ids_add_bos` is shown for
completeness and is identical to `ids` for this model). `data/samples/`
contains a few visible sample texts; hidden tests use different strings,
including Unicode, combining marks, code, whitespace and special tokens.

The tokenizer is part of the correctness score (exact token ids). The dequant,
forward-pass and generation suites use token ids supplied by the harness, so a
tokenizer bug does not hide an inference bug.
