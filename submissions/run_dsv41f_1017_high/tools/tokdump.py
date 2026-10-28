import os
import json
from gguf import GGUFReader

path = "/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf"
out = "/workspace/tools/tokdata"
os.makedirs(out, exist_ok=True)
r = GGUFReader(path, "r")

def field(name):
    f = r.fields[name]
    return f.contents()

tokens = field("tokenizer.ggml.tokens")
merges = field("tokenizer.ggml.merges")
types = field("tokenizer.ggml.token_type")

def dump_hex_list(name, arr):
    with open(os.path.join(out, name), "w", encoding="ascii") as fh:
        for s in arr:
            if isinstance(s, bytes):
                b = s
            else:
                b = str(s).encode("utf-8")
            fh.write(b.hex())
            fh.write("\n")

dump_hex_list("tokens.hex", tokens)
dump_hex_list("merges.hex", merges)

with open(os.path.join(out, "types.txt"), "w") as fh:
    for t in types:
        fh.write(str(int(t)))
        fh.write("\n")

meta = {}
for k in ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
          "tokenizer.ggml.padding_token_id", "tokenizer.ggml.add_bos_token",
          "tokenizer.ggml.model", "tokenizer.ggml.pre",
          "tokenizer.ggml.add_eos_token"):
    if k in r.fields:
        meta[k] = field(k)
with open(os.path.join(out, "meta.json"), "w") as fh:
    json.dump({k: (v.item() if hasattr(v, "item") else v) for k, v in meta.items()}, fh, indent=1)

# summary of token types and examples of specials
from collections import Counter
cnt = Counter(int(t) for t in types)
print("type counts", dict(cnt))
for i, t in enumerate(types):
    if int(t) != 1:
        s = tokens[i]
        if isinstance(s, bytes):
            s = s.decode("utf-8", "replace")
        print("special", i, int(t), repr(s))
print("n tokens", len(tokens), "n merges", len(merges))
