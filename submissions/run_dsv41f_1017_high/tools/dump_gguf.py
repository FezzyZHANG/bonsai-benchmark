import sys
from gguf import GGUFReader

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf"
r = GGUFReader(path, "r")

print("=== fields ===")
for name, f in r.fields.items():
    t = f.types
    if name in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type", "tokenizer.ggml.scores", "tokenizer.chat_template"):
        print(name, "count=", len(f.data))
        continue
    try:
        val = f.contents()
    except Exception as e:
        val = "<err %s>" % e
    print(name, "=", repr(val)[:300])

print("=== tensors ===")
for t in r.tensors:
    print(t.name, t.tensor_type.name, list(t.shape), t.n_bytes, t.data_offset)
