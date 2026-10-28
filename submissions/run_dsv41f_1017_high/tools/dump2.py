from gguf import GGUFReader
import json, sys, struct

r = GGUFReader("/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf","r")
for name in ["prism.hadamard.weight_names","prism.hadamard.sign_widths","prism.hadamard.sign_values","prism.hadamard.inverse_weight_names","prism.hadamard.block_size","prism.hadamard.gdn_v_grouped"]:
    f = r.fields[name]
    v = f.contents()
    print(name, type(v), len(v) if hasattr(v,'__len__') else '')
    print(v)
    print()
print("fields keys:")
for k in r.fields: print(k)
tensors = r.tensors
from collections import Counter
print(Counter(t.tensor_type.name for t in tensors))
print(len(tensors))
# Also check alignment/data start
print("data_offset", r.data_offset)
# dump header of safetensors
with open("/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors","rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(n))
print("ST entries", len(hdr))
names = [k for k in hdr if k!="__metadata__"]
print(sorted(names)[:20])
for k in sorted(names)[:10]:
    print(k, hdr[k])
