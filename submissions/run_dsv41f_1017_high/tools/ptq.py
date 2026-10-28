import numpy as np, struct, json
from gguf import GGUFReader
r = GGUFReader("/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf","r")
t = [x for x in r.tensors if x.name=="blk.0.attn_gate.weight"][0]
raw = bytes(t.data.reshape(-1)[:112])
print("first 112 bytes:", raw.hex())
for off in range(0,112,28):
    b = raw[off:off+28]
    s = np.frombuffer(b[0:2], dtype=np.float16)[0]
    print("block", off//28, "h0", s, b.hex())

path="/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors"
fh=open(path,"rb")
n=struct.unpack("<Q",fh.read(8))[0]
hdr=json.loads(fh.read(n))
base=8+n
e=hdr["blk.0.attn_gate.weight"]
print(e)
fh.seek(base+e["data_offsets"][0])
data=np.frombuffer(fh.read(2560*2*2),dtype=np.float16).reshape(2,2560).astype(np.float32)
w0=data[0]
print("w0[:16]", w0[:16])
print("w1[:16]", data[1][:16])
print("uniq row0", np.unique(np.round(w0,6))[:20], len(np.unique(np.round(w0,6))))
