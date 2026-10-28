import numpy as np, struct, json
from gguf import GGUFReader
r = GGUFReader("/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf","r")
t = [x for x in r.tensors if x.name=="blk.0.attn_gate.weight"][0]
raw = bytes(t.data.reshape(-1)[:28*20])
path="/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors"
fh=open(path,"rb")
n=struct.unpack("<Q",fh.read(8))[0]
hdr=json.loads(fh.read(n))
base=8+n
e=hdr["blk.0.attn_gate.weight"]
fh.seek(base+e["data_offsets"][0])
rows=np.frombuffer(fh.read(2560*4*2),dtype=np.float16).reshape(4,2560).astype(np.float32)

def decode_tq(qs, groups, digits=5):
    out=[]
    off=0
    for gw in groups:
        for dgt in range(digits):
            for j in range(gw):
                out.append((int(qs[off+j]) * (3**dgt) * 3) >> 8)
        off+=gw
    return (np.array(out,dtype=np.int32)-1).astype(np.int8)

blk=0
b=np.frombuffer(raw[blk*28:(blk+1)*28],dtype=np.uint8)
cand=decode_tq(b[0:24],[24])
cand=np.concatenate([cand, decode_tq(b[24:26],[2],digits=4)])
w=rows[0][blk*128:(blk+1)*128]
s=np.max(np.abs(w))
ref=np.round(w/s).astype(np.int8)
print("ref vals uniq", np.unique(w))
for i in range(128):
    mark = "OK" if cand[i]==ref[i] else "XX"
    print(i, mark, "ref", ref[i], "cand", cand[i], "w", w[i])
