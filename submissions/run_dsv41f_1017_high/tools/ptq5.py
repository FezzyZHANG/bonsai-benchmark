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

def base3(byte, dgt):
    v = (int(byte) * (3**dgt)) & 0xFF
    return ((v * 3) >> 8) - 1

def decode(qs, digit_major=True, group=24, digits=5):
    out=[]
    if digit_major:
        for dgt in range(digits):
            for j in range(group):
                out.append(base3(qs[j], dgt))
    else:
        for j in range(group):
            for dgt in range(digits):
                out.append(base3(qs[j], dgt))
    return np.array(out,dtype=np.int8)

blk=0
b=np.frombuffer(raw[blk*28:(blk+1)*28],dtype=np.uint8)
w=rows[0][blk*128:(blk+1)*128]
s=np.max(np.abs(w))
ref=np.round(w/s).astype(np.int8)
for dm in (True,False):
    cand=decode(b[0:24],dm)
    print("digit_major",dm,"match", (cand==ref[:120]).mean())
    print(''.join('+' if x>0 else '-' if x<0 else '0' for x in cand[:40]))
    print(''.join('+' if x>0 else '-' if x<0 else '0' for x in ref[:40]))
