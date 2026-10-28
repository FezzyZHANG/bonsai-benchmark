import numpy as np, struct, json
from gguf import GGUFReader
r = GGUFReader("/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf","r")
t = [x for x in r.tensors if x.name=="blk.0.attn_gate.weight"][0]
raw = bytes(t.data.reshape(-1)[:28*20*4])
path="/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors"
fh=open(path,"rb")
n=struct.unpack("<Q",fh.read(8))[0]
hdr=json.loads(fh.read(n))
base=8+n
e=hdr["blk.0.attn_gate.weight"]
fh.seek(base+e["data_offsets"][0])
rows=np.frombuffer(fh.read(2560*8*2),dtype=np.float16).reshape(8,2560).astype(np.float32)

def base3(byte, dgt):
    v = (int(byte) * (3**dgt)) & 0xFF
    return ((v * 3) >> 8) - 1

def decode_block(b, group=24):
    out=[]
    for dgt in range(5):
        for j in range(group):
            out.append(base3(b[j], dgt))
    for dgt in range(4):
        for j in range(2):
            out.append(base3(b[24+j], dgt))
    return np.array(out,dtype=np.int8)

def ref_trits(w, s):
    return np.round(w/s).astype(np.int8)

# test all blocks of row 0..7
tot=0; ok=0
for row in range(8):
    for blk in range(20):
        b=np.frombuffer(raw[(row*20+blk)*28:(row*20+blk+1)*28],dtype=np.uint8)
        w=rows[row][blk*128:(blk+1)*128]
        s=np.max(np.abs(w))
        ref=ref_trits(w,s)
        cand=decode_block(b)
        m=(cand==ref).mean()
        tot+=1; ok+= (m>0.9)
        print(row,blk,"match %.3f"%m, "uniq", len(np.unique(w)), "s",s,"dlast",np.frombuffer(bytes(b[26:28]),dtype=np.float16)[0])
print("blocks >0.9:", ok, "/", tot)
