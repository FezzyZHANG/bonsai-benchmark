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
# row0 and row1 = 5120 elems
rows=np.frombuffer(fh.read(2560*4*2),dtype=np.float16).reshape(4,2560).astype(np.float32)
for blk in range(4):
    b=raw[blk*28:(blk+1)*28]
    w=rows[0][blk*128:(blk+1)*128]
    s=np.max(np.abs(w))
    trits=np.round(w/s).astype(np.int8) if s>0 else np.zeros(128,np.int8)
    print("blk",blk,"hex",b.hex())
    print("  d0=",np.frombuffer(b[0:2],dtype=np.float16)[0],"dlast=",np.frombuffer(b[26:28],dtype=np.float16)[0],"s=",s)
    print("  trits[:64]",''.join('+' if x>0 else '-' if x<0 else '0' for x in trits[:64]))
    print("  bytes", ' '.join(f'{x:02x}' for x in b))
    # print middle 24 bits
    bits=''.join(f'{x:08b}' for x in b[2:26])
    print("  bits2..26", bits)
