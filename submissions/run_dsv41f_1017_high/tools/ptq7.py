import numpy as np, struct, json, itertools
from gguf import GGUFReader
r = GGUFReader("/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf","r")
t = [x for x in r.tensors if x.name=="blk.0.attn_gate.weight"][0]
raw = bytes(t.data.reshape(-1)[:28*20*8])
path="/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors"
fh=open(path,"rb")
n=struct.unpack("<Q",fh.read(8))[0]
hdr=json.loads(fh.read(n))
base=8+n
e=hdr["blk.0.attn_gate.weight"]
fh.seek(base+e["data_offsets"][0])
NROW,NCOL=4096,2560
allw=np.frombuffer(fh.read(2560*256*2),dtype=np.float16).reshape(256,2560).astype(np.float32)

def base3(byte, dgt, reverse=False):
    k = 4-dgt if reverse else dgt
    v = (int(byte) * (3**k)) & 0xFF
    return ((v * 3) >> 8) - 1

def decode_variant(b, kind, dm=True, rev=False):
    if kind=="A":
        chunks=[(b[0:24],24,5),(b[24:26],2,4)]
    elif kind=="B":
        chunks=[(b[24:26],2,4),(b[0:24],24,5)]
    elif kind=="C":
        chunks=[(b[0:16],16,5),(b[16:24],8,5),(b[24:26],2,4)]
    elif kind=="D":
        chunks=[(b[0:2],2,4),(b[2:18],16,5),(b[18:26],8,5)]
    out=[]
    for qs,gw,nd in chunks:
        if dm:
            for dgt in range(nd):
                for j in range(gw):
                    out.append(base3(qs[j],dgt,rev))
        else:
            for j in range(gw):
                for dgt in range(nd):
                    out.append(base3(qs[j],dgt,rev))
    return np.array(out,dtype=np.int8)

def ref_from(w):
    s=np.max(np.abs(w))
    return np.round(w/s).astype(np.int8)

for blk in range(4):
    b=np.frombuffer(raw[blk*28:(blk+1)*28],dtype=np.uint8)
    # row-major block
    roww=allw[0][blk*128:(blk+1)*128]
    # column0: elements i*2560+0 for i in 0..127
    colw=allw[:128,0]
    for label,w in [("row",roww),("col",colw)]:
        T=ref_from(w)
        for kind in "ABCD":
            for dm in (True,False):
                for rev in (False,True):
                    c=decode_variant(b,kind,dm,rev)
                    m=(c==T).mean()
                    if m>0.7:
                        print("MATCH",blk,label,kind,"dm",dm,"rev",rev,m)
print("done")
