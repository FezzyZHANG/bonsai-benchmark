import numpy as np, struct, json, itertools
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

def ref_trits(blk):
    w=rows[0][blk*128:(blk+1)*128]
    s=np.max(np.abs(w))
    return np.round(w/s).astype(np.int8)

def decode_qs(qs, groups, digits=5):
    # qs: bytes array length L, groups: list of group byte-widths
    out=[]
    off=0
    for gw in groups:
        # TQ1_0 style: digit-major over group width
        for dgt in range(digits):
            for j in range(gw):
                out.append(((qs[off+j] * (3**dgt)) >> 0) % 3)
        off+=gw
    return np.array(out,dtype=np.int8)-1

# TQ1_0 dequant: qs_byte digits: value = sum(qsbyte * [1,3,9,...]) then ((v*3)>>8)? For TQ1_0: qs = ((qs.astype(uint16)*3)>>8)-1
# That's a different trick: they store ceil(q/243*256)? quantize: qs = (q*256 + 242)//243.
# Let's test both normal base3 and the TQ trick.
def decode_tq(qs, groups, digits=5):
    out=[]
    off=0
    for gw in groups:
        for dgt in range(digits):
            for j in range(gw):
                out.append(((qs[off+j].astype(np.uint16) * (3**dgt)) * 3 >> 8).astype(np.int8))
        off+=gw
    return np.array(out,dtype=np.int8)-1

for blk in range(3):
    ref=ref_trits(blk)
    b=np.frombuffer(raw[blk*28:(blk+1)*28],dtype=np.uint8)
    print("blk",blk,"ref trits", ''.join('+' if x>0 else '-' if x<0 else '0' for x in ref[:40]))
    for split in [("first",0),("last",26)]:
        pass
    for name, qs_slice, qh_slice, d_slice in [
        ("qs0_24_qh_2_d_last", slice(0,24), slice(24,26), slice(26,28)),
        ("d_first_qs_24_qh2", slice(2,26), slice(26,28), slice(0,2)),
        ("d_first_qh2_qs24", slice(4,28), slice(2,4), slice(0,2)),
    ]:
        qs=b[qs_slice]; qh=b[qh_slice]
        d=np.frombuffer(bytes(b[d_slice]),dtype=np.float16)[0]
        for fn in (decode_qs, decode_tq):
            # qs 24 bytes as one group of 24 (5 digits each -> 120)
            cand=fn(qs,[24])
            cand2=fn(qh,[2],digits=4)
            full=np.concatenate([cand,cand2])
            # try compare orders: full vs ref
            m1=(full==ref).mean()
            print(name, fn.__name__, "match",m1, "d",d, "cand", ''.join('+' if x>0 else '-' if x<0 else '0' for x in full[:40]))
