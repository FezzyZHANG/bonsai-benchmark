import numpy as np, struct, json
path="/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors"
fh=open(path,"rb")
n=struct.unpack("<Q",fh.read(8))[0]
hdr=json.loads(fh.read(n))
base=8+n
def get(name, rows=8):
    e=hdr[name]; fh.seek(base+e["data_offsets"][0])
    sh=e["shape"]
    if len(sh)==1:
        d=np.frombuffer(fh.read(sh[0]*2 if e["dtype"]=="F16" else sh[0]*4),dtype=np.float16 if e["dtype"]=="F16" else np.float32)
        return d.astype(np.float32)
    cols=sh[1]
    d=np.frombuffer(fh.read(cols*rows*2 if e["dtype"]=="F16" else cols*rows*4),dtype=np.float16 if e["dtype"]=="F16" else np.float32)
    return d.reshape(rows,cols).astype(np.float32)
for name in ["token_embd.weight","blk.0.attn_qkv.weight","blk.0.ffn_gate.weight","blk.0.ssm_out.weight","blk.3.attn_q.weight","blk.3.attn_v.weight","blk.3.attn_output.weight","blk.0.ffn_down.weight","output_norm.weight","blk.0.ssm_a","blk.0.ssm_conv1d.weight","blk.0.ssm_norm.weight","blk.0.ssm_alpha.weight"]:
    d=get(name)
    print(name, hdr[name]["dtype"], hdr[name]["shape"], "uniq",len(np.unique(d)), "min",d.min(),"max",d.max(), "first", d.reshape(-1)[:6])
