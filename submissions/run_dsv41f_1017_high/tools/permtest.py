import numpy as np
z=np.load("/workspace/tools/layer0.npz")
def H(x):
    n=x.shape[0]//512
    pc=np.array([[bin(i&j).count('1') for j in range(512)] for i in range(512)])
    M=np.where(pc%2==0,1.0,-1.0).astype(np.float32)/np.sqrt(512)
    return (x.reshape(n,512)@M.T).reshape(-1)
signs=np.load("/workspace/tools/signs.npy") if False else None
# signs from gguf quickly
from gguf import GGUFReader
r=GGUFReader("/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf","r")
signs=np.array(r.fields["prism.hadamard.sign_values"].contents(),dtype=np.float32)
S2560,S4096,S9216=signs[:2560],signs[2560:6656],signs[6656:]
def rms(x,w,eps=1e-6): return x/np.sqrt(np.mean(x*x)+eps)*w
def silu(x): return x/(1+np.exp(-x))
def sig(x): return 1/(1+np.exp(-x))
def sp(x): return np.log1p(np.exp(x))
e=z["token_embd.weight"][12675]
h=S2560*H(e)
n0=rms(h,z["blk.0.attn_norm.weight"])
xh=H(S2560*n0)
qkv=z["blk.0.attn_qkv.weight"]@xh
alpha=z["blk.0.ssm_alpha.weight"]@n0
g=sp(alpha+z["blk.0.ssm_dt.bias"])*z["blk.0.ssm_a"]
bs=sig(z["blk.0.ssm_beta.weight"]@n0)
conv=z["blk.0.ssm_conv1d.weight"][:,3]*qkv
cs=silu(conv)
qk=cs[:4096].reshape(32,128); qk=qk/np.sqrt(np.sum(qk*qk,axis=1,keepdims=True)+1e-6)
q=qk[:16]; k=qk[16:]; v=cs[4096:].reshape(32,128)
S0=np.zeros((32,128,128),np.float32); o=np.zeros((32,128),np.float32)
for hh in range(32):
    kh=hh%16; A=S0[hh]*np.float32(np.exp(g[hh]))
    d=bs[hh]*(v[hh]-A@k[kh]); A=A+np.outer(d,k[kh]); o[hh]=A@(q[kh]/np.sqrt(128))
fo=silu(z["blk.0.attn_gate.weight"]@xh).reshape(32,128)*np.stack([rms(o[hh],z["blk.0.ssm_norm.weight"]) for hh in range(32)])
target67=2.556323; target69=0.998533; target_out=1.267481
Wout=z["blk.0.ssm_out.weight"]
def blocks(f): return [f[i*128:(i+1)*128] for i in range(32)]
B=blocks(fo.reshape(-1))
def test(tag,p):
    fp=np.concatenate([B[p(hh)] for hh in range(32)]).reshape(-1)
    s=(S4096*fp).sum(); t=H(S4096*fp); y=Wout@t
    print(f"{tag:24s} node67={s:.4f} node69={t.sum():.4f} out={y.sum():.4f}")
test("identity",lambda h:h)
test("p4 s+2kh",lambda h:(h//16)+2*(h%16))
test("p2 16s+kh",lambda h:16*(h//16)+(h%16))
test("p2b",lambda h:(h%2)*16+h//2)
test("swap halves",lambda h:(h+16)%32)
test("rev",lambda h:31-h)
test("xor16",lambda h:h^16)
test("bitrev5",lambda h:int(format(h,'05b')[::-1],2))
test("group2",lambda h:(h//2)+16*(h%2))
test("group2b",lambda h:2*(h//16)+(h%16))
print("targets node67=2.556323 node69=0.998533 out=1.267481")
