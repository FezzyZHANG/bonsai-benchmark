import numpy as np
z=np.load("/workspace/tools/layer0.npz")
from gguf import GGUFReader
r=GGUFReader("/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf","r")
signs=np.array(r.fields["prism.hadamard.sign_values"].contents(),dtype=np.float32)
S2560,S4096,S9216=signs[:2560],signs[2560:6656],signs[6656:]
PC=np.array([[bin(i&j).count('1') for j in range(512)] for i in range(512)])
M=np.where(PC%2==0,1.0,-1.0).astype(np.float32)/np.sqrt(512)
def H(x):
    n=x.shape[0]//512
    return (x.reshape(n,512)@M.T).reshape(-1)
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
cs=silu(z["blk.0.ssm_conv1d.weight"][:,3]*qkv)
qk=cs[:4096].reshape(32,128); qk=qk/np.sqrt(np.sum(qk*qk,axis=1,keepdims=True)+1e-6)
q=qk[:16]; k=qk[16:]; v=cs[4096:].reshape(32,128)
S0=np.zeros((32,128,128),np.float32); o=np.zeros((32,128),np.float32)
for hh in range(32):
    kh=hh%16; A=S0[hh]*np.float32(np.exp(g[hh]))
    d=bs[hh]*(v[hh]-A@k[kh]); A=A+np.outer(d,k[kh]); o[hh]=A@(q[kh]/np.sqrt(128))
fo=silu(z["blk.0.attn_gate.weight"]@xh).reshape(32,128)*np.stack([rms(o[hh],z["blk.0.ssm_norm.weight"]) for hh in range(32)])
B=[fo.reshape(-1)[i*128:(i+1)*128] for i in range(32)]
fp=np.concatenate([B[16*(hh%2)+hh//2] for hh in range(32)])
attn_out=z["blk.0.ssm_out.weight"]@H(S4096*fp)
res=attn_out+h
pn=rms(res,z["blk.0.post_attention_norm.weight"])
x2=H(S2560*pn)
fg=z["blk.0.ffn_gate.weight"]@x2; fu=z["blk.0.ffn_up.weight"]@x2
f=silu(fg)*fu
fd=z["blk.0.ffn_down.weight"]@H(S9216*f)
lo=fd+res
print("l_out sum %.6f (0.322398)"%lo.sum())
rn=rms(lo,z["output_norm.weight"])
t=H(S2560*rn)
logits=z["token_embd.weight"]@t
print("logits first3",np.round(logits[:3],4),"oracle [8.6292,3.7464,0.9412]")
print("logits sum %.2f oracle -977708.625"%logits.sum())
print("argmax",int(np.argmax(logits)))
