import numpy as np
g2={}
exec(open("/workspace/tools/elcheck.py").read(),g2)
globals().update(g2)
# rebuild through GDN with variants, compare node61 rows
alpha=z["blk.0.ssm_alpha.weight"]@n0
def sp(x): return np.log1p(np.exp(x))
g=sp(alpha+z["blk.0.ssm_dt.bias"])*z["blk.0.ssm_a"]
bs=1/(1+np.exp(-(z["blk.0.ssm_beta.weight"]@n0)))
conv=z["blk.0.ssm_conv1d.weight"][:,3]*qkv
cs=conv/(1+np.exp(-conv))
qk=cs[:4096].reshape(32,128); qk=qk/np.sqrt(np.sum(qk*qk,axis=1,keepdims=True)+1e-6)
q=qk[:16]; k=qk[16:]; v=cs[4096:].reshape(32,128)
zz=(z["blk.0.attn_gate.weight"]@xh).reshape(32,128)
sw=z["blk.0.ssm_norm.weight"]
def rms(x,w,eps=1e-6): return x/np.sqrt(np.mean(x*x)+eps)*w
def run(mode,orient):
    o=np.zeros((32,128),np.float32)
    for hh in range(32):
        kh=hh%16 if mode==0 else hh//2
        A=np.zeros((128,128),np.float32)*np.float32(np.exp(g[hh]))
        if orient==0:
            d=bs[hh]*(v[hh]-A@k[kh]); A=A+np.outer(d,k[kh]); o[hh]=A@(q[kh]/np.sqrt(128))
        else:
            d=bs[hh]*(v[hh]-k[kh]@A); A=A+np.outer(k[kh],d); o[hh]=(q[kh]/np.sqrt(128))@A
    fo=np.stack([rms(o[hh],sw) for hh in range(32)])*np.stack([1/(1+np.exp(-zz[hh])) for hh in range(32)]) # silu
    fo=np.stack([rms(o[hh],sw) for hh in range(32)])
    fo=zz_silu*fo
    return o,fo
# silu(z)
zz_silu=zz/(1+np.exp(-zz))
for mode in (0,1):
    for orient in (0,1):
        o,fo=run(mode,orient)
        print("mode",mode,"orient",orient,"o head0",np.round(o[0,:3],5),np.round(o[0,-3:],5),"head1",np.round(o[1,:3],5),"head31",np.round(o[31,:3],5))
        print("   fo head0",np.round(fo[0,:3],4),np.round(fo[0,-3:],4),"head1",np.round(fo[1,:3],4),np.round(fo[1,-3:],4),"head31",np.round(fo[31,:3],4))
print("oracle fo head0 [0.0106,-0.0055,0.1381] [0.0001,-0.0002,0.0002]")
print("oracle fo head1 ? from node61 rows: [-0.0005,-0.0119,-0.0152]")
print("oracle o head0 [0.0,0.0,0.0001] [-0.0,0.0001,0.0001] head1 [0.0002,0.0001,0.0001] head31 [0.0,-0.0,0.0]")
