exec(open("/workspace/tools/sim6.py").read().split("# try l2 norm")[0])
def gdn():
    S=np.zeros((32,128,128),np.float32); out=np.zeros((32,128),np.float32)
    for hh in range(32):
        kh=hh%16; kk=k[kh]; qq=q[kh]/np.sqrt(128); vv=v[hh]; bb=bs[hh]
        A=S[hh]*np.float32(np.exp(g[hh]))
        delta=bb*(vv-A@kk); A=A+np.outer(delta,kk)
        S[hh]=A; out[hh]=A@qq
    return S,out
S,o=gdn()
print("state sum %.6f (16.188332) out sum %.6f (0.480252)"%(S.sum(),o.sum()))
sw=deq("blk.0.ssm_norm.weight")
nrm=np.stack([rmsnorm(o[hh],sw) for hh in range(32)])
zr=z.reshape(32,128)
for name,fo in [("silu(z)*nrm",silu(zr)*nrm),("z*silu(nrm)",zr*silu(nrm))]:
    print(name,"sum %.6f (node61 -2.671541)"%fo.sum())
fo=silu(zr)*nrm
# permute: input [128,16,2] -> [128,2,16]
fp=fo.reshape(128,16,2).transpose(0,2,1).reshape(-1)
# hadamard transform + sign4096
t=H(S4096*fp)
outw=deq("blk.0.ssm_out.weight")@t
print("linear_attn_out sum %.6f (1.267481)"%outw.sum())
res=outw+h
print("attn_residual sum %.6f (0.868838)"%res.sum())
pn=rmsnorm(res,deq("blk.0.post_attention_norm.weight"))
x2=H(S2560*pn)
fg=deq("blk.0.ffn_gate.weight")@x2; fu=deq("blk.0.ffn_up.weight")@x2
for name,f in [("silu(gate)*up",silu(fg)*fu),("gate*silu(up)",fg*silu(fu))]:
    print(name,"ffn_swiglu sum %.6f (1.22137)"%f.sum())
f=fg*silu(fu)
t3=H(S9216*f)
fd=deq("blk.0.ffn_down.weight")@t3
print("ffn_out sum %.6f (-0.546436)"%fd.sum())
lo=fd+res
print("l_out sum %.6f (0.322398)"%lo.sum())
