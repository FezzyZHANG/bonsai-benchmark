exec(open("/workspace/tools/sim7.py").read().split("nrm=rmsnorm")[0])
sw=deq("blk.0.ssm_norm.weight")
nrm=np.stack([rmsnorm(o[hh],sw) for hh in range(32)])
fo=silu(zr:=z.reshape(32,128))*nrm
fp=fo.reshape(128,16,2).transpose(0,2,1).reshape(-1)
print("node61 sum %.6f (-2.671541)"%fo.sum())
print("node67 signed sum %.6f (2.556323)"%(S4096*fp).sum(), "first",np.round((S4096*fp)[:3],4))
t=H(S4096*fp)
print("node69 sum %.6f (0.998533)"%t.sum(),"first",np.round(t[:3],4))
Wout=deq("blk.0.ssm_out.weight")
y=Wout@t
print("ssm_out sum %.6f (1.267481)"%y.sum(),"first",np.round(y[:6],4))
# variants: no permute
for tag,v in [("noperm",fo.reshape(-1)),("perm",fp)]:
    tt=H(S4096*v); yy=Wout@tt
    print(tag,"signed sum %.4f Hsum %.4f out sum %.4f"%( (S4096*v).sum(), tt.sum(), yy.sum()))
# maybe permute order (0,2,1) vs (0,2,1) with reshape (128,2,16) -> (128,16,2)?
