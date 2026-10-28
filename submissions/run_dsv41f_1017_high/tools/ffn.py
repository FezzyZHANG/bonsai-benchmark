import numpy as np
g2={}
exec(open("/workspace/tools/perm2.py").read().split('tests={')[0],g2)
globals().update(g2)
o,fo=run(0,0)
B=[fo.reshape(-1)[i*128:(i+1)*128] for i in range(32)]
fp=np.concatenate([B[16*(h%2)+h//2] for h in range(32)])
S2560r=np.array(r.fields["prism.hadamard.sign_values"].contents(),dtype=np.float32)[:2560]
S9216=np.array(r.fields["prism.hadamard.sign_values"].contents(),dtype=np.float32)[6656:]
attn_out=z["blk.0.ssm_out.weight"]@H(S4096*fp)
res=attn_out+h
print("res first3",np.round(res[:3],4),"sum %.6f (0.868838)"%res.sum(),"oracle [1.6005,-0.0018,-0.0202]")
def rms(x,w,eps=1e-6): return x/np.sqrt(np.mean(x*x)+eps)*w
pn=rms(res,z["blk.0.post_attention_norm.weight"])
print("post_norm sum %.6f (-19.285259)"%pn.sum())
sg=S2560r*pn; print("node76 sum %.6f (1.223924)"%sg.sum())
xh2=H(sg); print("node78 sum %.6f (-9.184924)"%xh2.sum())
fg=z["blk.0.ffn_gate.weight"]@xh2; fu=z["blk.0.ffn_up.weight"]@xh2
print("fg sum %.4f (-490.054)"%fg.sum(),"first3",np.round(fg[:3],4),"oracle [0.6459,0.3667,0.0339]")
print("fu sum %.4f (46.233)"%fu.sum(),"first3",np.round(fu[:3],4),"oracle [0.1708,0.2452,-0.0081]")
def silu(x): return x/(1+np.exp(-x))
for tag,f in [("silu(g)*u",silu(fg)*fu),("g*silu(u)",fg*silu(fu))]:
    print(tag,"sum %.5f (1.22137)"%f.sum(),"first3",np.round(f[:3],4))
f=silu(fg)*fu
print("node83 sum %.5f (8.883944)"%(S9216*f).sum())
xh4=H(S9216*f); print("node85 sum %.5f (-6.308198)"%xh4.sum())
fd=z["blk.0.ffn_down.weight"]@xh4
print("ffn_out sum %.6f (-0.546436)"%fd.sum(),"first3",np.round(fd[:3],4),"oracle [0.0274,-0.0605,0.0087]")
lo=fd+res
print("l_out sum %.6f (0.322398)"%lo.sum(),"first3",np.round(lo[:3],4),"oracle [1.628,-0.0623,-0.0116]")
