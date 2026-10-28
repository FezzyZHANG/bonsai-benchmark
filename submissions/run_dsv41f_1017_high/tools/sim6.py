exec(open("/workspace/tools/sim0.py").read().split("show(\"input_embed\"")[0])
emb=deq("token_embd.weight"); e=emb[12675]
h=S2560*H(e)
n0=rmsnorm(h,deq("blk.0.attn_norm.weight"))
xh=H(S2560*n0)
qkv=deq("blk.0.attn_qkv.weight")@xh
alpha=matvec(deq("blk.0.ssm_alpha.weight"),n0)
sp=softplus(alpha+deq("blk.0.ssm_dt.bias"))
g=sp*deq("blk.0.ssm_a")
bs=sigmoid(matvec(deq("blk.0.ssm_beta.weight"),n0))
convw=deq("blk.0.ssm_conv1d.weight")  # [8192,4]
conv=convw[:,3]*qkv
cs=silu(conv)
print("conv_silu sum %.6f oracle 36.978474"%cs.sum(), "first",np.round(cs[:3],4))
qk=cs[0:4096].reshape(32,128)
qk_l2=(qk/np.sqrt(np.sum(qk*qk,axis=1,keepdims=True)+1e-6)).reshape(-1)
print("qk_l2 sum %.6f oracle 15.260769"%qk_l2.sum(),"first",np.round(qk_l2[:6],4),"q sum %.6f"%qk_l2[:2048].sum(),"k %.6f"%qk_l2[2048:].sum())
q=qk_l2[0:2048].reshape(16,128); k=qk_l2[2048:4096].reshape(16,128); v=cs[4096:8192].reshape(32,128)
z=deq("blk.0.attn_gate.weight")@xh
print("z sum %.6f oracle -634.179871"%z.sum())
# try l2 norm without eps / with sqrt(dim)?
for eps in (1e-6,0.0):
    qkl=cs[0:4096].reshape(32,128); qkl=qkl/np.sqrt(np.sum(qkl*qkl,axis=1,keepdims=True)+eps)
    print("eps",eps,"sum",qkl.sum())
# maybe l2norm = x/sqrt(sum x^2) * sqrt(dim) ? check first head norm
print("head0 norm",np.linalg.norm(cs[0:128]),"ratio first val",qk_l2[0]/cs[0])
def gdn(S,k,q,v,g,beta,variant):
    S=S.copy(); out=np.zeros((32,128),np.float32)
    for hh in range(32):
        kh=hh//2
        Sh=S[hh]; dec=np.exp(g[hh]); Sh=Sh*dec
        kk=k[kh]; qq=q[kh]; vv=v[hh]; bb=beta[hh]
        if variant==0:
            delta=bb*(vv-Sh@kk); Sh=Sh+np.outer(delta,kk); o=Sh@qq
        elif variant==1:
            delta=bb*(vv-kk@Sh); Sh=Sh+np.outer(kk,delta); o=qq@Sh
        elif variant==2: # no decay-first: delta computed before decay
            delta=bb*(vv-Sh@kk); Sh=Sh*dec; Sh=Sh+np.outer(delta,kk); o=Sh@qq
        elif variant==3:
            delta=bb*(vv-kk@Sh); Sh=Sh*dec+np.outer(kk,delta); o=qq@Sh
        S[hh]=Sh; out[hh]=o
    return S,out
for var in range(4):
    S1,o=gdn(np.zeros((32,128,128),np.float32),k,q,v,g,bs,var)
    print("var",var,"newstate sum %.6f"%S1.sum(),"out sum %.6f"%o.sum(),"out first",np.round(o.reshape(-1)[:4],5))
print("oracle new_state 16.188332 attn_output 0.480252 first [0,0,0.0001,-0]")
print("g",g[:4],"beta",bs[:4],"q norm0",np.linalg.norm(q[0]),"k norm0",np.linalg.norm(k[0]))
