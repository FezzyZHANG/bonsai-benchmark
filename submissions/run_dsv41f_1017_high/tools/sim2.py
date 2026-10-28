exec(open("/workspace/tools/sim0.py").read().split("show(\"input_embed\"")[0])
n0=rmsnorm(h,deq("blk.0.attn_norm.weight"))
W=deq("blk.0.attn_qkv.weight")
xh=H(S2560*n0)
oracle=np.array([-0.4049,-0.7346,0.5102,-0.5206,-1.1018,-1.4062])
def cmp(tag,y):
    v=np.concatenate([y[:3],y[-3:]])
    print(tag,"relmax=%.5f"%np.max(np.abs(v-oracle)/np.maximum(1e-6,np.abs(oracle))),"vals",np.round(v,4),"sum=%.5f"%y.sum())
cmp("f32",W@xh)
cmp("w16",(W.astype(np.float16).astype(np.float32))@xh)
cmp("x16",W@(xh.astype(np.float16).astype(np.float32)))
cmp("both16",(W.astype(np.float16).astype(np.float32))@(xh.astype(np.float16).astype(np.float32)))
# blockwise dot in f16 accumulation
y=np.zeros(W.shape[0],np.float32)
xb=xh.astype(np.float16).astype(np.float32)
for b in range(W.shape[1]//128):
    part=(W[:,b*128:(b+1)*128]*xb[b*128:(b+1)*128]).astype(np.float16).astype(np.float32)
    y+=part.sum(axis=1)
cmp("chunk16acc",y)
# f64 exact
cmp("f64",(W.astype(np.float64)@xh.astype(np.float64)).astype(np.float32))
# maybe xh uses H/512? etc
for name,M in [("H1",None)]:
    pass
# check whether oracle matches W@xh with xh sign applied AFTER H but signs also folded into W? no
