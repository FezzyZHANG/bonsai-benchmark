exec(open("/workspace/tools/sim0.py").read().split("show(\"input_embed\"")[0])
n0=rmsnorm(h,deq("blk.0.attn_norm.weight"))
W=deq("blk.0.attn_qkv.weight")
def Hm(x):
    n=x.shape[0]//512
    pc=np.array([[bin(i&j).count('1') for j in range(512)] for i in range(512)])
    M=np.where(pc%2==0,1.0,-1.0).astype(np.float32)/np.sqrt(512)
    return (x.reshape(n,512)@M.T).reshape(-1)
oracle=np.array([-0.4049,-0.7346,0.5102,-0.5206,-1.1018,-1.4062])
S=S2560
xs={"n0":n0,"S n0":S*n0,"H n0":Hm(n0),"H S n0":Hm(S*n0),"S H n0":S*Hm(n0)}
def foldW(tag):
    # W columns transformed: y = W @ (M x) ; equivalently Wf = W M^T applied on columns
    if tag=="W": return W
    if tag=="W S": return W*S
    if tag=="W H": return W@Hm(np.eye(2560,dtype=np.float32)).T
    if tag=="W H S": return (W@Hm(np.eye(2560,dtype=np.float32)).T)*S
    if tag=="W S H": return (W*S)@Hm(np.eye(2560,dtype=np.float32)).T
for wt in ["W","W S","W H","W H S","W S H"]:
    Wf=foldW(wt)
    for xt,xv in xs.items():
        y=Wf@xv
        v=np.concatenate([y[:3],y[-3:]])
        err=np.max(np.abs(v-oracle)/np.maximum(1e-6,np.abs(oracle)))
        if err<0.02:
            print("MATCH?",wt,"@",xt,"err %.4f"%err,"sum %.4f"%y.sum(),np.round(v,4))
print("done")
