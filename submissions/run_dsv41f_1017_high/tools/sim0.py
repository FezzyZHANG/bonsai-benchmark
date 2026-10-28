import numpy as np, struct, json, sys
from gguf import GGUFReader
r = GGUFReader("/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf","r")
raw={t.name: np.frombuffer(bytes(t.data.reshape(-1)),dtype=np.uint8) for t in r.tensors}
info={t.name:(t.tensor_type.name,list(int(x) for x in t.shape)) for t in r.tensors}
signs=np.array(r.fields["prism.hadamard.sign_values"].contents(),dtype=np.float32)
S2560=signs[0:2560]; S4096=signs[2560:6656]; S9216=signs[6656:15872]

def deq(name):
    typ,sh=info[name]
    n=int(np.prod(sh))
    if typ=="F32":
        return np.frombuffer(raw[name].tobytes(),dtype=np.float32).reshape(sh[::-1]) if len(sh)==2 else np.frombuffer(raw[name].tobytes(),dtype=np.float32)
    inner=sh[0]; nblk=inner//128
    out=np.zeros((sh[1],sh[0]),np.float32)
    data=raw[name].tobytes()
    rows=sh[1] if len(sh)==2 else 1
    for row in range(rows):
        base=row*nblk*28
        vec=np.zeros(inner,np.float32)
        for b in range(nblk):
            blk=data[base+b*28:base+(b+1)*28]
            d=np.frombuffer(blk[26:28],dtype=np.float16)[0]
            vals=[]
            for chunk,gw,nd in [(blk[0:16],16,5),(blk[16:24],8,5),(blk[24:26],2,4)]:
                for dg in range(nd):
                    for j in range(gw):
                        v=(chunk[j]*(3**dg))&0xFF
                        vals.append(((v*3)>>8)-1)
            vec[b*128:(b+1)*128]=np.array(vals,np.float32)*float(d)
        out[row]=vec
    return out

def H(x, block=512):
    N=block; n=x.shape[0]//N
    pc=np.array([[bin(i&j).count('1') for j in range(N)] for i in range(N)])
    M=np.where(pc%2==0,1.0,-1.0).astype(np.float32)/np.sqrt(N)
    return (x.reshape(n,N)@M.T).reshape(-1)

def rmsnorm(x,w,eps=1e-6):
    ms=np.mean(x*x)+eps
    return x/np.sqrt(ms)*w

def silu(x): return x/(1+np.exp(-x))
def sigmoid(x): return 1/(1+np.exp(-x))
def softplus(x): return np.log1p(np.exp(x))

def matvec(W,x): return W@x   # W [out,in]

def show(name, arr, oracle_sum, oname):
    print(f"{name:22s} sum={arr.sum():.6f} oracle {oname} sum={oracle_sum:.6f} first={arr.reshape(-1)[:6]}")

emb=deq("token_embd.weight")
e=emb[12675]
h=S2560*H(e)
show("input_embed",h,-0.398644,"input_embed")
n0=rmsnorm(h,deq("blk.0.attn_norm.weight"))
show("attn_norm",n0,-44.322887,"attn_norm")
Wqkv=deq("blk.0.attn_qkv.weight")
xh=H(S2560*n0)
show("xh(node19)",xh,91.889381,"node_19")
qkv=matvec(Wqkv,xh)
show("qkv",qkv,57.664261,"node_21")
# also try safetensors copy
import struct as st
fh=open("/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors","rb")
nh=st.unpack("<Q",fh.read(8))[0]; hdr=json.loads(fh.read(nh)); base=8+nh
def stget(name,rows):
    e=hdr[name]; fh.seek(base+e["data_offsets"][0]); cols=e["shape"][1]
    n=min(rows,e["shape"][0])
    return np.frombuffer(fh.read(cols*n*2),dtype=np.float16).reshape(n,cols).astype(np.float32)
Wqkv_st=stget("blk.0.attn_qkv.weight",8192)
print("maxdiff Wqkv", np.abs(Wqkv_st-Wqkv).max())
q2=Wqkv_st@xh
print("qkv from st sum",q2.sum())
print("qkv oracle first 6: [-0.4049,-0.7346,0.5102,-0.5206,-1.1018,-1.4062] sum 57.664261")
# alpha beta
alpha=matvec(deq("blk.0.ssm_alpha.weight"),n0)
show("alpha",alpha,-17.180998,"alpha")
sp=softplus(alpha+deq("blk.0.ssm_dt.bias"))
show("softplus",sp,94.492844,"a_softplus")
g=sp*deq("blk.0.ssm_a")
show("gate",g,-14.338453,"gate")
beta=matvec(deq("blk.0.ssm_beta.weight"),n0)
show("beta",beta,12.6335,"beta")
bs=sigmoid(beta)
show("beta_sig",bs,18.378262,"beta_sigmoid")
# conv
convw=deq("blk.0.ssm_conv1d.weight")  # GGUF [4,8192] but our deq returns shape [4,8192]? deq expects 2D and inner=4 -> block mismatch; F32 so fine
print("convw shape",convw.shape)
conv=convw[3]*qkv  # only current timestep nonzero prev states
show("conv_raw",conv,0,"?")
cs=silu(conv)
show("conv_silu",cs,36.978474,"conv_output_silu")
q=cs[0:2048].reshape(16,128); k=cs[2048:4096].reshape(16,128); v=cs[4096:8192].reshape(32,128)
# l2 norm per head over q+k
qk=cs[0:4096].reshape(32,128)
qk_l2=qk/np.sqrt(np.sum(qk*qk,axis=1,keepdims=True)+1e-6)
show("qk_l2",qk_l2.reshape(-1),15.260769,"qk_l2")
print("q_l2 sum",qk_l2[:16].sum(),"oracle",4.028579,"k sum",qk_l2[16:].sum(),"oracle",11.232174)
q=qk_l2[:16]; k=qk_l2[16:]
z=matvec(deq("blk.0.attn_gate.weight"),xh)
show("z",z,-634.179871,"z")
# GDN variants
state=np.zeros((32,128,128),np.float32)  # [vhead, dv?, dk?]
# variant A: S[v] *= exp(g); vnew = v - S^T k (S [dk,dv]?); S += beta*outer(k,vnew); out = S^T q
def gdn(S,k,q,v,g,beta,variant):
    S=S.copy(); out=np.zeros((32,128),np.float32)
    for h in range(32):
        kh=h//2
        Sh=S[h]  # [dv,dk] or [dk,dv]
        dec=np.exp(g[h])
        Sh=Sh*dec
        kk=k[kh]; qq=q[kh]; vv=v[h]; bb=beta[h]
        if variant==0:  # S[dv,dk]: delta = b*(v - S k); S += outer(delta,k); o = S q
            pred=Sh@kk
            delta=bb*(vv-pred)
            Sh=Sh+np.outer(delta,kk)
            o=Sh@qq
        elif variant==1: # S[dk,dv]: delta=b*(v - k^T S)
            pred=kk@Sh
            delta=bb*(vv-pred)
            Sh=Sh+np.outer(kk,delta)
            o=qq@Sh
        S[h]=Sh; out[h]=o
    return S,out
for var in (0,1):
    S1,o=gdn(state,k,q,v,g,bs,var)
    print("variant",var,"state sum",S1.sum(),"out sum",o.sum(), "out first",o.reshape(-1)[:6])
print("oracle: new_state sum 16.188332, attn_output sum 0.480252 first [0,0,0.0001,-0,0.0001,0.0001]")
print("q first",q.reshape(-1)[:6],"k first",k.reshape(-1)[:6],"v first",v.reshape(-1)[:6],"g",g[:6],"beta",bs[:6])
