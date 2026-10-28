import numpy as np
z=np.load("/workspace/tools/layer0.npz")
from gguf import GGUFReader
r=GGUFReader("/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf","r")
signs=np.array(r.fields["prism.hadamard.sign_values"].contents(),dtype=np.float32)
S2560=signs[:2560]
PC=np.array([[bin(i&j).count('1') for j in range(512)] for i in range(512)])
M=np.where(PC%2==0,1.0,-1.0).astype(np.float32)/np.sqrt(512)
def H(x):
    n=x.shape[0]//512
    return (x.reshape(n,512)@M.T).reshape(-1)
def rms(x,w,eps=1e-6): return x/np.sqrt(np.mean(x*x)+eps)*w
e=z["token_embd.weight"][12675]
h=S2560*H(e)
n0=rms(h,z["blk.0.attn_norm.weight"])
xh=H(S2560*n0)
print("xh first3",np.round(xh[:3],4),"last3",np.round(xh[-3:],4))
print("oracle node19 row0 [ -1.2299 -1.2781 -0.1792] last [ 1.5622 -1.6255 -0.1771]")
qkv=z["blk.0.attn_qkv.weight"]@xh
cs=np.array([0.0]) # placeholder
print("qkv first3",np.round(qkv[:3],4),"last3",np.round(qkv[-3:],4),"oracle [-0.4049,-0.7346,0.5102] [-0.5206,-1.1018,-1.4062]")
zz=z["blk.0.attn_gate.weight"]@xh
print("z first3",np.round(zz[:3],4),"head0 last3",np.round(zz[125:128],4),"oracle [0.784,-0.3077,2.5918] [-0.0267,-2.3062,-1.6296]")
alpha=z["blk.0.ssm_alpha.weight"]@n0
print("alpha first3",np.round(alpha[:3],4),"oracle [1.3275,-0.0508,-0.0345]")
conv=z["blk.0.ssm_conv1d.weight"][:,3]*qkv
cs=np.where(conv>0,conv,0)+np.minimum(conv,0)/(1+np.exp(-np.abs(conv)))  # silu via stable
cs=conv/(1+np.exp(-conv))
qk=cs[:4096].reshape(32,128); qk=qk/np.sqrt(np.sum(qk*qk,axis=1,keepdims=True)+1e-6)
print("qk_l2 head0 first3",np.round(qk[0,:3],4),"oracle [0.0278,-0.0443,0.0343]")
print("qk_l2 head16 first3",np.round(qk[16,:3],4),"oracle k head0 [-0.0008,0.0013,-0.0002]")
print("v head0 first3",np.round(cs[4096:4096+3],4),"oracle [0.0046,0.0087,0.012]")
