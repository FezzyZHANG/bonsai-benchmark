import numpy as np, struct, json
from gguf import GGUFReader
r = GGUFReader("/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf","r")
f = r.fields["prism.hadamard.sign_values"]
signs = np.array(f.contents(), dtype=np.float32)
print("len",len(signs),"s0..5",signs[:5], "2560..2563",signs[2560:2564], "6656..6659", signs[6656:6660])
# token_embd row for token 12675
t=[x for x in r.tensors if x.name=="token_embd.weight"][0]
raw=np.frombuffer(bytes(t.data.reshape(-1)),dtype=np.uint8)
def deq_block(b):
    out=[]
    def b3(byte,d):
        v=(int(byte)*(3**d))&0xFF
        return ((v*3)>>8)-1
    for chunk,gw,nd in [(b[0:16],16,5),(b[16:24],8,5),(b[24:26],2,4)]:
        for d in range(nd):
            for j in range(gw):
                out.append(b3(chunk[j],d))
    d=np.frombuffer(bytes(b[26:28]),dtype=np.float16)[0]
    return np.array(out,np.float32)*float(d)
row=12675
nb=2560//128
blk_bytes=raw[row*nb*28:(row+1)*nb*28]
e=np.concatenate([deq_block(blk_bytes[i*28:(i+1)*28]) for i in range(nb)])
print("e first",e[:5],"sum",e.sum())
# H natural order
N=512
idx=np.arange(N)
H=np.empty((N,N),np.float32)
for i in range(N):
    H[i]=np.where(np.bitwise_and(i,idx).astype(np.uint32)!=0, 0,0)
# efficient: parity of popcount(i&j)
pop=np.array([bin(x).count("1") for x in range(N)],dtype=np.int32)
M=((-1.0)**(pop[:,None]&pop[None,:]))/np.sqrt(N)  # wrong: parity of popcount(i&j), compute properly
pc=np.array([[bin(i&j).count('1') for j in range(N)] for i in range(N)])
M=np.where(pc%2==0,1.0,-1.0).astype(np.float32)/np.sqrt(N)
blocks=e.reshape(5,512)
y=(blocks@M.T)  # H x per block
print("H e first", y.reshape(-1)[:5], "sum", y.sum())
s2560=signs[:2560]
zi=y.reshape(-1)*s2560
print("after sign first",zi[:5],"sum",zi.sum())
# oracle: node_2 first 0.0253,-0.0104,0.0108 ; sign result model.input_embed first -0.0253,0.0104,-0.0108, sum -0.398644
# note node_2 had first .0253 and model.input_embed first -.0253
print("oracle node2 first 0.0253 -0.0104 0.0108 sum 0.052826")
print("oracle input_embed first -0.0253 0.0104 -0.0108 sum -0.398644")
print("signs first 8", s2560[:8])
print("signs4096 first4", signs[2560:2564], "signs9216 first4", signs[6656:6660])
