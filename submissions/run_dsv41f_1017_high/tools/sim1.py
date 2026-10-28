exec(open("/workspace/tools/sim0.py").read().split("show(\"input_embed\"")[0])
# custom H returning candidate transforms
n0=rmsnorm(h,deq("blk.0.attn_norm.weight"))
W=deq("blk.0.attn_qkv.weight")
oracle_first=[-0.4049,-0.7346,0.5102]; oracle_sum=57.664261; oracle_last=[-0.5206,-1.1018,-1.4062]
def report(tag,y):
    print(tag,"sum=%.6f"%y.sum(),"first3",np.round(y[:3],4),"last3",np.round(y[-3:],4),"| oracle",oracle_first,oracle_last,oracle_sum)
xh=H(S2560*n0)
print("xh first3",np.round(xh[:3],4),"last3(row0: idx509-511)",np.round(xh[509:512],4),"oracle",[-1.2299,-1.2781,-0.1792],[1.5622,-1.6255,-0.1771])
report("W@xh",W@xh)
report("(W*S)@xh",(W*S2560)@xh)
report("(W*S)@H(n0)",(W*S2560)@H(n0))
report("W@(S*H n0)",W@(S2560*H(n0)))
# maybe H is applied with sign then hadamard but hadamard matrix transpose ordering reversed bits
N=512
pc=np.array([[bin(i&j).count('1') for j in range(N)] for i in range(N)])
M=np.where(pc%2==0,1.0,-1.0).astype(np.float32)/np.sqrt(N)
def Hm(x,M):
    n=x.shape[0]//512
    return (x.reshape(n,512)@M.T).reshape(-1)
report("W@Hm(M.T? same)",W@Hm(S2560*n0,M))
# br (bit reversed) hadamard
br=np.array([int(format(i,'09b')[::-1],2) for i in range(512)])
Mbr=M[np.ix_(br,br)]
report("W@Hbr(S n0)",W@Hm(S2560*n0,Mbr))
