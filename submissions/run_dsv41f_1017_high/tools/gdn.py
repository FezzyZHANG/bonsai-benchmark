exec(open("/workspace/tools/sim6.py").read().split("# try l2 norm")[0])
# vectorized state ops: S shape (32,128,128)
def run(kh_mode, orient, qscale, decay_mode, out_orient):
    S=np.zeros((32,128,128),np.float32); out=np.zeros((32,128),np.float32)
    for hh in range(32):
        kh = hh//2 if kh_mode==0 else hh%16
        kk=k[kh]; qq=q[kh]*qscale; vv=v[hh]; bb=bs[hh]
        A=S[hh].copy()
        dec=np.float32(np.exp(g[hh]) if decay_mode==0 else np.exp(g[hh]*0.5))
        if orient==0:   # A is [dv,dk]: pred=A k ; delta=bb(v-pred); A=A+outer(delta,k)
            A=A*dec; delta=bb*(vv-A@kk); A=A+np.outer(delta,kk)
            o = A@qq if out_orient==0 else A.T@qq[::-1]
        else:           # A is [dk,dv]
            A=A*dec; pred=kk@A; delta=bb*(vv-pred); A=A+np.outer(kk,delta)
            o = qq@A if out_orient==0 else A@qq
        S[hh]=A; out[hh]=o
    return S,out
best=[]
for khm in (0,1):
    for orient in (0,1):
        for qs in (1.0, 1/np.sqrt(128)):
            for dm in (0,):
                for oo in (0,1):
                    S,o=run(khm,orient,qs,dm,oo)
                    best.append((abs(S.sum()-16.188332),abs(o.sum()-0.480252),khm,orient,qs,oo,S.sum(),o.sum()))
best.sort()
for b in best[:8]: print("dh=%.4f do=%.4f khm=%d orient=%d qs=%.4f oo=%d state=%.4f out=%.4f"%b)
