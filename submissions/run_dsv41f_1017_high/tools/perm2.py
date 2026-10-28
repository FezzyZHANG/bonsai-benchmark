import numpy as np
g2={}
exec(open("/workspace/tools/gdnmap.py").read().split("print(\"mode\"")[0],g2)
globals().update(g2)
o,fo=run(0,0)
S4096=np.array(r.fields["prism.hadamard.sign_values"].contents(),dtype=np.float32)[2560:6656]
B=[fo.reshape(-1)[i*128:(i+1)*128] for i in range(32)]
W=z["blk.0.ssm_out.weight"]
import itertools
def H(x):
    n=x.shape[0]//512
    return (x.reshape(n,512)@M.T).reshape(-1)
tests={
 "id":lambda h:h,
 "p2b":lambda h:16*(h%2)+h//2,
 "p4":lambda h:(h//16)+2*(h%16),
 "rev":lambda h:31-h,
 "xor16":lambda h:h^16,
 "bitrev":lambda h:int(format(h,'05b')[::-1],2),
 "rot16":lambda h:(h+16)%32,
}
for name,p in tests.items():
    fp=np.concatenate([B[p(h)] for h in range(32)])
    y=W@H(S4096*fp)
    print(name,"first3",np.round(y[:3],4),"last3",np.round(y[-3:],4),"sum %.4f"%y.sum())
print("oracle linear_attn_out first [1.6259,-0.0121,-0.0094] last [-0.002,-0.0028,-0.0065] sum 1.267481")
