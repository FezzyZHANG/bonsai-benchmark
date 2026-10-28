exec(open("/workspace/tools/sim0.py").read().split("show(\"input_embed\"")[0])
n0=rmsnorm(h,deq("blk.0.attn_norm.weight"))
xh=H(S2560*n0)
for c in range(5):
    print("chunk",c,np.round(xh[c*512:c*512+3],4),np.round(xh[c*512+509:c*512+512],4))
print("oracle row0 [-1.2299,-1.2781,-0.1792] [1.5622,-1.6255,-0.1771]")
print("row1 [0.1335,-0.0533,-1.3868] [-1.3056,-1.5434,-1.154]")
print("row2 [1.3741,1.3866,0.0465] [1.3673,-0.0916,1.2518]")
print("row3 [-1.4753,0.086,1.5793] [0.013,-1.3314,1.3999]")
print("row4 [1.5074,-1.5697,1.5785] [-0.0953,-0.0364,1.3577]")
W=deq("blk.0.attn_qkv.weight")
oracle=np.array([-0.4049,-0.7346,0.5102,-0.5206,-1.1018,-1.4062])
def q8(x):
    out=np.zeros_like(x); ds=np.zeros(x.shape[0]//32,np.float32)
    for b in range(x.shape[0]//32):
        seg=x[b*32:(b+1)*32]; am=np.max(np.abs(seg))
        d=np.float16(am/127.0).astype(np.float16).item()
        ds[b]=d
        id_=1.0/d if d!=0 else 0.0
        out[b*32:(b+1)*32]=np.clip(np.round(seg*id_),-127,127)
    return out,ds
qx,ds=q8(xh)
# weight: dequantized W, but dot must use original per-128 scale * trits; W is already dequantized, so approximate weight quant? Use exact W but activation q8:
y=(W*(qx.astype(np.float32)))@np.ones(W.shape[1],np.float32)  # nonsense
# proper: dot = sum_i W[o,i]*qx[i]*ds[i//32] but ds applies to x only if W exact:
# reconstruct xhat = qx*ds (q8_0 dequant) then dot with W
xhat=qx*ds.repeat(32)
v=np.concatenate([(W@xhat)[:3],(W@xhat)[-3:]])
print("q8x exactW",np.round(v,4),"relerr",np.max(np.abs(v-oracle)/np.abs(oracle)),"sum",(W@xhat).sum())
