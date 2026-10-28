exec(open("/workspace/tools/sim0.py").read().split("show(\"input_embed\"")[0])
n0=rmsnorm(h,deq("blk.0.attn_norm.weight"))
xh=H(S2560*n0).reshape(512,5)
for c in range(5):
    print(c, np.round(xh[:3,c],4), np.round(xh[-3:,c],4))
print("oracle chunks:")
print([-1.2299,-1.2781,-0.1792],[1.5622,-1.6255,-0.1771])
print([0.1335,-0.0533,-1.3868],[-1.3056,-1.5434,-1.154],[1.3741,1.3866,0.0465],[1.3673,-0.0916,1.2518])
print([-1.4753,0.086,1.5793],[0.013,-1.3314,1.3999],[1.5074,-1.5697,1.5785],[-0.0953,-0.0364,1.3577])
