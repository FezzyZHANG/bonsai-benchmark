import numpy as np, sys
sys.path.insert(0,"/workspace/tools")
exec(open("/workspace/tools/sim0.py").read().split("emb=deq")[0])
names=["token_embd.weight","output_norm.weight","blk.0.attn_norm.weight","blk.0.post_attention_norm.weight",
"blk.0.attn_qkv.weight","blk.0.attn_gate.weight","blk.0.ssm_out.weight","blk.0.ffn_gate.weight","blk.0.ffn_up.weight","blk.0.ffn_down.weight",
"blk.0.ssm_alpha.weight","blk.0.ssm_beta.weight","blk.0.ssm_conv1d.weight","blk.0.ssm_norm.weight","blk.0.ssm_a","blk.0.ssm_dt.bias"]
d={}
for n in names:
    d[n.replace("/","_")]=deq(n)
np.savez("/workspace/tools/layer0.npz", **d)
print("saved")
