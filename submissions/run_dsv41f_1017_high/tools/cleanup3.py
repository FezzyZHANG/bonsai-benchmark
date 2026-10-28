import os, glob
pats=["/workspace/tools/ref_*.bin","/workspace/tools/*.bin","/workspace/tools/ld*.txt","/workspace/tools/gguf.txt","/workspace/tools/dump2.txt","/workspace/tools/opt.txt","/workspace/tools/sim*.txt","/workspace/tools/k0.txt","/workspace/tools/k1.txt","/workspace/tools/l3.txt","/workspace/tools/layer0_oracle.txt","/workspace/tools/ptq*.txt","/workspace/tools/fsapi.txt","/workspace/tools/permtest.txt","/workspace/tools/longctx*.txt"]
for p in pats:
    for f in glob.glob(p):
        if os.path.getsize(f) > 5000:
            os.remove(f)
print("done")
