import ctypes, numpy as np, subprocess, time
lib=ctypes.CDLL("/workspace/zig-out/lib/libbonsai.so")
lib.bonsai_model_open.restype=ctypes.c_int32
lib.bonsai_model_open.argtypes=[ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_n_vocab.restype=ctypes.c_int32; lib.bonsai_n_vocab.argtypes=[ctypes.c_void_p]
lib.bonsai_tokenize.restype=ctypes.c_int32
lib.bonsai_tokenize.argtypes=[ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
lib.bonsai_context_create.restype=ctypes.c_int32
lib.bonsai_context_create.argtypes=[ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_eval.restype=ctypes.c_int32
lib.bonsai_eval.argtypes=[ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
m=ctypes.c_void_p(); lib.bonsai_model_open(b"/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m))
nv=lib.bonsai_n_vocab(m)
text=open("/workspace/data/samples/corpus.txt","rb").read()*2
n=lib.bonsai_tokenize(m, text, 0, None, 0)
buf=(ctypes.c_int32*n)(); lib.bonsai_tokenize(m, text, 0, buf, n)
ids=list(buf)[:1024]
cmd="load /workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf\nctx 0 2048 vulkan\neval 0 %d %s\nlogits 0 /workspace/tools/ref_1024.bin\nquit\n"%(len(ids)," ".join(map(str,ids)))
open("/workspace/tools/cmd_1024.txt","w").write(cmd)
r=subprocess.run(["/workspace/ref/bin/bonsai-ref"],stdin=open("/workspace/tools/cmd_1024.txt"),capture_output=True,text=True,timeout=1800)
print("ref",r.stdout.strip().splitlines()[:4])
ref=np.fromfile("/workspace/tools/ref_1024.bin",dtype=np.float32)
ctx=ctypes.c_void_p(); lib.bonsai_context_create(m, 2048, 512, ctypes.byref(ctx))
out=(ctypes.c_float*nv)(); arr=(ctypes.c_int32*len(ids))(*ids)
t0=time.time(); rc=lib.bonsai_eval(ctx,arr,len(ids),out,nv); dt=time.time()-t0
got=np.frombuffer(out,dtype=np.float32).copy()
print("eval rc",rc,"t=%.1fs %.1f tok/s"%(dt,len(ids)/dt))
cos=float(got@ref/np.linalg.norm(got)/np.linalg.norm(ref))
def sm(x):
    x=x-x.max(); e=np.exp(x); return e/e.sum()
p=sm(ref.astype(np.float64)); q=sm(got.astype(np.float64))
kl=float(np.sum(p*np.log(np.maximum(p,1e-30)/np.maximum(q,1e-30))))
print("cos %.6f top1 %d/%d KL %.6f"%(cos,np.argmax(got),np.argmax(ref),kl))
