import ctypes, numpy as np, sys
lib=ctypes.CDLL("/workspace/zig-out/lib/libbonsai.so")
lib.bonsai_model_open.restype=ctypes.c_int32
lib.bonsai_model_open.argtypes=[ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_last_error.restype=ctypes.c_char_p
lib.bonsai_n_vocab.restype=ctypes.c_int32; lib.bonsai_n_vocab.argtypes=[ctypes.c_void_p]
lib.bonsai_context_create.restype=ctypes.c_int32
lib.bonsai_context_create.argtypes=[ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_eval.restype=ctypes.c_int32
lib.bonsai_eval.argtypes=[ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
m=ctypes.c_void_p()
rc=lib.bonsai_model_open(b"/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m))
print("open",rc,lib.bonsai_last_error())
nv=lib.bonsai_n_vocab(m)
ctx=ctypes.c_void_p()
rc=lib.bonsai_context_create(m, 512, 512, ctypes.byref(ctx))
print("ctx",rc,lib.bonsai_last_error())
tests=[([12675],"ref_1tok.bin"),([12675,0],"ref_2tok.bin"),([32,3841,13477,37550,33075,888],"ref_6tok.bin")]
for toks,fn in tests:
    arr=(ctypes.c_int32*len(toks))(*toks)
    out=(ctypes.c_float*nv)()
    rc=lib.bonsai_eval(ctx, arr, len(toks), out, nv)
    got=np.frombuffer(out,dtype=np.float32).copy()
    ref=np.fromfile("/workspace/tools/"+fn,dtype=np.float32)
    print("eval",toks,"rc",rc,"err",lib.bonsai_last_error())
    cos=float(got@ref/np.linalg.norm(got)/np.linalg.norm(ref))
    top1=int(np.argmax(got)); rtop=int(np.argmax(ref))
    # KL ref||got with softmax
    def sm(x):
        x=x-x.max(); e=np.exp(x); return e/e.sum()
    p=sm(ref.astype(np.float64)); q=sm(got.astype(np.float64))
    kl=float(np.sum(p*np.log(np.maximum(p,1e-30)/np.maximum(q,1e-30))))
    print("  cos %.6f top1 %d/%d KL %.5f got_max %.3f ref_max %.3f"%(cos,top1,rtop,kl,got.max(),ref.max()))
    print("  got top5",np.argsort(-got)[:5],"ref top5",np.argsort(-ref)[:5])
    lib.bonsai_context_reset(ctx)
