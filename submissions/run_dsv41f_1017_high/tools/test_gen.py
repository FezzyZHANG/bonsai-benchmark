import ctypes, numpy as np, time
lib=ctypes.CDLL("/workspace/zig-out/lib/libbonsai.so")
lib.bonsai_model_open.restype=ctypes.c_int32
lib.bonsai_model_open.argtypes=[ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_n_vocab.restype=ctypes.c_int32; lib.bonsai_n_vocab.argtypes=[ctypes.c_void_p]
lib.bonsai_context_create.restype=ctypes.c_int32
lib.bonsai_context_create.argtypes=[ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_eval.restype=ctypes.c_int32
lib.bonsai_eval.argtypes=[ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
lib.bonsai_context_reset.restype=ctypes.c_int32; lib.bonsai_context_reset.argtypes=[ctypes.c_void_p]
lib.bonsai_generate_greedy.restype=ctypes.c_int32
lib.bonsai_generate_greedy.argtypes=[ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)]
m=ctypes.c_void_p(); lib.bonsai_model_open(b"/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m))
nv=lib.bonsai_n_vocab(m)
ctx=ctypes.c_void_p(); lib.bonsai_context_create(m, 512, 512, ctypes.byref(ctx))
def gen(tokens, n, ignore=1, chunk=None):
    lib.bonsai_context_reset(ctx)
    if chunk is None:
        arr=(ctypes.c_int32*len(tokens))(*tokens); out=(ctypes.c_float*nv)()
        rc=lib.bonsai_eval(ctx, arr, len(tokens), out, nv)
    else:
        allout=None
        for i in range(0,len(tokens),chunk):
            part=tokens[i:i+chunk]; arr=(ctypes.c_int32*len(part))(*part); out=(ctypes.c_float*nv)()
            rc=lib.bonsai_eval(ctx, arr, len(part), out, nv); allout=out
        out=allout
    o=(ctypes.c_int32*n)()
    t0=time.time(); got=lib.bonsai_generate_greedy(ctx, None, 0, n, ignore, o); dt=time.time()-t0
    return rc, list(o[:got]), dt, np.frombuffer(out,dtype=np.float32).copy()
rc,g1,dt1,logits1=gen([32,3841,13477,37550,33075,888],16)
print("greedy1",g1)
print("oracle1",[279,15217,5388,13,198,32,3841,13477,37550,33075,888,279,15217,5388,13,198])
rc,g2,dt2,logits2=gen([12675,0],16)
print("greedy2",g2)
print("oracle2",[353,2688,264,5286,852,6587,310,3827,1204,310,1236,264,3779,13,353,2688])
rc,_,_,logits3=gen([32,3841,13477,37550,33075,888],1,chunk=2)
print("incremental logits equal:", np.allclose(logits2 if False else logits1, logits3), "maxdiff", np.abs(logits1-logits3).max())
print("timing gen1 %.2fs gen2 %.2fs"%(dt1,dt2))
