import ctypes, numpy as np
lib=ctypes.CDLL("/workspace/zig-out/lib/libbonsai.so")
lib.bonsai_model_open.restype=ctypes.c_int32
lib.bonsai_model_open.argtypes=[ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_n_vocab.restype=ctypes.c_int32; lib.bonsai_n_vocab.argtypes=[ctypes.c_void_p]
lib.bonsai_context_create.restype=ctypes.c_int32
lib.bonsai_context_create.argtypes=[ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_eval.restype=ctypes.c_int32
lib.bonsai_eval.argtypes=[ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
lib.bonsai_context_close.argtypes=[ctypes.c_void_p]
lib.bonsai_model_close.argtypes=[ctypes.c_void_p]
lib.bonsai_generate_greedy.restype=ctypes.c_int32
lib.bonsai_generate_greedy.argtypes=[ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)]
ms=[]
for i in range(2):
    m=ctypes.c_void_p(); print("open",i,lib.bonsai_model_open(b"/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf",ctypes.byref(m))); ms.append(m)
for i,m in enumerate(ms):
    nv=lib.bonsai_n_vocab(m); ctx=ctypes.c_void_p(); lib.bonsai_context_create(m,64,64,ctypes.byref(ctx))
    a=(ctypes.c_int32*1)(12675); o=(ctypes.c_float*nv)()
    rc=lib.bonsai_eval(ctx,a,1,o,nv)
    print("model",i,"eval",rc,"top3",list(np.argsort(-np.frombuffer(o,dtype=np.float32))[:3]))
    lib.bonsai_context_close(ctx)
lib.bonsai_model_close(ms[0])
# second model still must work after first closed
m=ms[1]; nv=lib.bonsai_n_vocab(m); ctx=ctypes.c_void_p(); lib.bonsai_context_create(m,64,64,ctypes.byref(ctx))
a=(ctypes.c_int32*1)(12675); o=(ctypes.c_float*nv)()
print("after close of m0, m1 eval",lib.bonsai_eval(ctx,a,1,o,nv),"top3",list(np.argsort(-np.frombuffer(o,dtype=np.float32))[:3]))
lib.bonsai_context_close(ctx); lib.bonsai_model_close(m)
print("done")
