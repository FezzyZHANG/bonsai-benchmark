import ctypes, numpy as np
lib=ctypes.CDLL("/workspace/tools/cleanbuild/zig-out/lib/libbonsai.so")
lib.bonsai_model_open.restype=ctypes.c_int32
lib.bonsai_model_open.argtypes=[ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_last_error.restype=ctypes.c_char_p
lib.bonsai_n_vocab.restype=ctypes.c_int32; lib.bonsai_n_vocab.argtypes=[ctypes.c_void_p]
lib.bonsai_context_create.restype=ctypes.c_int32
lib.bonsai_context_create.argtypes=[ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_eval.restype=ctypes.c_int32
lib.bonsai_eval.argtypes=[ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
lib.bonsai_context_close.argtypes=[ctypes.c_void_p]
lib.bonsai_model_close.argtypes=[ctypes.c_void_p]
m=ctypes.c_void_p()
rc=lib.bonsai_model_open(b"/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m))
print("open",rc,lib.bonsai_last_error())
nv=lib.bonsai_n_vocab(m)
ctx=ctypes.c_void_p(); print("ctx",lib.bonsai_context_create(m,256,256,ctypes.byref(ctx)))
a=(ctypes.c_int32*2)(12675,0); o=(ctypes.c_float*nv)()
print("eval",lib.bonsai_eval(ctx,a,2,o,nv))
got=np.frombuffer(o,dtype=np.float32)
print("top5",np.argsort(-got)[:5],"max",got.max())
lib.bonsai_context_close(ctx); lib.bonsai_model_close(m)
print("closed")
