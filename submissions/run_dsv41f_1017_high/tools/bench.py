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
t0=time.time(); m=ctypes.c_void_p(); print("open",lib.bonsai_model_open(b"/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m)), "t=%.2fs"%(time.time()-t0))
nv=lib.bonsai_n_vocab(m)
ctx=ctypes.c_void_p(); lib.bonsai_context_create(m, 256, 256, ctypes.byref(ctx))
arr=(ctypes.c_int32*1)(12675); out=(ctypes.c_float*nv)()
t0=time.time(); lib.bonsai_eval(ctx,arr,1,out,nv); dt=time.time()-t0
print("1 token eval %.3fs -> %.2f tok/s"%(dt,1/dt))
lib.bonsai_context_reset(ctx)
# prompt 32 tokens, eval one call
prompt=[12675,0]*16
pa=(ctypes.c_int32*32)(*prompt)
t0=time.time(); lib.bonsai_eval(ctx,pa,32,out,nv); dt=time.time()-t0
print("32 token eval %.3fs -> %.2f tok/s"%(dt,32/dt))
lib.bonsai_context_reset(ctx)
t0=time.time()
n=lib.bonsai_generate_greedy(ctx, pa, 8, 8, 0, (ctypes.c_int32*8)())
o=(ctypes.c_int32*8)()
n=lib.bonsai_generate_greedy(ctx, pa, 8, 8, 0, o)
dt=time.time()-t0
print("gen 8 after 8 prompt: n=%d ids=%s t=%.2fs"%(n,list(o),dt))
lib.bonsai_model_close.argtypes=[ctypes.c_void_p]
lib.bonsai_model_close(m)
