import ctypes, numpy as np
lib=ctypes.CDLL("/workspace/zig-out/lib/libbonsai.so")
lib.bonsai_abi_version.restype=ctypes.c_uint32
lib.bonsai_model_open.restype=ctypes.c_int32
lib.bonsai_model_open.argtypes=[ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_last_error.restype=ctypes.c_char_p
lib.bonsai_n_vocab.restype=ctypes.c_int32; lib.bonsai_n_vocab.argtypes=[ctypes.c_void_p]
lib.bonsai_n_ctx_train.restype=ctypes.c_int32; lib.bonsai_n_ctx_train.argtypes=[ctypes.c_void_p]
lib.bonsai_tokenize.restype=ctypes.c_int32
lib.bonsai_tokenize.argtypes=[ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
lib.bonsai_context_create.restype=ctypes.c_int32
lib.bonsai_context_create.argtypes=[ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_eval.restype=ctypes.c_int32
lib.bonsai_eval.argtypes=[ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
lib.bonsai_context_reset.restype=ctypes.c_int32; lib.bonsai_context_reset.argtypes=[ctypes.c_void_p]
lib.bonsai_generate_greedy.restype=ctypes.c_int32
lib.bonsai_generate_greedy.argtypes=[ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)]
lib.bonsai_tensor_dequant.restype=ctypes.c_int64
lib.bonsai_tensor_dequant.argtypes=[ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
lib.bonsai_tensor_info.argtypes=[ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64)]
lib.bonsai_context_close.argtypes=[ctypes.c_void_p]
lib.bonsai_model_close.argtypes=[ctypes.c_void_p]
def err():
    e=lib.bonsai_last_error(); return e.decode() if e else None
print("abi",lib.bonsai_abi_version())
print("null model n_vocab",lib.bonsai_n_vocab(None),err())
m=ctypes.c_void_p()
print("bad path",lib.bonsai_model_open(b"/nonexistent.gguf",ctypes.byref(m)),err())
print("open",lib.bonsai_model_open(b"/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf",ctypes.byref(m)))
print("n_vocab",lib.bonsai_n_vocab(m),"n_ctx_train",lib.bonsai_n_ctx_train(m))
print("tok empty",lib.bonsai_tokenize(m,b"",0,None,0))
n=lib.bonsai_tokenize(m,b"hi",0,None,0)
small=(ctypes.c_int32*1)()
print("tok cap small",lib.bonsai_tokenize(m,b"hi",0,small,1),err(),"need",n)
buf=(ctypes.c_int32*n)(); print("tok ok",lib.bonsai_tokenize(m,b"hi",0,buf,n),list(buf))
print("tok addbos",lib.bonsai_tokenize(m,b"hi",1,None,0))
# tensor info bounds
name=ctypes.c_char_p(); typ=ctypes.c_int32(); rank=ctypes.c_int32(); ne=(ctypes.c_int64*4)()
print("tinfo -1",lib.bonsai_tensor_info(m,-1,ctypes.byref(name),ctypes.byref(typ),ctypes.byref(rank),ne),err())
print("tinfo 99999",lib.bonsai_tensor_info(m,99999,ctypes.byref(name),ctypes.byref(typ),ctypes.byref(rank),ne),err())
print("dequant unknown",lib.bonsai_tensor_dequant(m,b"nope",None,0),err())
out=(ctypes.c_float*4)()
print("dequant small",lib.bonsai_tensor_dequant(m,b"blk.0.ssm_a",out,1),err())
print("dequant ok",lib.bonsai_tensor_dequant(m,b"blk.0.ssm_a",out,4))
# context
ctx=ctypes.c_void_p()
print("ctx bad",lib.bonsai_context_create(m,0,1,ctypes.byref(ctx)),err())
print("ctx ok",lib.bonsai_context_create(m,64,64,ctypes.byref(ctx)))
nv=lib.bonsai_n_vocab(m); o=(ctypes.c_float*nv)()
a1=(ctypes.c_int32*1)(12675)
print("eval null toks",lib.bonsai_eval(ctx,None,1,o,nv),err())
print("eval n=0",lib.bonsai_eval(ctx,a1,0,o,nv),err())
print("eval small logits",lib.bonsai_eval(ctx,a1,1,out,4),err())
print("eval bad token",lib.bonsai_eval(ctx,(ctypes.c_int32*1)(-5),1,o,nv),err())
print("eval ok",lib.bonsai_eval(ctx,a1,1,o,nv))
big=(ctypes.c_int32*100)(*([12675]*100))
print("eval overflow",lib.bonsai_eval(ctx,big,100,o,nv),err())
lib.bonsai_context_reset(ctx)
print("gen prompt",lib.bonsai_generate_greedy(ctx,a1,1,3,0,(ctypes.c_int32*3)()))
print("gen ignore eos",lib.bonsai_generate_greedy(ctx,a1,1,3,1,(ctypes.c_int32*3)()))
lib.bonsai_context_reset(ctx)
print("gen empty fresh",lib.bonsai_generate_greedy(ctx,None,0,3,0,(ctypes.c_int32*3)()),err())
lib.bonsai_context_close(ctx)
lib.bonsai_model_close(m)
print("closed ok")
