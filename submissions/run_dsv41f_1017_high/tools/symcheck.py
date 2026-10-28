import ctypes
lib=ctypes.CDLL("/workspace/zig-out/lib/libbonsai.so")
names=["bonsai_abi_version","bonsai_last_error","bonsai_model_open","bonsai_model_close","bonsai_n_vocab","bonsai_n_ctx_train","bonsai_tokenize","bonsai_tensor_count","bonsai_tensor_info","bonsai_tensor_dequant","bonsai_context_create","bonsai_context_close","bonsai_context_reset","bonsai_eval","bonsai_generate_greedy"]
for n in names:
    try:
        f=getattr(lib,n); print(n,"OK")
    except AttributeError:
        print(n,"MISSING")
print("abi_version",lib.bonsai_abi_version())
