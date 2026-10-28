import ctypes, numpy as np, struct, json, sys
lib=ctypes.CDLL("/workspace/zig-out/lib/libbonsai.so")
lib.bonsai_model_open.restype=ctypes.c_int32
lib.bonsai_model_open.argtypes=[ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_tensor_count.restype=ctypes.c_int32; lib.bonsai_tensor_count.argtypes=[ctypes.c_void_p]
lib.bonsai_tensor_info.argtypes=[ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64)]
lib.bonsai_tensor_dequant.restype=ctypes.c_int64
lib.bonsai_tensor_dequant.argtypes=[ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
m=ctypes.c_void_p(); print("open",lib.bonsai_model_open(b"/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m)))
fh=open("/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors","rb")
nh=struct.unpack("<Q",fh.read(8))[0]; hdr=json.loads(fh.read(nh)); base=8+nh
name=ctypes.c_char_p(); typ=ctypes.c_int32(); rank=ctypes.c_int32(); ne=(ctypes.c_int64*4)()
bad=0; checked=0; total=lib.bonsai_tensor_count(m)
for i in range(total):
    lib.bonsai_tensor_info(m,i,ctypes.byref(name),ctypes.byref(typ),ctypes.byref(rank),ctypes.cast(ne, ctypes.POINTER(ctypes.c_int64)))
    nm=name.value.decode()
    e=hdr[nm]
    cnt=1
    for k in range(rank.value): cnt*=ne[k]
    cn=list(ne)[:rank.value]
    # expected torch shape = reversed gguf dims
    exp_shape=list(reversed(cn))
    if list(e["shape"])!=exp_shape:
        print("SHAPE MISMATCH",nm,exp_shape,e["shape"]); bad+=1; continue
    out=(ctypes.c_float*cnt)()
    r=lib.bonsai_tensor_dequant(m, name.value, out, cnt)
    if r!=cnt:
        print("RET MISMATCH",nm,r,cnt); bad+=1; continue
    got=np.frombuffer(out,dtype=np.float32)
    fh.seek(base+e["data_offsets"][0])
    nbytes=e["data_offsets"][1]-e["data_offsets"][0]
    dtype=np.float16 if e["dtype"]=="F16" else np.float32
    ref=np.frombuffer(fh.read(nbytes),dtype=dtype).astype(np.float32)
    md=float(np.abs(got-ref).max()) if cnt else 0.0
    if md!=0.0:
        print("VALUE MISMATCH",nm,md); bad+=1
    checked+=1
print("checked",checked,"of",total,"bad",bad)
