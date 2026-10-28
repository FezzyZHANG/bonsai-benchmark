import ctypes, numpy as np, json, sys, struct
lib=ctypes.CDLL("/workspace/zig-out/lib/libbonsai.so")
lib.bonsai_model_open.restype=ctypes.c_int32
lib.bonsai_model_open.argtypes=[ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_last_error.restype=ctypes.c_char_p
lib.bonsai_n_vocab.restype=ctypes.c_int32
lib.bonsai_n_vocab.argtypes=[ctypes.c_void_p]
lib.bonsai_tensor_count.restype=ctypes.c_int32
lib.bonsai_tensor_count.argtypes=[ctypes.c_void_p]
m=ctypes.c_void_p()
rc=lib.bonsai_model_open(b"/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m))
print("open rc",rc,"err",lib.bonsai_last_error(), "model",m)
if rc!=0: sys.exit(1)
print("n_vocab",lib.bonsai_n_vocab(m),"n_tensors",lib.bonsai_tensor_count(m))
lib.bonsai_tokenize.restype=ctypes.c_int32
lib.bonsai_tokenize.argtypes=[ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
vecs=json.load(open("/workspace/TOKENIZER-VECTORS.json"))
for v in vecs:
    n=lib.bonsai_tokenize(m, v["text"].encode(), 0, None, 0)
    buf=(ctypes.c_int32*n)()
    lib.bonsai_tokenize(m, v["text"].encode(), 0, buf, n)
    got=list(buf)
    print("tok", got==v["ids"], v["text"][:25], got[:8], v["ids"][:8])
# tensor dequant check vs safetensors for a couple tensors
lib.bonsai_tensor_dequant.restype=ctypes.c_int64
lib.bonsai_tensor_dequant.argtypes=[ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
lib.bonsai_tensor_info.argtypes=[ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64)]
name=ctypes.c_char_p(); typ=ctypes.c_int32(); rank=ctypes.c_int32(); ne=(ctypes.c_int64*4)()
rc=lib.bonsai_tensor_info(m, 0, ctypes.byref(name), ctypes.byref(typ), ctypes.byref(rank), ne)
print("tinfo0",rc,name.value,typ.value,rank.value,list(ne))
fh=open("/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors","rb")
nh=struct.unpack("<Q",fh.read(8))[0]; hdr=json.loads(fh.read(nh)); base=8+nh
for tname in [b"blk.0.attn_gate.weight", b"token_embd.weight"]:
    # find index
    idx=None
    for i in range(lib.bonsai_tensor_count(m)):
        lib.bonsai_tensor_info(m,i,ctypes.byref(name),ctypes.byref(typ),ctypes.byref(rank),ne)
        if name.value==tname: idx=i; break
    lib.bonsai_tensor_info(m,idx,ctypes.byref(name),ctypes.byref(typ),ctypes.byref(rank),ne)
    cnt=1
    for k in range(rank.value): cnt*=ne[k]
    cap=cnt
    out=(ctypes.c_float*cap)()
    r=lib.bonsai_tensor_dequant(m,tname,out,cap)
    e=hdr[tname.decode()]
    fh.seek(base+e["data_offsets"][0])
    rows=e["shape"][0]; cols=e["shape"][1]
    ref=np.frombuffer(fh.read(rows*cols*2),dtype=np.float16).astype(np.float32)
    got=np.frombuffer(out,dtype=np.float32)
    print(tname,"deq ret",r,"cnt",cnt,"maxdiff",np.abs(got-ref).max(),"ne",list(ne),"stshape",e["shape"])
