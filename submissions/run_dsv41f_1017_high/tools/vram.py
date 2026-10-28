import ctypes, os, glob, time
def vram():
    tot=0
    for f in glob.glob("/proc/self/fdinfo/*"):
        try: txt=open(f).read()
        except Exception: continue
        for line in txt.splitlines():
            if line.startswith("drm-memory-vram:"):
                tot+=int(line.split(":")[1].strip().split()[0])
    return tot
def hwm():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"): return line.strip()
lib=ctypes.CDLL("/workspace/zig-out/lib/libbonsai.so")
lib.bonsai_model_open.restype=ctypes.c_int32
lib.bonsai_model_open.argtypes=[ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_n_vocab.restype=ctypes.c_int32; lib.bonsai_n_vocab.argtypes=[ctypes.c_void_p]
lib.bonsai_context_create.restype=ctypes.c_int32
lib.bonsai_context_create.argtypes=[ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_eval.restype=ctypes.c_int32
lib.bonsai_eval.argtypes=[ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
print("vram before", vram()/2**20, "MiB", hwm())
t0=time.time()
m=ctypes.c_void_p(); rc=lib.bonsai_model_open(b"/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m))
print("open rc",rc,"t=%.2f"%(time.time()-t0),"vram",vram()/2**20,"MiB",hwm())
nv=lib.bonsai_n_vocab(m)
ctx=ctypes.c_void_p(); lib.bonsai_context_create(m, 16384, 512, ctypes.byref(ctx))
print("after ctx16384 vram",vram()/2**20,"MiB",hwm())
arr=(ctypes.c_int32*1)(12675); out=(ctypes.c_float*nv)()
lib.bonsai_eval(ctx,arr,1,out,nv)
print("after eval vram",vram()/2**20,"MiB",hwm())
