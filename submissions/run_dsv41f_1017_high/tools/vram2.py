import ctypes, glob
lib=ctypes.CDLL("/workspace/zig-out/lib/libbonsai.so")
lib.bonsai_model_open.restype=ctypes.c_int32
lib.bonsai_model_open.argtypes=[ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
m=ctypes.c_void_p(); print("open", lib.bonsai_model_open(b"/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m)))
for f in glob.glob("/proc/self/fdinfo/*"):
    try: txt=open(f).read()
    except Exception: continue
    lines=[l for l in txt.splitlines() if l.startswith("drm")]
    if lines:
        print(f, lines[:8])
