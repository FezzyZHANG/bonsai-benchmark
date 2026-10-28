#!/usr/bin/env python3
"""Minimal ctypes driver for the ABI stub."""
import ctypes
import sys

lib = ctypes.CDLL(sys.argv[1] if len(sys.argv) > 1 else "zig-out/lib/libbonsai.so")
lib.bonsai_abi_version.restype = ctypes.c_uint32
lib.bonsai_last_error.restype = ctypes.c_char_p

print("abi version:", lib.bonsai_abi_version())
print("last error :", lib.bonsai_last_error())

lib.bonsai_model_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_model_open.restype = ctypes.c_int32
model = ctypes.c_void_p()
rc = lib.bonsai_model_open(b"data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(model))
print("model_open :", rc, "model:", model.value)
