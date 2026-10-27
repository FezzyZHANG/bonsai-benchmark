#!/usr/bin/env python3
"""Dequantize a PTQ1_0 GGUF and write a single safetensors file.

PTQ1_0 values are trit * fp16_group_scale, so float16 represents them exactly.
F32 tensors in the GGUF (norms/biases/SSM scalars) are kept as F32, because most
of them are not exactly representable in F16.

Tensor names are preserved from the GGUF. GGUF's ne[0] is the contiguous/inner
dimension, so 2-D tensors are written with PyTorch/HF shape (ne1, ne0).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptq1_0 import dequantize_tensor, BLOCK_BYTES, BLOCK_SIZE  # noqa: E402

PTQ1_0 = 143
F32 = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out")
    ap.add_argument("--gguf-py", default="llama.cpp-prism/gguf-py")
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    sys.path.insert(0, args.gguf_py)
    from gguf import GGUFReader

    t0 = time.time()
    reader = GGUFReader(args.gguf)
    print(f"loaded {len(reader.tensors)} tensors from {args.gguf}", flush=True)

    tensors = {}
    manifest = []
    total_elems = 0
    f16_elems = 0
    f32_elems = 0

    for i, t in enumerate(reader.tensors):
        name = t.name
        ne = [int(x) for x in t.shape.tolist()]
        st = time.time()

        if t.tensor_type == PTQ1_0:
            if len(ne) != 2:
                raise ValueError(f"{name}: expected 2-D PTQ1_0 tensor, got {ne}")
            ne0, ne1 = ne[0], ne[1]
            raw = np.asarray(t.data, dtype=np.uint8).reshape(-1)
            if raw.size != (ne0 // BLOCK_SIZE) * BLOCK_BYTES * ne1:
                raise ValueError(f"{name}: raw size {raw.size} inconsistent with {ne}")
            arr = dequantize_tensor(raw, ne0, ne1)
            d = raw.reshape(-1, BLOCK_BYTES)[:, 26:28].copy().view("<f2")
            if not np.isfinite(d).all():
                raise ValueError(f"{name}: non-finite group scale")
            assert arr.dtype == np.float16
            out_dtype = "F16"
            f16_elems += arr.size
        elif t.tensor_type == F32:
            arr = np.asarray(t.data, dtype=np.float32)
            out_dtype = "F32"
            f32_elems += arr.size
        else:
            raise ValueError(f"{name}: unsupported type {t.tensor_type}")

        arr = np.ascontiguousarray(arr)
        tensors[name] = arr
        total_elems += arr.size
        manifest.append({
            "name": name,
            "source_type": t.tensor_type.name,
            "output_dtype": out_dtype,
            "gguf_ne": ne,
            "shape": list(arr.shape),
            "n_elements": int(arr.size),
        })
        if i % 20 == 0 or i + 1 == len(reader.tensors):
            print(f"[{i+1:3d}/{len(reader.tensors)}] {name:40s} {out_dtype} "
                  f"shape={arr.shape} {time.time()-st:.2f}s "
                  f"total={time.time()-t0:.1f}s", flush=True)

    metadata = {
        "format": "pt",
        "source_gguf": args.gguf.split("/")[-1],
        "source_quant": "PTQ1_0 (ggml type 143, group-128 ternary, 1.75 bpw)",
        "dequant_note": "PTQ1_0 values are trit * fp16 group scale; F16 is bit-exact. "
                        "F32 GGUF tensors (norms/biases/SSM scalars) are preserved as F32.",
        "tensor_names": "llama.cpp/GGUF names",
        "shape_order": "PyTorch row-major; GGUF ne[0] (contiguous/inner) appears last",
    }

    print(f"saving {len(tensors)} tensors, {total_elems} elements "
          f"(F16 {f16_elems}, F32 {f32_elems}) to {args.out}", flush=True)
    ts = time.time()
    save_file(tensors, args.out, metadata=metadata)
    print(f"saved in {time.time()-ts:.1f}s; total {time.time()-t0:.1f}s", flush=True)

    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump({
                "source_gguf": args.gguf,
                "output": args.out,
                "n_tensors": len(tensors),
                "n_elements": total_elems,
                "f16_elements": f16_elems,
                "f32_elements": f32_elems,
                "tensors": manifest,
            }, f, indent=2)
        print(f"wrote manifest {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
