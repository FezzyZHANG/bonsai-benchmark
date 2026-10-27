#!/usr/bin/env python3
"""Generate hidden golden data for the benchmark with the CPU reference."""
import argparse
import json
import os
import sys

import numpy as np
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ref_client import RefClient  # noqa: E402


def build_cases(corpus_tokens):
    # Last-token logits and greedy prompts at four contexts.
    logits_specs = [
        ("ctx512", 512, 0, 448),
        ("ctx2048", 2048, 512, 1984),
        ("ctx4096", 4096, 2500, 4032),
        ("ctx8192", 8192, 7000, 8128),
    ]
    logits = []
    for name, n_ctx, off, n in logits_specs:
        toks = corpus_tokens[off:off + n]
        if len(toks) < n:
            raise SystemExit(f"corpus too short for {name}")
        logits.append({"name": name, "n_ctx": n_ctx, "tokens": toks})

    greedy_specs = [
        ("greedy_short", 1024, 0, 64),
        ("greedy_med", 4096, 512, 2048),
        ("greedy_long", 16384, 7000, 8192),
    ]
    greedy = []
    for name, n_ctx, off, n in greedy_specs:
        toks = corpus_tokens[off:off + n]
        if len(toks) < n:
            raise SystemExit(f"corpus too short for {name}")
        greedy.append({"name": name, "n_ctx": n_ctx, "tokens": toks})

    positions = []
    for name, n_ctx in [("pos512", 512), ("pos2048", 2048)]:
        toks = corpus_tokens[:n_ctx]
        if len(toks) < n_ctx:
            raise SystemExit(f"corpus too short for {name}")
        step = max(1, (n_ctx - 128) // 32)
        pos = list(range(128, n_ctx, step))[:32]
        positions.append({"name": name, "n_ctx": n_ctx, "tokens": toks, "positions": pos})
    return logits, greedy, positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--tests", default=os.path.join(HERE, "..", "tests"))
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tests = os.path.abspath(args.tests)
    out = os.path.abspath(args.out or os.path.join(tests, "golden"))
    os.makedirs(out, exist_ok=True)

    corpus = open(os.path.join(tests, "corpus.txt"), encoding="utf-8").read()
    tok_cases = json.load(open(os.path.join(tests, "tokenizer_cases.json")))
    ref = RefClient(args.oracle, stderr_path=os.path.join(out, "oracle.log"))
    ref.load(args.model)

    corpus_tokens = ref.tokenize(corpus, 0)
    tokenizer = []
    for text in tok_cases:
        tokenizer.append({"text": text, "ids": ref.tokenize(text, 0)})

    logits_cases, greedy_cases, pos_cases = build_cases(corpus_tokens)
    manifest = {
        "corpus_tokens": corpus_tokens,
        "tokenizer": tokenizer,
        "logits": [],
        "greedy": [],
        "positions": [],
        "dequant": [],
    }

    for i, case in enumerate(logits_cases):
        ref.ctx(i, case["n_ctx"], "cpu")
        ref.eval(i, case["tokens"])
        f = os.path.join(out, f"{case['name']}.logits.bin")
        n_vocab = ref.logits(i, f)
        ids = ref.gen(i, 64, True)
        ref.reset(i)
        manifest["logits"].append({
            "name": case["name"], "n_ctx": case["n_ctx"], "tokens": case["tokens"],
            "n_vocab": n_vocab, "logits": os.path.basename(f), "greedy": ids,
        })
        print(f"logits {case['name']} ({case['n_ctx']}, {len(case['tokens'])} tok)", flush=True)

    for i, case in enumerate(greedy_cases):
        cid = 100 + i
        ref.ctx(cid, case["n_ctx"], "cpu")
        ref.eval(cid, case["tokens"])
        ids = ref.gen(cid, 64, True)
        ref.reset(cid)
        manifest["greedy"].append({
            "name": case["name"], "n_ctx": case["n_ctx"], "tokens": case["tokens"], "ids": ids,
        })
        print(f"greedy {case['name']} ({case['n_ctx']}, {len(case['tokens'])} tok)", flush=True)

    for i, case in enumerate(pos_cases):
        cid = 200 + i
        ref.ctx(cid, case["n_ctx"], "cpu")
        wanted = set(case["positions"])
        files = []
        for t, tok in enumerate(case["tokens"]):
            ref.eval(cid, [tok])
            if t in wanted:
                f = os.path.join(out, f"{case['name']}_{t:05d}.logits.bin")
                ref.logits(cid, f)
                files.append((t, os.path.basename(f)))
        ref.reset(cid)
        manifest["positions"].append({
            "name": case["name"], "n_ctx": case["n_ctx"], "tokens": case["tokens"],
            "samples": files,
        })
        print(f"positions {case['name']} ({len(files)} samples)", flush=True)

    # Dequant samples: full small/medium tensors, stored as .npy.
    dequant_specs = [
        "blk.0.attn_gate.weight",
        "blk.3.attn_v.weight",
        "blk.0.ssm_alpha.weight",
        "blk.0.ssm_conv1d.weight",
        "output_norm.weight",
    ]
    with safe_open(args.safetensors, framework="numpy") as f:
        for j, name in enumerate(dequant_specs):
            arr = f.get_tensor(name)
            fn = f"dequant_{j}.npy"
            np.save(os.path.join(out, fn), arr)
            manifest["dequant"].append({
                "name": name, "shape": list(arr.shape), "dtype": str(arr.dtype),
                "file": fn,
            })
            print(f"dequant {name} {arr.shape} {arr.dtype}", flush=True)

    ref.close()
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    print(f"wrote {out}/manifest.json")


if __name__ == "__main__":
    main()
