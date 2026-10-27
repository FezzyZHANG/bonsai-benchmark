#!/usr/bin/env python3
"""Correctness / performance / VRAM harness for a libbonsai.so implementation."""
import argparse
import ctypes
import json
import math
import os
import re
import sys
import threading
import time

import numpy as np

OK = 0
IGNORE_EOS = 1


# ---------------------------------------------------------------------------
# ABI wrapper

class Engine:
    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        self.lib.bonsai_abi_version.restype = ctypes.c_uint32
        self.lib.bonsai_last_error.restype = ctypes.c_char_p
        self.lib.bonsai_model_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bonsai_model_open.restype = ctypes.c_int32
        self.lib.bonsai_model_close.argtypes = [ctypes.c_void_p]
        self.lib.bonsai_n_vocab.argtypes = [ctypes.c_void_p]
        self.lib.bonsai_n_vocab.restype = ctypes.c_int32
        self.lib.bonsai_n_ctx_train.argtypes = [ctypes.c_void_p]
        self.lib.bonsai_n_ctx_train.restype = ctypes.c_int32
        self.lib.bonsai_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
                                             ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
        self.lib.bonsai_tokenize.restype = ctypes.c_int32
        self.lib.bonsai_tensor_count.argtypes = [ctypes.c_void_p]
        self.lib.bonsai_tensor_count.restype = ctypes.c_int32
        self.lib.bonsai_tensor_info.argtypes = [ctypes.c_void_p, ctypes.c_int32,
                                                ctypes.POINTER(ctypes.c_char_p),
                                                ctypes.POINTER(ctypes.c_int32),
                                                ctypes.POINTER(ctypes.c_int32),
                                                ctypes.POINTER(ctypes.c_int64)]
        self.lib.bonsai_tensor_info.restype = ctypes.c_int32
        self.lib.bonsai_tensor_dequant.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                   ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
        self.lib.bonsai_tensor_dequant.restype = ctypes.c_int64
        self.lib.bonsai_context_create.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
                                                   ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bonsai_context_create.restype = ctypes.c_int32
        self.lib.bonsai_context_close.argtypes = [ctypes.c_void_p]
        self.lib.bonsai_context_reset.argtypes = [ctypes.c_void_p]
        self.lib.bonsai_context_reset.restype = ctypes.c_int32
        self.lib.bonsai_eval.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                         ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
        self.lib.bonsai_eval.restype = ctypes.c_int32
        self.lib.bonsai_generate_greedy.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                                    ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32,
                                                    ctypes.POINTER(ctypes.c_int32)]
        self.lib.bonsai_generate_greedy.restype = ctypes.c_int32
        self.model = ctypes.c_void_p()

    def err(self, what):
        msg = self.lib.bonsai_last_error()
        raise RuntimeError(f"{what}: {msg.decode() if msg else 'unknown error'}")

    def open(self, model_path):
        if self.lib.bonsai_model_open(model_path.encode(), ctypes.byref(self.model)) != 0:
            self.err("model_open")
        return self

    def close(self):
        if self.model:
            self.lib.bonsai_model_close(self.model)
            self.model = ctypes.c_void_p()

    def n_vocab(self):
        n = self.lib.bonsai_n_vocab(self.model)
        if n < 0:
            self.err("n_vocab")
        return n

    def n_ctx_train(self):
        return self.lib.bonsai_n_ctx_train(self.model)

    def tokenize(self, text, add_bos=0):
        raw = text.encode("utf-8")
        n = self.lib.bonsai_tokenize(self.model, raw, add_bos, None, 0)
        if n < 0:
            self.err("tokenize(count)")
        buf = (ctypes.c_int32 * n)()
        r = self.lib.bonsai_tokenize(self.model, raw, add_bos, buf, n)
        if r != n:
            self.err("tokenize")
        return list(buf)

    def tensor_count(self):
        n = self.lib.bonsai_tensor_count(self.model)
        if n < 0:
            self.err("tensor_count")
        return n

    def tensor_dequant(self, name, shape):
        arr = np.empty(int(np.prod(shape)), dtype=np.float32)
        r = self.lib.bonsai_tensor_dequant(self.model, name.encode(),
                                           arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                           arr.size)
        if r != arr.size:
            self.err(f"tensor_dequant({name}) -> {r} != {arr.size}")
        return arr.reshape(shape)

    def ctx(self, n_ctx, n_batch=512):
        c = ctypes.c_void_p()
        if self.lib.bonsai_context_create(self.model, n_ctx, n_batch, ctypes.byref(c)) != 0:
            self.err("context_create")
        return c

    def reset(self, ctx):
        if self.lib.bonsai_context_reset(ctx) != 0:
            self.err("context_reset")

    def free_ctx(self, ctx):
        self.lib.bonsai_context_close(ctx)

    def eval(self, ctx, ids, want_logits=True):
        n = len(ids)
        buf = (ctypes.c_int32 * n)(*ids)
        n_vocab = self.n_vocab()
        logits = np.empty(n_vocab, dtype=np.float32) if want_logits else None
        lp = logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if want_logits else None
        rc = self.lib.bonsai_eval(ctx, buf, n, lp, n_vocab if want_logits else 0)
        if rc != 0:
            self.err("eval")
        return logits

    def generate(self, ctx, prompt, n_predict, ignore_eos=True):
        p = (ctypes.c_int32 * len(prompt))(*prompt) if prompt else None
        out = (ctypes.c_int32 * n_predict)()
        rc = self.lib.bonsai_generate_greedy(ctx, p, len(prompt), n_predict,
                                             IGNORE_EOS if ignore_eos else 0, out)
        if rc < 0:
            self.err("generate")
        return list(out[:rc])


# ---------------------------------------------------------------------------
# VRAM / RSS monitor

def _proc_vram(pid=None):
    pid = pid or os.getpid()
    base = f"/proc/{pid}/fdinfo"
    try:
        names = os.listdir(base)
    except OSError:
        return 0
    clients = {}
    for name in names:
        try:
            txt = open(os.path.join(base, name)).read()
        except OSError:
            continue
        if "drm-driver:" not in txt:
            continue
        cid = re.search(r"drm-client-id:\s+(\d+)", txt)
        cid = cid.group(1) if cid else name
        m = re.search(r"drm-memory-vram:\s+(\d+)\s+KiB", txt)
        if m:
            clients[cid] = max(clients.get(cid, 0), int(m.group(1)) * 1024)
    return sum(clients.values())


def _vm_hwm():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class Monitor(threading.Thread):
    def __init__(self, interval=0.01):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_flag = threading.Event()
        self.peak = 0

    def run(self):
        while not self.stop_flag.is_set():
            self.peak = max(self.peak, _proc_vram())
            time.sleep(self.interval)

    def stop(self):
        self.stop_flag.set()
        self.join()


# ---------------------------------------------------------------------------
# Metrics

def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def logit_stats(gold, got):
    a, b = np.asarray(gold, np.float32), np.asarray(got, np.float32)
    pa, pb = softmax(a), softmax(b)
    top1 = int(np.argmax(a) == np.argmax(b))
    eps = 1e-300
    kl = float(np.sum(pa * np.log((pa + eps) / (pb + eps))))
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return top1, kl, cos


def logit_score(top1, kl, cos):
    return 0.4 * top1 + 0.4 * math.exp(-max(0.0, kl) / 0.01) + 0.2 * max(0.0, min(1.0, (cos - 0.99) / 0.01))


# ---------------------------------------------------------------------------
# Suites

def suite_tokenizer(eng, golden):
    cases = golden["tokenizer"]
    exact = 0
    details = []
    for c in cases:
        got = eng.tokenize(c["text"], 0)
        ok = got == c["ids"]
        exact += int(ok)
        details.append({"text": c["text"][:80], "ok": ok})
    score = exact / len(cases) if cases else 0.0
    return score, {"exact": exact, "total": len(cases), "details": details}


def suite_dequant(eng, golden, golden_dir):
    cases = golden["dequant"]
    exact = 0
    details = []
    for c in cases:
        ref = np.load(os.path.join(golden_dir, c["file"]))
        got = eng.tensor_dequant(c["name"], ref.shape)
        ok = got.shape == ref.shape and np.array_equal(got.view(np.uint32), ref.view(np.uint32))
        exact += int(ok)
        details.append({"name": c["name"], "shape": c["shape"], "ok": ok})
    score = exact / len(cases) if cases else 0.0
    return score, {"exact": exact, "total": len(cases), "details": details}


def suite_logits(eng, golden, golden_dir):
    cases = golden["logits"]
    scores = []
    details = []
    for i, c in enumerate(cases):
        ctx = eng.ctx(c["n_ctx"])
        got = eng.eval(ctx, c["tokens"])
        ref = np.fromfile(os.path.join(golden_dir, c["logits"]), dtype=np.float32)
        top1, kl, cos = logit_stats(ref, got)
        s = logit_score(top1, kl, cos)
        scores.append(s)
        details.append({"name": c["name"], "n_ctx": c["n_ctx"], "top1": top1,
                        "kl": kl, "cos": cos, "score": s})
        eng.free_ctx(ctx)
    return (sum(scores) / len(scores) if scores else 0.0), {"cases": details}


def suite_positions(eng, golden, golden_dir):
    cases = golden["positions"]
    all_top1 = []
    all_kl = []
    details = []
    for c in cases:
        ctx = eng.ctx(c["n_ctx"])
        cursor = 0
        top1s, kls = [], []
        for pos, fname in c["samples"]:
            prefix = c["tokens"][cursor:pos + 1]
            got = eng.eval(ctx, prefix)
            cursor = pos + 1
            ref = np.fromfile(os.path.join(golden_dir, fname), dtype=np.float32)
            top1, kl, _ = logit_stats(ref, got)
            top1s.append(top1)
            kls.append(kl)
        eng.free_ctx(ctx)
        mean_top1 = sum(top1s) / len(top1s) if top1s else 0.0
        mean_kl = sum(kls) / len(kls) if kls else 0.0
        all_top1.extend(top1s)
        all_kl.extend(kls)
        details.append({"name": c["name"], "samples": len(top1s),
                        "top1": mean_top1, "kl": mean_kl})
    top1 = sum(all_top1) / len(all_top1) if all_top1 else 0.0
    kl = sum(all_kl) / len(all_kl) if all_kl else 0.0
    score = 0.5 * top1 + 0.5 * math.exp(-max(0.0, kl) / 0.01)
    return score, {"top1": top1, "kl": kl, "score": score, "cases": details}


def suite_greedy(eng, golden):
    cases = golden["greedy"]
    total = 0
    same = 0
    details = []
    for c in cases:
        ctx = eng.ctx(c["n_ctx"])
        got = eng.generate(ctx, c["tokens"], 64, True)
        eng.free_ctx(ctx)
        n = min(len(got), len(c["ids"]))
        s = sum(1 for i in range(n) if got[i] == c["ids"][i])
        # missing tokens count as wrong
        same += s
        total += 64
        details.append({"name": c["name"], "n_ctx": c["n_ctx"],
                        "matched": s, "generated": len(got),
                        "lcp": next((i for i in range(n) if got[i] != c["ids"][i]), n)})
    score = same / total if total else 0.0
    return score, {"matched": same, "total": total, "cases": details}


def run_perf(eng, corpus_tokens, reps=3):
    def prompt_of(n):
        if n == 0:
            return []
        reps_ = (n + len(corpus_tokens) - 1) // len(corpus_tokens)
        return (corpus_tokens * reps_)[:n]

    pp = {}
    for n in [512, 2048, 4096, 8192]:
        toks = prompt_of(n)
        times = []
        for _ in range(reps):
            ctx = eng.ctx(n + 256)
            eng.eval(ctx, toks[:16], want_logits=False)  # warm
            eng.reset(ctx)
            t0 = time.perf_counter()
            eng.eval(ctx, toks, want_logits=False)
            times.append(time.perf_counter() - t0)
            eng.free_ctx(ctx)
        med = sorted(times)[len(times) // 2]
        pp[n] = n / med
    tg = {}
    for depth in [0, 512, 2048, 4096, 8192]:
        toks = prompt_of(depth)
        times = []
        for _ in range(reps):
            ctx = eng.ctx(depth + 256)
            if toks:
                eng.eval(ctx, toks, want_logits=False)
            t0 = time.perf_counter()
            eng.generate(ctx, [], 128, True)
            times.append(time.perf_counter() - t0)
            eng.free_ctx(ctx)
        med = sorted(times)[len(times) // 2]
        tg[depth] = 128 / med
    return {"pp": {str(k): v for k, v in pp.items()}, "tg": {str(k): v for k, v in tg.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-perf", action="store_true")
    args = ap.parse_args()

    golden_dir = os.path.dirname(os.path.abspath(args.golden))
    golden = json.load(open(args.golden))

    mon = Monitor()
    mon.start()
    eng = Engine(args.library)
    t0 = time.perf_counter()
    eng.open(args.model)
    load_s = time.perf_counter() - t0
    result = {
        "abi_version": eng.lib.bonsai_abi_version(),
        "n_vocab": eng.n_vocab(),
        "n_ctx_train": eng.n_ctx_train(),
        "tensor_count": eng.tensor_count(),
        "model_load_s": load_s,
    }

    scores = {}
    details = {}
    for name, fn in [
        ("tokenizer", lambda: suite_tokenizer(eng, golden)),
        ("dequant", lambda: suite_dequant(eng, golden, golden_dir)),
        ("logits", lambda: suite_logits(eng, golden, golden_dir)),
        ("positions", lambda: suite_positions(eng, golden, golden_dir)),
        ("greedy", lambda: suite_greedy(eng, golden)),
    ]:
        try:
            s, d = fn()
        except Exception as e:  # noqa: BLE001
            s, d = 0.0, {"error": repr(e)}
        scores[name] = s
        details[name] = d
        print(f"{name}: {s:.4f}", flush=True)

    weights = {"tokenizer": 0.10, "dequant": 0.15, "logits": 0.30,
               "positions": 0.15, "greedy": 0.30}
    correctness = sum(weights[k] * scores[k] for k in weights)
    result["correctness"] = {"score": correctness, "suites": scores, "details": details,
                             "accepted": correctness >= 0.90}

    if not args.skip_perf:
        result["perf"] = run_perf(eng, golden["corpus_tokens"])

    eng.close()
    mon.stop()
    result["vram_peak_bytes"] = mon.peak
    result["host_peak_rss_bytes"] = _vm_hwm()
    result["wall_s"] = time.perf_counter() - t0
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: result[k] for k in ("correctness", "vram_peak_bytes", "host_peak_rss_bytes")}, indent=1))


if __name__ == "__main__":
    main()
