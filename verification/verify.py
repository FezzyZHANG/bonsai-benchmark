#!/usr/bin/env python3
"""Build a submission, run correctness/perf/VRAM tests, and write a report."""
import argparse
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
PY = os.path.join(BENCH, "deps/python/bin/python")
MODEL = os.path.join(BENCH, "deps/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf")
ST = os.path.join(BENCH, "deps/data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors")
ORACLE = os.path.join(BENCH, "deps/ref/bin/bonsai-ref")
ORACLE_CPU = os.path.join(HERE, "bin/bonsai-ref-cpu")
if not os.path.exists(ORACLE_CPU):
    ORACLE_CPU = os.path.join(BENCH, "deps/ref/bin/bonsai-ref-cpu")
GOLDEN = os.path.join(HERE, "tests/golden/manifest.json")
AGENT_IMAGE = os.environ.get("AGENT_IMAGE", "bonsai-agent:1.0")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def run(cmd, **kw):
    log("+", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, **kw)


MAX_SUBMISSION_BYTES = 32 * 2**20


def submission_size(sub):
    """Bytes of regular files, ignoring build outputs and the mounted trees."""
    skip = {".git", "zig-out", ".zig-cache", "zig-cache", "node_modules",
            ".claude", ".claude-home", ".tmp", "data", "docs", "ref"}
    total = 0
    # Symlinked directories are not followed: build_submission copies symlinks
    # as symlinks, so the size check and the copied tree stay consistent.
    for root, dirs, files in os.walk(sub, followlinks=False):
        dirs[:] = [d for d in dirs if d not in skip]
        for name in files:
            if name == ".claude.json":
                continue
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def build_submission(sub, work, no_container):
    src = os.path.join(work, "src")
    ignored = shutil.ignore_patterns(
        ".git", "zig-out", ".zig-cache", "zig-cache", "node_modules",
        ".claude", ".claude-home", ".claude.json", ".tmp",
        "data", "docs", "ref")

    def ignore(dirname, names):
        out = set(ignored(dirname, names))
        for name in names:
            if name in out:
                continue
            try:
                mode = os.lstat(os.path.join(dirname, name)).st_mode
            except OSError:
                out.add(name)
                continue
            # Sockets/FIFOs/device nodes are not safe for copytree/copyfile.
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode)):
                out.add(name)
        return out

    # Copy symlinks as symlinks: a directory symlink must not pull an excluded
    # tree (or a weight file) into the build, and submission_size ignores it.
    shutil.copytree(sub, src, symlinks=True, ignore=ignore)
    if no_container:
        zig = os.environ.get("ZIG", os.path.join(BENCH, "deps/toolchains/zig-x86_64-linux-0.17.0/zig"))
        env = dict(os.environ, ZIG_GLOBAL_CACHE_DIR=os.path.join(work, "zig-cache"))
        try:
            r = run([zig, "build", "-Doptimize=ReleaseFast"], cwd=src, env=env,
                    capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            return None, "build timed out after 3600 s"
    else:
        name = f"bonsai-build-{os.getpid()}"
        cache = os.path.join(work, "zig-cache")
        os.makedirs(cache, exist_ok=True)
        try:
            r = run(["podman", "run", "--rm", "--name", name,
                     "--network=none", "--security-opt", "label=disable",
                     "-v", f"{src}:/src", "-v", f"{cache}:/cache", "-w", "/src",
                     "-e", "ZIG_GLOBAL_CACHE_DIR=/cache",
                     AGENT_IMAGE, "zig", "build", "-Doptimize=ReleaseFast"],
                    capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            run(["podman", "rm", "-f", name], capture_output=True)
            return None, "build timed out after 3600 s"
    lib = os.path.join(src, "zig-out/lib/libbonsai.so")
    ok = r.returncode == 0 and os.path.exists(lib)
    return lib if ok else None, (r.stdout + r.stderr)[-5000:]


def ensure_golden():
    if os.path.exists(GOLDEN):
        return
    log("golden data missing; generating with CPU oracle (this takes a few minutes)")
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    try:
        r = run([PY, os.path.join(HERE, "tools/generate_golden.py"),
                 "--model", MODEL, "--oracle", ORACLE_CPU, "--safetensors", ST,
                 "--tests", os.path.join(HERE, "tests"), "--out", os.path.dirname(GOLDEN)],
                timeout=3600)
    except subprocess.TimeoutExpired:
        raise SystemExit("golden generation timed out after 3600 s")
    if r.returncode != 0:
        raise SystemExit("golden generation failed")


def run_harness(lib, out_json, no_container, skip_perf, perf_short=False,
                perf_threshold=900.0):
    if no_container:
        cmd = [PY, os.path.join(HERE, "harness/agent_harness.py"),
               "--library", lib, "--model", MODEL, "--golden", GOLDEN, "--out", out_json]
        if skip_perf:
            cmd.append("--skip-perf")
        if perf_short:
            cmd.append("--perf-short")
        cmd += ["--perf-threshold", str(perf_threshold)]
    else:
        name = f"bonsai-harness-{os.getpid()}"
        cmd = ["podman", "run", "--rm", "--name", name,
               "--device", "/dev/dri", "--group-add", "keep-groups",
               "--security-opt", "label=disable", "--network=none"]
        if os.environ.get("BONSAI_SHIM_BACKEND"):
            cmd += ["-e", "BONSAI_SHIM_BACKEND=" + os.environ["BONSAI_SHIM_BACKEND"]]
        cmd += ["-v", f"{lib}:/libbonsai.so:ro",
               "-v", f"{MODEL}:/model.gguf:ro",
               "-v", f"{os.path.dirname(GOLDEN)}:/golden:ro",
               "-v", f"{HERE}/harness:/harness:ro",
               "-v", f"{os.path.dirname(out_json)}:/out",
               AGENT_IMAGE, "/opt/bonsai/venv/bin/python", "/harness/agent_harness.py",
               "--library", "/libbonsai.so", "--model", "/model.gguf",
               "--golden", "/golden/manifest.json", "--out", f"/out/{os.path.basename(out_json)}"]
        if skip_perf:
            cmd.append("--skip-perf")
        if perf_short:
            cmd.append("--perf-short")
        cmd += ["--perf-threshold", str(perf_threshold)]
    try:
        r = run(cmd, timeout=7200)
    except subprocess.TimeoutExpired:
        if not no_container:
            run(["podman", "rm", "-f", name], capture_output=True)
        raise SystemExit("harness timed out after 7200 s")
    if r.returncode != 0:
        raise SystemExit(f"harness failed (exit {r.returncode})")


def _proc_vram(pid):
    """Sum amdgpu VRAM across this process's DRM clients (deduped)."""
    import re
    clients = {}
    try:
        names = os.listdir(f"/proc/{pid}/fdinfo")
    except OSError:
        return 0
    for name in names:
        try:
            txt = open(f"/proc/{pid}/fdinfo/{name}").read()
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


def _proc_rss_hwm(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def measure_reference(no_container):
    """Measure reference Vulkan pp/tg and peak VRAM/RSS on this machine."""
    import threading
    sys.path.insert(0, os.path.join(HERE, "tools"))
    from ref_client import RefClient
    tok = RefClient(ORACLE, stderr_path=None)
    tok.load(MODEL)
    corpus = json.load(open(GOLDEN))["corpus_tokens"]

    def toks(n):
        if n == 0:
            return []
        r = (n + len(corpus) - 1) // len(corpus)
        return (corpus * r)[:n]

    # Peak VRAM at the largest context any test uses (greedy_long: n_ctx
    # 16384). Sample while the oracle actually decodes, as the harness does.
    stopped = threading.Event()
    peak = [0]

    def watch():
        while not stopped.is_set():
            peak[0] = max(peak[0], _proc_vram(tok.p.pid))
            time.sleep(0.01)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    tok.ctx(900, 16384, "vulkan")
    tok.eval(900, toks(8192))
    tok.gen(900, 64, True)
    stopped.set()
    t.join()
    ref_peak = peak[0]
    ref_rss = _proc_rss_hwm(tok.p.pid)
    tok.reset(900)

    pp = {}
    for n in [512, 2048, 4096, 8192]:
        times = []
        for _ in range(5):
            tok.ctx(901, n + 256, "vulkan")
            tok.eval(901, toks(16))  # warm
            tok.reset(901)
            t0 = time.perf_counter()
            tok.eval(901, toks(n))
            times.append(time.perf_counter() - t0)
            tok.reset(901)
        pp[str(n)] = n / sorted(times)[len(times) // 2]
    tg = {}
    for depth in [0, 512, 2048, 4096, 8192]:
        times = []
        for _ in range(5):
            tok.ctx(902, max(depth, 1) + 256, "vulkan")
            tok.eval(902, toks(depth) if depth else [248044])
            t0 = time.perf_counter()
            tok.gen(902, 128, True)
            times.append(time.perf_counter() - t0)
            tok.reset(902)
        tg[str(depth)] = 128 / sorted(times)[len(times) // 2]
    tok.close()
    return {"pp": pp, "tg": tg, "vram_peak_bytes": ref_peak, "host_peak_rss_bytes": ref_rss}


def geom(vals):
    vals = [v for v in vals if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0


def write_report(rep, out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    c = rep.get("correctness", {})
    L = ["# bonsai-vk verification report", ""]
    L.append(f"- build: **{'ok' if rep.get('build_ok') else 'FAILED'}**")
    if not rep.get("build_ok"):
        L += ["", "```", rep.get("build_log", ""), "```"]
    else:
        if c.get("partial"):
            cur = rep.get("correctness_current")
            what = ("harness failed" if rep.get("harness_error")
                    else "not all suites ran")
            if cur:
                what += f" during {cur}"
            L.append(f"- **correctness (partial): {c.get('score', 0):.4f}** "
                     f"({len(c.get('finished', []))}/{c.get('suites_total', 5)} "
                     f"suites; {what})")
            L.append("  - the subtotal treats missing suites as 0; it is a "
                     "lower bound, not an acceptance score")
        else:
            L.append(f"- **correctness (primary): {c.get('score', 0):.4f}** "
                     f"({'accepted' if c.get('accepted') else 'below 0.90 gate'})")
        for k, v in c.get("suites", {}).items():
            L.append(f"  - {k}: {v:.4f}")
        err = c.get("details", {}).get("error")
        if err:
            L.append(f"- harness error: `{err}`")
        if rep.get("harness_error"):
            L.append(f"- harness failed: `{rep['harness_error']}`")
        p = rep.get("perf", {})
        if p.get("error"):
            L.append(f"- performance: FAILED (`{p['error']}`)")
        elif p.get("speedup_geomean") is not None:
            suffix = (" (illustrative subset)"
                      if rep.get("perf_mode") == "illustrative" else "")
            L.append(f"- performance: geomean speedup vs llama.cpp Vulkan = "
                     f"{p['speedup_geomean']:.3f}{suffix}")
            L += ["", "| test | agent t/s | reference t/s | speedup |", "|---|---:|---:|---:|"]
            for k, v in p.get("rows", {}).items():
                L.append(f"| {k} | {v['agent']:.2f} | {v['ref']:.2f} | {v['speedup']:.3f} |")
            if rep.get("perf_mode") == "illustrative":
                cw = rep.get("correctness_wall_s", 0)
                th = rep.get("perf_threshold_s", 0)
                reason = ("forced by --perf-short" if cw <= th else
                          f"correctness took {cw:.0f}s > {th:.0f}s")
                L.append("")
                L.append(f"- perf mode: illustrative subset ({reason}); "
                         f"the full 5-rep pp/tg run was skipped.")
        v = rep.get("vram", {})
        if v:
            L.append(f"- VRAM: agent peak {v.get('agent_peak_bytes', 0)/2**20:.1f} MiB, "
                     f"reference {v.get('ref_peak_bytes', 0)/2**20:.1f} MiB, "
                     f"gate {'PASS' if v.get('passed') else 'FAIL'}")
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-perf", action="store_true")
    ap.add_argument("--perf-short", action="store_true",
                    help="force the short illustrative perf subset")
    ap.add_argument("--perf-threshold", type=float, default=900.0,
                    help="correctness wall seconds above which perf is illustrative")
    ap.add_argument("--no-container", action="store_true")
    ap.add_argument("--library", help="prebuilt library (debug/calibration)")
    args = ap.parse_args()

    sub = os.path.abspath(args.submission)
    out = os.path.abspath(args.out or sub + "-report")
    work = tempfile.mkdtemp(prefix="bonsai-verify-")
    rep = {"submission": sub, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    size = submission_size(sub)
    rep["submission_bytes"] = size
    if size > MAX_SUBMISSION_BYTES:
        rep["build_ok"] = False
        rep["build_log"] = (f"submission is {size} bytes, over the "
                            f"{MAX_SUBMISSION_BYTES}-byte source limit; embedding "
                            f"model weights or generated data is not allowed")
        write_report(rep, out)
        log(f"submission too large; report in {out}")
        return 1

    ensure_golden()
    if args.library:
        lib, err = os.path.abspath(args.library), None
    else:
        lib, err = build_submission(sub, work, args.no_container)
    rep["build_ok"] = lib is not None
    rep["build_log"] = err or ""
    if lib is None:
        write_report(rep, out)
        log(f"build failed; report in {out}")
        return 1

    harness_json = os.path.join(work, "agent_result.json")
    harness_error = None
    try:
        run_harness(lib, harness_json, args.no_container, args.skip_perf,
                    args.perf_short, args.perf_threshold)
    except SystemExit as e:
        harness_error = str(e)
        rep["harness_error"] = harness_error

    # The harness writes agent_result.json after every suite, so a crash still
    # leaves the scores and subtotal of the finished suites behind.
    res = None
    if os.path.exists(harness_json):
        try:
            res = json.load(open(harness_json))
        except Exception as e:  # noqa: BLE001
            rep["harness_error"] = (rep.get("harness_error") or "") + \
                f"; cannot read partial result: {e!r}"
    if res is not None:
        rep["correctness"] = res.get("correctness") or {}
        rep["correctness_current"] = res.get("correctness_current")
        rep["perf_mode"] = res.get("perf_mode")
        rep["correctness_wall_s"] = res.get("correctness_wall_s")
        rep["perf_threshold_s"] = res.get("perf_threshold_s")
        rep["agent_vram_peak_bytes"] = res.get("vram_peak_bytes")
        rep["agent_host_rss_bytes"] = res.get("host_peak_rss_bytes")
        rep["child_processes"] = res.get("child_processes")
    if harness_error is not None:
        write_report(rep, out)
        return 1
    if res is not None and (res.get("correctness") or {}).get("partial"):
        rep.setdefault("harness_error",
                       "harness stopped before all correctness suites completed")
        write_report(rep, out)
        return 1
    if res is None:
        rep.setdefault("harness_error", "harness produced no result")
        write_report(rep, out)
        return 1

    if not args.skip_perf and "perf" not in res:
        rep["perf"] = {"rows": {}, "speedup_geomean": 0.0,
                       "error": res.get("perf_error", "perf run failed")}
    if not args.skip_perf and "perf" in res:
        try:
            ref = measure_reference(args.no_container)
        except Exception as e:  # noqa: BLE001
            rep["perf"] = {"rows": {}, "speedup_geomean": 0.0,
                           "error": f"reference measurement failed: {e!r}"}
            write_report(rep, out)
            log(f"reference measurement failed; report in {out}")
            return 1
        rows = {}
        for kind in ("pp", "tg"):
            for k, av in res.get("perf", {}).get(kind, {}).items():
                rv = ref[kind].get(k)
                if rv:
                    label = f"pp{k}" if kind == "pp" else f"tg128 @ d{k}"
                    rows[label] = {"agent": av, "ref": rv, "speedup": av / rv}
        speedups = [r["speedup"] for r in rows.values()]
        rep["perf"] = {"rows": rows, "speedup_geomean": geom(speedups),
                       "reference": ref}

        ref_peak = ref.get("vram_peak_bytes", 0)
        ref_rss = ref.get("host_peak_rss_bytes", 0)
        cap = max(1.5 * ref_peak, ref_peak + 512 * 2**20)
        rss_cap = ref_rss + 2 * 2**30
        ap_peak = res.get("vram_peak_bytes", 0)
        ok = ap_peak <= cap and ap_peak >= 450 * 2**20 and res.get("host_peak_rss_bytes", 0) <= rss_cap
        rep["vram"] = {"agent_peak_bytes": ap_peak, "ref_peak_bytes": ref_peak,
                       "cap_bytes": cap, "agent_host_rss_bytes": res.get("host_peak_rss_bytes"),
                       "rss_cap_bytes": rss_cap, "passed": ok}
        if not ok:
            rep["perf"]["speedup_geomean"] = 0.0

    write_report(rep, out)
    log(f"report in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
