#!/usr/bin/env python3
"""Report local development and official GPU evaluation prerequisites."""
import argparse
import glob
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

BENCH = Path(__file__).resolve().parent.parent


def probe(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return p.returncode == 0, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)


def inspect(require_gpu=False):
    checks = []

    def add(name, ok, detail, required=True):
        checks.append(dict(name=name, ok=bool(ok), detail=detail, required=required))

    add("host", True, f"{platform.system()} {platform.machine()}")
    for tool in ("python3", "git", "codex", "deno"):
        path = shutil.which(tool)
        ok, version = probe([tool, "--version"]) if path else (False, "not installed")
        add(tool, ok, version.splitlines()[0] if version else str(path),
            not require_gpu or tool in ("python3", "git"))
    ok, detail = probe([str(BENCH / "deps/python/bin/python"), "-c",
                        "import numpy,safetensors,yaml; print('numpy/safetensors/yaml available')"])
    add("python-data", ok, detail)
    linux = platform.system() == "Linux"
    add("grading-host", linux and platform.machine() == "x86_64",
        "requires Linux x86_64; macOS cannot run the official RADV /proc VRAM grader", require_gpu)
    devices = glob.glob("/dev/dri/renderD*")
    add("GPU-device", bool(devices), ", ".join(devices) or "no /dev/dri/renderD*", require_gpu)
    for tool in ("podman", "timeout"):
        add(tool, bool(shutil.which(tool)), shutil.which(tool) or "not installed", require_gpu)
    if shutil.which("podman"):
        ok, detail = probe(["podman", "info", "--format", "{{.Host.Arch}}"])
        add("podman-service", ok, detail, require_gpu)
        for image in (os.environ.get("AGENT_IMAGE", "bonsai-agent:1.0"),
                      os.environ.get("PROXY_IMAGE", "bonsai-proxy:1.0")):
            ok, _ = probe(["podman", "image", "exists", image])
            add("image", ok, image, require_gpu)
    for relative in (
        "deps/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf",
        "deps/data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors",
        "deps/docs/vkspec.html", "deps/docs/device/vulkaninfo.json",
        "deps/ref/bin/bonsai-ref", "deps/ref/bin/bonsai-ref-cpu",
    ):
        path = BENCH / relative
        add("artifact", path.is_file() and path.stat().st_size > 0, relative, require_gpu)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true",
                        help="fail unless the official grading prerequisites are present")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    checks = inspect(args.require_gpu)
    ready = all(c["ok"] for c in checks if c["required"])
    if args.json:
        print(json.dumps(dict(ready=ready, scope="gpu" if args.require_gpu else "local-development",
                              checks=checks), indent=2))
    else:
        for c in checks:
            status = "PASS" if c["ok"] else "FAIL" if c["required"] else "MISSING"
            print(f"{status:7} {c['name']}: {c['detail']}")
        print("Official grading also requires the compute features listed in agent/DEVICE.md.")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
