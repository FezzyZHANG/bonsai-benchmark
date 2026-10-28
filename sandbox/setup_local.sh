#!/bin/bash
# Install the Python data tools used for runner development on Linux/macOS.
# Does not claim to provision the Linux/RADV benchmark on a Mac.
set -euo pipefail
BENCH="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BENCH"
for tool in python3 git deno; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done
[ -x deps/python/bin/python ] || python3 -m venv deps/python
deps/python/bin/python -m pip install -r sandbox/requirements-local.txt
deps/python/bin/python -c 'import numpy, safetensors, yaml; print("Python data environment ready")'
python3 -m unittest discover -s sandbox/tests -v
deno run --allow-all sandbox/proxy/test_retry.ts
if [ -f sandbox/smoke_codex.py ]; then
  command -v codex >/dev/null || { echo 'Codex adapter present; install Codex CLI to run its smoke test' >&2; exit 1; }
  python3 sandbox/doctor.py
  python3 sandbox/smoke_codex.py
else
  echo 'Local environment ready; Codex integration checks run when its adapter is installed.'
fi
