# macOS local development

The local development workflow supports macOS's system Bash 3.2 and BSD
utilities. Official GPU evaluation still targets Linux x86_64, RADV and the
device features in `agent/DEVICE.md`; this change does not port Vulkan scoring
or `/proc` VRAM accounting to macOS.

## Install and check

Install Python 3.10+, Git and Deno, then run:

```bash
sandbox/setup_local.sh
```

The script creates `deps/python`, installs NumPy, safetensors and PyYAML,
runs the portability tests and exercises the proxy against a local mock
upstream. It does not download the multi-gigabyte model, build Linux images,
use a real API key or start a benchmark run. Tests need loopback socket access.

If the Codex adapter is also installed, this same command requires the Codex
CLI and additionally runs its regression suite, environment checks and local
mock integration test. Install the CLI version pinned by the adapter. The
mock integration can then be repeated directly:

```bash
python3 sandbox/smoke_codex.py
```

Codex integration reports are saved under `sandbox/generated/local-smoke/`.
The Mac bootstrap also works independently with the original Claude runner;
it does not require the Codex adapter.

## Prepare an offline workspace

Choose a run directory outside the benchmark checkout. Workspace preparation
uses Python copying instead of GNU-only `cp` options, preserves the supplied
run root in the printed launch command, and rejects path traversal and
symlinks back into the operator repository:

```bash
sandbox/prepare_run.sh local01 /path/outside/benchmark/bonsai-runs
```

This prepares files only. The Linux runner and formal grader still need their
GPU environment.

## Optional artifact prefetch

```bash
./setup.sh --prepare-only
```

This fetches and verifies the original Linux toolchain archives, downloads
the specs and model, and exports safetensors. It skips reference builds,
container images and device capture. Partial toolchain/model downloads can
resume on the next run. Downloaded Linux executables cannot run natively on
macOS. Calling the full `setup.sh` on macOS fails before any download and
points to the local-development and prefetch alternatives.

Docker Desktop does not expose the Mac GPU as this benchmark's Linux RADV
`/dev/dri` device. Local mock checks validate runner behavior, not GPU
correctness, performance, VRAM usage or container isolation.
