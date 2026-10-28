# Codex runner

## Official evaluation on Linux

The benchmark still targets Linux x86_64, AMD RX 9070 / RADV, Vulkan 1.4 and
the compute features in `agent/DEVICE.md`. The Codex adapter does not change
the model, ABI, acceptance thresholds, hidden tests, VRAM sampler or scoring.

1. Install Podman, Python 3, Git, curl, xz, CMake, a C++ compiler and GNU
   coreutils. Configure access to `/dev/dri` for the user running Podman.
2. Clone `https://github.com/PrismML-Eng/llama.cpp`, branch `prism`, outside
   the agent workspace. Set `LLAMA_SRC` to that checkout.
3. Run `./setup.sh` to fetch the pinned Zig/Vulkan toolchains, docs and data,
   build reference binaries and build both images. Rebuild existing images
   with `sandbox/build_images.sh` after pulling this adapter, so the agent
   image contains `@openai/codex@0.162.1`.
4. Run `python3 sandbox/doctor.py --require-gpu`. This checks prerequisite
   presence; confirm the device feature set with `vulkaninfo` against
   `agent/DEVICE.md` as well.
5. Export `OPENAI_API_KEY`, `OPENAI_BASE_URL` (default
   `https://api.openai.com/v1`) and `BENCH_MODEL` in the operator shell.
6. Prepare and run:

   ```bash
   export BENCH_AGENT=codex
   export RUNS_ROOT=/mnt/workspace/bonsai-runs
   sandbox/prepare_run.sh codex01 "$RUNS_ROOT"
   sandbox/run_agent.sh codex01 "$RUNS_ROOT"
   sandbox/monitor_log "$RUNS_ROOT/codex01.transcript.jsonl"
   ```

7. Grade the saved workspace with the unchanged verifier:

   ```bash
   python3 verification/verify.py "$RUNS_ROOT/codex01" --out "$RUNS_ROOT/codex01-report"
   ```

Use a fresh name for every run. Claude Code remains available with
`BENCH_AGENT=claude` or with `BENCH_AGENT` unset. Prepare and run must use the
same agent; preparation removes Claude's command-classifier workaround from
the Codex prompt without changing the benchmark task.

## Configuration and logs

| Operator setting | Codex behavior |
| --- | --- |
| `BENCH_MODEL` | Required model ID, recorded in the run configuration |
| `BENCH_EFFORT` | `model_reasoning_effort`; use a level supported by the selected model |
| `BENCH_COMPACT_WINDOW` | `model_auto_compact_token_limit`, positive token count |
| `BENCH_MAX_CONTEXT` | `model_context_window`, positive token count |
| `RUN_TIMEOUT` | Wall-clock budget in seconds; default 7200 |
| `MAX_REQUESTS` | Proxy request budget; default 10000 |
| `OPENAI_BASE_URL` | Full upstream API prefix, including `/v1` if required |
| `OPENAI_API_KEY` | Upstream bearer token; passed only to the proxy |

Codex runs a single agent. Claude-specific `BENCH_MAX_OUTPUT`,
`BENCH_MAX_TURNS`, subagent knobs and `BENCH_SIMPLE_SYSTEM_PROMPT` are not
mapped to unrelated Codex settings and cause a configuration error if set.

The proxy forwards only `POST /responses` and `POST /responses/compact`.
For example, internal `/responses` maps to
`https://api.openai.com/v1/responses`; `/v1` is not duplicated. A gateway URL
such as `https://gateway.example/openai/v1` works the same way. SSE is streamed
unchanged. Codex uses HTTP Responses streaming, with WebSockets and request
compression disabled for gateway compatibility. API model discovery is not
required: the operator supplies the model ID.

`NAME.transcript.jsonl` contains Codex events; `NAME.stderr.log` contains CLI
diagnostics; `NAME.proxy.jsonl` contains redacted proxy logs; `NAME.tuning`
contains the effective TOML (no credential); `NAME.meta.json` records the
agent, start/end/deadline, exit status and timeout. The host's Codex login,
configuration, MCP servers and plugins are not mounted into the container.

Full filesystem access is configured **inside the Podman container only**.
The agent has no internet route and the data/docs/ref mounts remain read-only.
The local smoke test explicitly overrides this with `workspace-write`.

## Local mock verification

The proxy launcher requires Bash 4+ (as supplied by the Linux benchmark
image). Install Codex CLI 0.162.1, Deno, Git and Python 3.10+ to run the mock
checks on a development host:

```bash
python3 -m unittest discover -s sandbox/tests -p 'test_codex.py' -v
python3 sandbox/smoke_codex.py
```

The smoke test starts a real Codex CLI against a local mock Responses server
through the credential proxy. No real model API or paid request is involved.
Tests require permission to bind loopback sockets. Results are saved under
`sandbox/generated/local-smoke/`. The mock asks Codex to run a harmless shell
command and write a temporary workspace file; checks cover tool execution,
final output, event logging, credential separation and absence of hosted web
search. It explicitly selects `workspace-write` on the host.

The regression suite uses a recording Podman replacement to exercise runner
arguments, exit/timeout metadata and read-only mounts. These checks do not
prove kernel/container isolation or GPU correctness. A real Linux GPU run is
still required to validate the complete benchmark. `run_codex.sh` rejects
macOS for official evaluation; local mock verification does not grade a model.

CLI behavior follows the official [non-interactive mode documentation](https://learn.chatgpt.com/docs/non-interactive-mode)
and [configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).
