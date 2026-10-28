# Review follow-ups (post-final-check)

Review date: 2026-10-27. This file records the issues from the final
pre-run review that were **not** fixed in the same pass. The four fixed items
are listed at the end for context.

## Correctness impact

None of the remaining items changes the correctness *scoring logic* for an
honest, ABI-compliant Vulkan submission. The correctness suites, weights,
golden data, dequant exactness, teacher-forced greedy and reference measurement
were checked and calibrated (0.9951/0.9952 controls).

The remaining items can still affect a run in these ways:

- **Benchmark validity** if the agent cheats or uses the proxy as a network /
  model escape (golden exposure, proxy body, link/Vulkan checks).
- **Whether a run completes** (verifier timeout handling, SELinux build label,
  setup PyYAML on a fresh machine, resource limits).
- **Performance only** (`n_batch` edge at `n_ctx=257`; perf measured on a
  failed-correctness submission).
- **Acceptance semantics** for a non-compliant submission (VRAM gate is only a
  perf zero, not an `accepted=false`).
- **Tiny numeric/doc mismatches** (L2-norm eps form) that are far inside the
  acceptance tolerance.

## A. Benchmark integrity / anti-cheat

- [ ] **High: hidden golden is mounted into the library's container.**
  `verification/verify.py:run_harness` mounts `/golden` and `/harness`
  read-only into the same container that `dlopen`s the submission. The library
  can read `/golden/manifest.json`, the `.logits.bin`/`.npy` files and the
  greedy ids (and `/proc/self/cmdline` reveals the path) and replay them.
  Mitigation: preload all golden data in the harness before `CDLL`, mount a
  writable temporary copy and delete it before loading, or run the engine in a
  separate golden-free process. In-process memory remains readable, so process
  separation is the only complete fix.
- [ ] **Docs claim link checks the verifier does not implement.**
  `agent/ABI.md` says the grader checks that the library does not link
  llama.cpp/ggml; `agent/ACCEPTANCE.md` says "Linked libraries are checked."
  `verify.py` runs no `ldd`/`readelf`/`file` check. Add the check or remove the
  claims.
- [ ] **No check for a prebuilt engine binary, and no Vulkan-usage check.**
  A CPU engine that allocates a dummy >=450 MiB VRAM buffer passes correctness
  and the VRAM lower bound; only the performance score suffers. A prebuilt
  `.so` under the 32 MiB cap could be copied by `build.zig`. Decide whether
  these are honor-system (update the docs) or enforced.
- [ ] **The proxy forwards arbitrary Messages requests.** The agent's `Bash`
  can call `POST /v1/messages` with the dummy token; the proxy injects the real
  credential and does not inspect the body. Server-side tools such as
  `web_search` therefore work even though the Claude `WebSearch` tool is
  denied; the request can also name a different model than `BENCH_MODEL`, and a
  nested `claude --dangerously-skip-permissions` with `IS_SANDBOX=1` is
  possible. Mitigation: parse the body and reject server-side tool types,
  optionally pin/rewrite the model, and document the nested-session risk.
- [ ] **Source-size cap is bypassable with a directory symlink.**
  `submission_size` uses `os.walk` (does not follow directory symlinks), while
  `build_submission`'s `shutil.copytree` follows them. A symlink to `data/` can
  copy weights into the build uncounted. The VRAM gate still blocks
  pre-dequantization, but the size check is not a hard boundary. Use
  `symlinks=True` or realpath checks.
- [ ] **Proxy method/path allowlist is a cross-product.**
  `sandbox/proxy/anthropic-proxy.ts` checks `allowedMethods.has(method)` and
  `allowedPaths.has(path)` independently, so `GET /v1/messages` and
  `POST /v1/models` are also allowed, contrary to `sandbox/proxy/README.md`.
  Use `(method, path)` pairs.
- [ ] **Minor: ABI version / metadata / determinism not validated.** The
  harness stores `bonsai_abi_version`, `n_vocab`, `n_ctx_train` and
  `tensor_count` but does not assert them; determinism is not explicitly tested
  (only one run per case).

## B. Operational robustness

- [ ] **Verifier timeout failures are uncaught.** `subprocess.run(timeout=...)`
  in `build_submission`/`run_harness` can raise `TimeoutExpired`;
  `measure_reference` and `RefClient` have no timeout. A hung engine/oracle
  crashes the verifier without a report. Catch and report these, and give
  `RefClient` a timeout.
- [ ] **Build container lacks `--security-opt label=disable`.**
  `verify.py:build_submission` does not pass it while `run_harness` and
  `sandbox/run_agent.sh` do; on an SELinux host the build bind mounts may fail.
- [ ] **`setup.sh` fallback is missing PyYAML.** `deps/python` installs only
  `numpy safetensors`, but `verification/tools/gguf_to_safetensors.py` imports
  `gguf`, which imports `yaml`. A fresh setup without the investigation-tree
  safetensors fails in the fallback. Add `pyyaml` (and consider `tqdm` and
  `requests` from the gguf-py dependency list).
- [ ] **No container resource limits.** Neither the agent nor the grading
  container sets `--memory`, `--cpus` or `--pids-limit`. Consider limits or
  document that the machine is dedicated.
- [ ] **Smoke-test coverage does not match its claims.** `smoke_test.sh`
  checks tool order, GPU, Write/Read, WebFetch and final text, but not: no
  egress, WebSearch denial, credential isolation, `data/docs/ref` read-only,
  or that Zig/`glslc`/`glslangValidator`/`gguf-py`/the reference binaries work.
  Add a Bash step covering those.
- [ ] **Low: environment/reproducibility.** `setup.sh` fetches docs from
  `latest`/`main`; the Containerfile's apt packages are unpinned;
  `measure_reference(no_container)` ignores its parameter and always measures
  the reference on the host (calibration is ~1.0, so this is fine, but the
  parameter is misleading). The smoke test's `tools[:4]` sequence is brittle.
  `run_agent.sh` does not pass a `--max-turns` CLI flag if `BENCH_MAX_TURNS` is
  set (it only sets the env var).
- [ ] **Confirm the subagent tool is permitted (if subagents are intended).**
  The `dontAsk` allowlist lists Bash/Read/Glob/Grep/Edit/Write/NotebookEdit but
  not `Task`/`Agent`. If those tools require permission in 2.1.310, the default
  one-concurrent-subagent configuration is denied. This is a run-capability
  check, not correctness.

## C. Scoring / acceptance semantics

- [ ] **VRAM gate vs. `accepted`.** `ACCEPTANCE.md` says acceptance requires
  passing the VRAM gate, but `agent_harness.py` sets `accepted` from
  correctness only; the gate zeroes performance. Decide whether a VRAM failure
  should set `accepted=false` (pre-dequantization rejection) or stay a perf
  zero, and make the docs/code agree.
- [ ] **Perf is measured when correctness < 0.90.** `ACCEPTANCE.md` says
  performance is measured only after correctness acceptance; `verify.py` still
  runs the perf suites and reports a geomean. Zero or skip it for a rejected
  submission if that is the intent.
- [ ] **Perf edge: `n_batch` hint.** `agent_harness.py:Engine.ctx` passes
  `n_batch=512` even for the `tg @ d0` context of `n_ctx=257`. An engine that
  asserts `n_batch <= n_ctx` fails the perf run (correctness suites are all
  `n_ctx >= 512`, so this is perf-only). Pass `min(512, n_ctx)`.

## D. Prompting / documentation nits

- [ ] **`MODEL.md` L2-norm form.** The doc says
  `q / sqrt(sum(q^2) + eps)`; ggml uses `q / max(sqrt(sum(q^2)), eps)`. The
  difference is negligible for these norms and far inside tolerance, but align
  the doc with the oracle.
- [ ] **`MODEL.md` "Do not invert/rotate the weight matrix ... at load time."**
  Rotating weights at load is a valid, likely faster alternative to the
  reference's per-token activation fold and is not forbidden by acceptance.
  Remove the prohibition or state that both are acceptable and the reference
  uses activation-side.
- [ ] **README "pin the model through the proxy".** The proxy does not enforce
  `BENCH_MODEL`; `run_agent.sh` only exports Claude Code's model env. Say
  "sets Claude Code's model" unless proxy enforcement is added.
- [ ] **Hidden-corpus tracking wording.** `reports/calibration.md` says the
  hidden corpus and golden are gitignored; only `verification/tests/golden/` is
  ignored. `verification/tests/corpus.txt` and `tokenizer_cases.json` are
  tracked (still not mounted into agent runs). Fix the wording if desired.
- [ ] **Minor: `ACCEPTANCE.md` §3 "performance is only scored when the
  correctness suites pass"** and §5 "linked libraries are checked" do not match
  the implementation (covered above).

## Fixed in this pass

- [x] `agent/TASK.md`, `agent/ACCEPTANCE.md`: maximum tested context is
  **16384**, not 8192 (the `greedy_long` case and the VRAM reference use
  16384). The harness/golden are unchanged, so the calibration remains valid.
- [x] `agent/TASK.md`, `agent/ACCEPTANCE.md`: document that the grading build
  runs in a clean container with only the submitted files; `data/`, `docs/`,
  `ref/` are absent at build time; no hard-coded `/workspace` paths.
- [x] `sandbox/run_agent.sh`: the proxy log is written to
  `RUNS_ROOT/RUN_NAME.proxy.jsonl` (outside the graded workspace) instead of
  `RUNS_ROOT/RUN_NAME/proxy.jsonl`; README and proxy README updated.
- [x] `agent/ABI.md`, `agent/TOKENIZER.md`: `add_bos` does not prepend BOS for
  this model because `tokenizer.ggml.add_bos_token=false`; the reference and
  `TOKENIZER-VECTORS.json`'s `ids_add_bos` agree with `ids`. (The harness never
  tests `add_bos`; this was documentation-only.)
