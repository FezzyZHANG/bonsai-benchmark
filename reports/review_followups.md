# Review follow-ups (post-final-check)

Review date: 2026-10-27. This file records the issues from the final
pre-run review that remain open. The items fixed so far are listed at the end
for context.

## Correctness impact

None of the remaining items changes the correctness *scoring logic* for an
honest, ABI-compliant Vulkan submission. The correctness suites, weights,
golden data, dequant exactness, teacher-forced greedy and reference measurement
were checked and calibrated (0.9951/0.9952 controls).

The remaining items can still affect a run in these ways:

- **Benchmark validity** if the agent cheats or uses the proxy as a network /
  model escape (golden exposure, proxy body, link/Vulkan checks, and the
  accepted Bash-script workaround).
- **Whether a run completes** (resource limits and the portability items
  below).
- **Performance only** (perf measured on a failed-correctness submission, and
  reference/container comparability).
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
- [ ] **Proxy method/path allowlist is a cross-product.**
  `sandbox/proxy/anthropic-proxy.ts` checks `allowedMethods.has(method)` and
  `allowedPaths.has(path)` independently, so `GET /v1/messages` and
  `POST /v1/models` are also allowed, contrary to `sandbox/proxy/README.md`.
  Use `(method, path)` pairs.
- [ ] **Minor: ABI version / metadata / determinism not validated.** The
  harness stores `bonsai_abi_version`, `n_vocab`, `n_ctx_train` and
  `tensor_count` but does not assert them; determinism is not explicitly tested
  (only one run per case).

### Accepted risk (well-behaved-agent assumption)

- **Bash script workaround.** The agent prompt tells the model to put a denied
  command into a workspace script and run the script; the smoke test uses the
  same workaround for its network and Python checks. This is known to let some
  commands past Claude Code's Bash command classifier and is the same class as
  the open items above (golden mount, raw proxy access, missing link/Vulkan
  checks). Accepted because the podman container and proxy are the actual
  boundary and the benchmark assumes a well-behaved agent. Do not treat scores
  from an adversarial agent as trustworthy.

## B. Operational robustness

- [ ] **No container resource limits.** Neither the agent nor the grading
  container sets `--memory`, `--cpus` or `--pids-limit`. Consider limits or
  document that the machine is dedicated.
- [ ] **Low: environment/reproducibility.** `setup.sh` fetches docs from
  `latest`/`main`; the Containerfile's apt packages are unpinned;
  `measure_reference(no_container)` ignores its parameter and always measures
  the reference on the host (calibration is ~1.0, so this is fine, but the
  parameter is misleading). `run_agent.sh` does not pass a `--max-turns` CLI
  flag if `BENCH_MAX_TURNS` is set (it only sets the env var).
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
## D. Prompting / documentation nits

- [x] **`MODEL.md` L2-norm form.** Moot: the architecture description
  (including the GDN L2-norm formula) was removed when `MODEL.md` was reduced
  to a reverse-engineering pointer.
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

### Remaining hint sources (not removed)

- **GGUF `prism.hadamard.*` metadata.** The GGUF itself names
  `normalized-sylvester-walsh-hadamard`, gives the block size, axis, sign mode,
  sign widths/values, the folded weight list, `inverse_weight_names`,
  `tied_output` and `gdn_v_grouped`. `agent/MODEL.md` no longer expands these,
  but the keys are visible. The reference loader requires them
  (`llama-model.cpp:1207-1351`), so removing them would break the oracle unless
  the fork is patched and the golden regenerated.
- **GGUF tensor names and shapes.** `attn_qkv`, `attn_gate`, `ssm_conv1d`,
  `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_norm`, `ssm_out`, etc. reveal the
  hybrid architecture and the full/full-attention schedule without renaming
  tensors (which would also break the reference).
- **Reference binary strings.** `strings deps/ref/bin/bonsai-ref` still shows
  `GATED_DELTA_NET`, `gated_delta_net(q, k, v, g, beta, s)`,
  `blk.%d.ssm_*`, `prism.hadamard`, and the fused GDN kernel names. Symbol
  stripping does not remove string literals. If the oracle is meant to be
  behavior-only, the binaries need an error/op-string-free build; otherwise
  `strings` is an easy shortcut. `llama-debug` may leak more.
- **`gguf-py` constants.** `deps/gguf-py/gguf/constants.py` names `PTQ1_0`
  and gives its block layout `(128, 2 + 24 + 2)`. Removing those entries (or
  shipping a reader that reports the type numerically) would make the PTQ1_0
  decoder more of a reverse-engineering task; the reader would still expose
  the raw bytes.
- **`agent/TOKENIZER.md`.** Gives the exact `qwen35` pre-tokenizer regex,
  `clean_spaces = false`, the byte table and the BPE algorithm. This is the
  largest remaining hand-holding for a scored suite (10%). It may be
  intentional (tokenizer is not the novel architecture); trim it to a pointer
  if the tokenizer should also be reverse-engineered.
- **Model/file names** contain `Ternary` and `PTQ1_0`, and `general.name` in
  the GGUF says `Ternary-Bonsai-2-4B`. The quantization family is therefore
  visible even with all metadata stripped.

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
- [x] `verification/harness/agent_harness.py`: clamp `n_batch` to `n_ctx`
  (`Engine.ctx`), so the `tg @ d0` perf context (257) no longer passes a 512
  hint.
- [x] `agent/MODEL.md`: reduced to an artifact/oracle pointer. The architecture,
  layouts, fold, recurrence and numerical formulas are intentionally
  undocumented so the model must be reverse-engineered.
- [x] `deps/data/*-f16.safetensors`: stripped the visible `__metadata__`
  (quantization name, group size, trit/f16 note). `setup.sh` now strips it for
  linked exports too, and `gguf_to_safetensors.py` no longer writes it.
- [x] `setup.sh`: dependency install is idempotent (an existing `deps/python`
  venv missing `pyyaml` is repaired; this was not caught by the agent smoke
  test because it checks the agent image, which always installs the packages).
- [x] `sandbox/mock_anthropic.py`: the smoke test's Python step now imports
  `yaml` and opens the safetensors file, asserting 426 tensors and no
  `__metadata__`, so a missing data dependency or unstripped header fails the
  sandbox check.
- [x] `deps/data/samples/`: the visible corpus/short prompt were the Prism
  llama.cpp README, which names `GATED_DELTA_NET`, recurrent/delta-net layers,
  state size 128, head size 256, value heads and PTQ1_0. They are now neutral
  GGUF/Vulkan documentation; `setup.sh` no longer copies the investigation
  corpus/p2k text.
- [x] `verification/verify.py`, `verification/tools/ref_client.py`: catch
  build/harness/golden-generation timeouts and reference-measurement failures;
  give `RefClient` a per-command timeout and clean up a timed-out build/harness
  container by name.
- [x] `verification/verify.py`: pass `--security-opt label=disable` to the
  build container, matching the harness and agent containers.
- [x] `setup.sh`: install PyYAML in `deps/python` (the gguf-py reader imports
  it), in both the venv and uv fallback paths.
- [x] `verification/verify.py`: copy symlinks as symlinks and skip
  sockets/FIFOs/devices, so a directory symlink cannot pull an excluded tree or
  a weight file into the build and the size check and copied tree agree.
- [x] `sandbox/mock_anthropic.py`, `sandbox/smoke_test.sh`: the smoke test now
  issues individual simple Bash commands (plus the workspace-script workaround
  for the classifier's `curl`/`python3 -c` false negatives) and checks no
  egress, WebSearch denial, credential isolation, `data/docs/ref` read-only,
  the GPU, Write/Read, and the Zig/`glslc`/`glslangValidator`/`slangc`/
  `spirv-val`/`gguf-py`/CPU-reference toolchain. It passes end to end against
  the mock (27 proxy requests). The script workaround is an accepted risk
  (section A).
- [x] `agent/PROMPT.md`, `agent/TASK.md`: tell the agent to keep Bash commands
  simple, avoid environment variables pointing outside the workspace, and use
  a workspace script if a safe command is still denied by the classifier.
