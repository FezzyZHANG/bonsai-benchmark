# Run records

Raw Claude Code artifacts for the bonsai-vk benchmark runs, kept alongside
`reports/` and `submissions/`.

| directory | model | effort | compact window | report |
|---|---|---|---|---|
| `run_dsv41f_1017_high` | DeepSeek-V4.1-Flash-1017 (`deepseek-flash[1m]`) | high | 786432 | `reports/run_dsv41f_1017_high/` (verifier failed: greedy-long SIGSEGV; partial 0.6958) |

Each directory contains:

* `settings.json` — the generated Claude Code settings used for the run
  (sandbox, model and environment). No credentials are present; the agent
  container received only a dummy token and the host-side proxy held the real
  upstream credential.
* `config.json` — the run's model/effort/compact/context settings, launch
  environment (auth token redacted), and the runner's `meta.json` timing.
* `meta.json`, `tuning`, `stderr.log`, `proxy.jsonl` — the runner and sandbox
  proxy artifacts (start/end/exit, effective tuning, stderr, proxy requests).
* `transcript/` — the session JSONL as saved under `~/.claude/projects/`,
  including the two subagent transcripts and persisted tool results where
  present. The main transcript is `<session-uuid>.jsonl` at its root;
  `stream.jsonl` is the runner's stream-json capture of the same session.
* `AUDIT.md` — transcript and source audit for gaming/compliance behaviors.

The transcripts are the agent sessions themselves and predate verification; see
`reports/` for the scores. Redaction was verified by scanning the copied tree
for `sk-ant-`/`sk-` API keys, `Bearer` tokens, and the upstream credential:
none were found. Proxy authorization headers are `<redacted>`. The only
token-like strings are the sandbox's deliberate `benchmark-dummy-token` (the
agent container never held the upstream credential) and the container-local
`CLAUDE_CODE_MESSAGING_TOKEN` (ephemeral IPC, not an upstream credential).

`submissions/<run>/` holds a snapshot of the run workspace at the deadline:
source, `build.zig`, `NOTES.md`, `tools/`, and the built `zig-out/lib/libbonsai.so`.
It excludes `.claude-home` (the transcript above), `.git` (the workspace's
initial template commit), and `.zig-cache` (intermediate Zig build cache).
