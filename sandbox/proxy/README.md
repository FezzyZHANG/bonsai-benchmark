# Credential-injecting API proxy

This is the user's `anthropic-proxy.ts` / `anthropic-proxy.sh`, vendored with
benchmark hardening:

- runs in a separate container on the internal podman network; the agent
  container has no other route
- only forwards `POST /v1/messages`, `POST /v1/messages/count_tokens` and
  `GET /v1/models` by default (`ALLOWED_PATHS`, `ALLOWED_METHODS`)
- optional `MAX_REQUESTS` budget (default 2000)
- `LOG_BODIES=0` disables request/response body logging (default on in the
  script; the run script sets it to 0)
- `AUTH_HEADER=authorization|x-api-key|both` selects the upstream auth header
  (default `authorization`, which is what Claude Code/DeepSeek use)
- `GET /__proxy_health` is a readiness endpoint and does not consume budget

The upstream credential is injected by the proxy and never enters the agent
container. Claude Code inside the agent container uses
`ANTHROPIC_BASE_URL=http://<proxy-ip>:8787` and a dummy `ANTHROPIC_AUTH_TOKEN`.

The launcher derives Deno's `--allow-net` from `HOST:PORT` and the upstream
host/port, so the proxy itself cannot be used to reach arbitrary hosts.

Logs are JSONL on stdout, captured to `RUNS_ROOT/RUN_NAME/proxy.jsonl`.
