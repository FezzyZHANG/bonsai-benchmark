# Credential-injecting API proxy

This is the user's `anthropic-proxy.ts` / `anthropic-proxy.sh`, vendored with
benchmark hardening:

- runs in a separate container on the internal podman network; the agent
  container has no other route
- only forwards `POST /v1/messages`, `POST /v1/messages/count_tokens` and
  `GET /v1/models` by default (`ALLOWED_PATHS`, `ALLOWED_METHODS`)
- optional `MAX_REQUESTS` budget (`0` = unlimited, the script default; the run
  script sets 10000)
- `LOG_BODIES=0` disables request/response body logging (default on in the
  script; the run script sets it to 0). Authorization headers are replaced by
  `<redacted>` in logs either way.
- `AUTH_HEADER=authorization|x-api-key|both` selects the upstream auth header
  (default `authorization`, which is what Claude Code/DeepSeek use)
- `GET /__proxy_health` is a readiness endpoint and does not consume budget
- `RETRY_RATE_LIMIT_422` (default 6) re-issues a request whose upstream 422
  body reports a per-minute token limit (`Rate limit reached ... tokens per
  min (TPM) ... try again in N s`) after the suggested delay. Each wait is
  capped by `RETRY_RATE_LIMIT_MAX_WAIT_S` (default 20 s); `0` disables. Some
  gateways (cc-switch) report an upstream TPM limit as 422 instead of 429, and
  Claude Code does not retry 422, so the run dies without this. A 422 that is
  not a token rate limit passes through unchanged. Retries are logged as
  `event: "retry"` records and the final `response` record carries `retries`.
- `deno run --allow-all /proxy/test_retry.ts` inside the proxy image covers the
  retry, pass-through and bounded-retry paths against a mock upstream.

The upstream credential is injected by the proxy and never enters the agent
container. Claude Code inside the agent container uses
`ANTHROPIC_BASE_URL=http://<proxy-ip>:8787` and a dummy `ANTHROPIC_AUTH_TOKEN`.

The launcher derives Deno's `--allow-net` from `HOST:PORT` and the upstream
host/port, so the proxy itself cannot be used to reach arbitrary hosts.

Logs are JSONL on stdout, captured to `RUNS_ROOT/RUN_NAME.proxy.jsonl`.
