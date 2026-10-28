#!/bin/bash
# End-to-end sandbox smoke test against a local mock Messages API.
#
#   sandbox/smoke_test.sh [RUN_NAME]
#
# Checks that Claude Code in the agent container can run tools, sees the GPU,
# can write in the workspace, and that WebFetch is denied. No real API
# credential is used: the proxy forwards to sandbox/mock_anthropic.py.
set -euo pipefail
BENCH="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
NAME="${1:-smoke-$(date +%s)}"
ROOT="${RUNS_ROOT:-/mnt/workspace/bonsai-runs}"
EGRESS_NET="${EGRESS_NET:-bonsai-egress}"
MOCK_IP="${MOCK_IP:-10.90.0.4}"
MOCK_NAME="bonsai-mock-$NAME"
AGENT_IMAGE="${AGENT_IMAGE:-bonsai-agent:1.0}"

podman image exists "$AGENT_IMAGE" || { echo "missing image $AGENT_IMAGE; run setup.sh" >&2; exit 1; }

if ! podman network exists "$EGRESS_NET"; then
  podman network create --subnet 10.90.0.0/24 "$EGRESS_NET" >/dev/null
fi
subnet=$(podman network inspect -f '{{range .Subnets}}{{.Subnet}} {{end}}' "$EGRESS_NET")
case " $subnet " in
  *" 10.90.0.0/24 "*) ;;
  *) echo "$EGRESS_NET is not 10.90.0.0/24 ($subnet); set MOCK_IP or recreate it" >&2; exit 1;;
esac

cleanup() { podman rm -f "$MOCK_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

podman run -d --name "$MOCK_NAME" --network "$EGRESS_NET:ip=$MOCK_IP" \
  -v "$BENCH/sandbox/mock_anthropic.py:/mock.py:ro" -e PYTHONUNBUFFERED=1 \
  "$AGENT_IMAGE" python3 -I /mock.py 8000 >/dev/null

"$BENCH/sandbox/prepare_run.sh" "$NAME" "$ROOT"
ANTHROPIC_BASE_URL="http://$MOCK_IP:8000" ANTHROPIC_AUTH_TOKEN=mock-upstream-secret \
  BENCH_MODEL=mock-model RUN_TIMEOUT="${RUN_TIMEOUT:-300}" \
  "$BENCH/sandbox/run_agent.sh" "$NAME" "$ROOT"

deps="$BENCH/deps/python/bin/python"
[ -x "$deps" ] || deps=python3
t="$ROOT/$NAME.transcript.jsonl"
"$deps" -I - "$t" <<'PY'
import json, sys
tools, texts, results = [], [], []
for line in open(sys.argv[1]):
    try:
        e = json.loads(line)
    except Exception:
        continue
    m = e.get("message", {})
    if e.get("type") == "assistant":
        for c in m.get("content", []):
            if c.get("type") == "tool_use":
                tools.append(c.get("name"))
            elif c.get("type") == "text":
                texts.append(c.get("text", ""))
    elif e.get("type") == "user":
        for c in m.get("content", []):
            if isinstance(c, dict) and c.get("type") == "tool_result":
                results.append((c.get("is_error", False), json.dumps(c.get("content"))))
ok = True
def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and cond
check("tool sequence", tools[:4] == ["Bash", "Write", "Read", "WebFetch"])
check("GPU visible", any("Radeon" in r[1] and "radv" in r[1] for r in results))
check("Write/Read allowed", any("written-by-mock" in r[1] for r in results) and
      any("File created successfully" in r[1] for r in results))
check("WebFetch denied", any(r[0] and "WebFetch" in r[1] for r in results))
check("final text", any("DONE mock" in t for t in texts))
sys.exit(0 if ok else 1)
PY
echo "mock requests:"
podman logs "$MOCK_NAME" 2>&1 | grep -c "auth=" || true
