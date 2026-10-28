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
MOCK_PORT="${MOCK_PORT:-8000}"
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
  -e "MOCK_IP=$MOCK_IP" -e "MOCK_PORT=$MOCK_PORT" \
  "$AGENT_IMAGE" python3 -I /mock.py "$MOCK_PORT" >/dev/null

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
check("tool coverage", {"Bash", "Write", "Read", "WebFetch", "WebSearch"} <= set(tools))
bash_results = [r for name, r in zip(tools, results) if name == "Bash"]
write_results = [r for name, r in zip(tools, results) if name == "Write"]
read_results = [r for name, r in zip(tools, results) if name == "Read"]


def res(items, i):
    return items[i] if len(items) > i else (True, "")


check("GPU visible", "Radeon" in res(bash_results, 5)[1] and
      "radv" in res(bash_results, 5)[1])
check("Write/Read allowed", len(write_results) >= 7 and
      all(not r[0] for r in write_results[:7]) and
      read_results and not read_results[0][0] and
      "written-by-mock" in read_results[0][1])
check("WebFetch denied", any(r[0] and "WebFetch" in r[1] for r in results))
check("WebSearch denied", any(r[0] and "WebSearch" in r[1] for r in results))
check("final text", any("DONE mock" in t for t in texts))
check("toolchain", all([
    not res(bash_results, 0)[0], "0.17.0" in res(bash_results, 0)[1],
    not res(bash_results, 1)[0], "shaderc" in res(bash_results, 1)[1].lower(),
    not res(bash_results, 2)[0], "glslang" in res(bash_results, 2)[1].lower(),
    not res(bash_results, 3)[0], "slang" in res(bash_results, 3)[1].lower(),
    not res(bash_results, 4)[0], "spirv" in res(bash_results, 4)[1].lower(),
    not res(bash_results, 6)[0],      # glslc smoke.comp -o smoke.spv
    not res(bash_results, 7)[0],      # spirv-val smoke.spv
    "py-ok" in res(bash_results, 8)[1],  # python script
    "arch=qwen35" in res(bash_results, 9)[1],
    not res(bash_results, 11)[0],     # zig build
]))
check("no egress", not res(bash_results, 12)[0] and
      "egress-blocked" in res(bash_results, 12)[1] and
      "egress-blocked-egress-net" in res(bash_results, 12)[1] and
      "egress-OPEN" not in res(bash_results, 12)[1])
check("data/docs/ref read-only", len(write_results) >= 10 and
      all(r[0] for r in write_results[7:10]))
check("credential isolation",
      "benchmark-dummy-token" in res(bash_results, 10)[1] and
      "mock-upstream-secret" not in res(bash_results, 10)[1])
sys.exit(0 if ok else 1)
PY
echo "mock requests:"
podman logs "$MOCK_NAME" 2>&1 | grep -c "auth=" || true
