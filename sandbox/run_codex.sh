#!/bin/bash
# Codex Responses API runner. The Podman container is the security boundary.
set -euo pipefail
BENCH="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:?usage: run_codex.sh RUN_NAME [RUNS_ROOT]}"
ROOT="${2:-${RUNS_ROOT:-/mnt/workspace/bonsai-runs}}"
case "$NAME" in ''|.|..|*[!a-zA-Z0-9_.-]*) echo 'invalid RUN_NAME' >&2; exit 1;; esac
ROOT="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$ROOT")"
RUN="$ROOT/$NAME"
[ -d "$RUN" ] || { echo "missing $RUN; run BENCH_AGENT=codex sandbox/prepare_run.sh first" >&2; exit 1; }
[ "$(cat "$ROOT/$NAME.agent" 2>/dev/null)" = codex ] || {
  echo 'workspace was not prepared for Codex; prepare a fresh run with BENCH_AGENT=codex' >&2; exit 1;
}
python3 "$BENCH/sandbox/doctor.py" --require-gpu
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (upstream credential)}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export OPENAI_API_KEY
AGENT_IMAGE="${AGENT_IMAGE:-bonsai-agent:1.0}"
PROXY_IMAGE="${PROXY_IMAGE:-bonsai-proxy:1.0}"
INT_NET="${INT_NET:-bonsai-int}"
EGRESS_NET="${EGRESS_NET:-bonsai-egress}"
PROXY_IP="${PROXY_IP:-10.89.0.2}"
AGENT_IP="${AGENT_IP:-10.89.0.3}"
PNAME="bonsai-proxy-$NAME"
ANAME="bonsai-agent-$NAME"
RUN_TIMEOUT="${RUN_TIMEOUT:-7200}"
[[ "$RUN_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo 'RUN_TIMEOUT must be positive integer seconds' >&2; exit 1; }
CONFIG="$BENCH/sandbox/generated/$NAME.codex.toml"
python3 "$BENCH/sandbox/codex_config.py" "$CONFIG" --proxy-url "http://$PROXY_IP:8787"
cp "$CONFIG" "$ROOT/$NAME.tuning"

podman image exists "$AGENT_IMAGE" || { echo "missing image $AGENT_IMAGE; run setup.sh" >&2; exit 1; }
podman image exists "$PROXY_IMAGE" || { echo "missing image $PROXY_IMAGE; run setup.sh" >&2; exit 1; }
if ! podman network exists "$INT_NET"; then
  podman network create --internal --subnet 10.89.0.0/24 "$INT_NET" >/dev/null
fi
if ! podman network exists "$EGRESS_NET"; then
  podman network create "$EGRESS_NET" >/dev/null
fi
# Refuse a pre-existing non-isolated network with the same name.
[ "$(podman network inspect -f '{{.Internal}}' "$INT_NET")" = true ] || {
  echo "$INT_NET is not an internal network" >&2; exit 1;
}
for container in "$PNAME" "$ANAME"; do
  if podman container exists "$container"; then
    echo "container $container already exists; choose a new run name" >&2
    exit 1
  fi
done
cleanup() {
  podman logs "$PNAME" >"$ROOT/$NAME.proxy.jsonl" 2>/dev/null || true
  podman rm -f "$ANAME" "$PNAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Forward secrets by environment name so they do not appear in process arguments.
podman run -d --name "$PNAME" \
  --network "$INT_NET:ip=$PROXY_IP" --network "$EGRESS_NET" \
  -e HOST="$PROXY_IP" -e PORT=8787 -e OPENAI_BASE_URL -e OPENAI_API_KEY \
  -e MAX_REQUESTS="${MAX_REQUESTS:-10000}" -e LOG_BODIES="${LOG_BODIES:-0}" \
  -v "$BENCH/sandbox/proxy:/proxy:ro" \
  "$PROXY_IMAGE" bash /proxy/openai-proxy.sh >/dev/null
for _ in $(seq 1 100); do
  if podman logs "$PNAME" 2>&1 | grep -q 'listening on'; then break; fi
  sleep 0.2
done
if ! podman logs "$PNAME" 2>&1 | grep -q 'listening on'; then
  echo 'proxy failed to start' >&2
  podman logs "$PNAME" >&2 || true
  exit 1
fi

START_EPOCH=$(date +%s)
DEADLINE_EPOCH=$((START_EPOCH + RUN_TIMEOUT))
set +e
timeout --signal=TERM --kill-after=60 "$RUN_TIMEOUT" podman run --rm --name "$ANAME" \
  --device /dev/dri --group-add keep-groups --security-opt label=disable \
  --network "$INT_NET:ip=$AGENT_IP" --shm-size=4g --tmpfs /tmp:rw,size=8g \
  -e BENCH_DEADLINE_EPOCH="$DEADLINE_EPOCH" \
  -v "$RUN:/workspace" \
  -v "$BENCH/deps/docs:/workspace/docs:ro" \
  -v "$BENCH/deps/data:/workspace/data:ro" \
  -v "$BENCH/deps/ref:/workspace/ref:ro" \
  -v "$CONFIG:/run/codex-config.toml:ro" \
  -e HOME=/root -e CODEX_HOME=/tmp/bonsai-codex \
  -e OPENAI_API_KEY=benchmark-dummy-token \
  "$AGENT_IMAGE" bash -lc '
    set -e
    mkdir -p "$CODEX_HOME"
    cp /run/codex-config.toml "$CODEX_HOME/config.toml"
    cd /workspace
    exec codex exec --strict-config --json --ephemeral --ignore-rules \
      --skip-git-repo-check --color never - < PROMPT.md
  ' >"$ROOT/$NAME.transcript.jsonl" 2>"$ROOT/$NAME.stderr.log"
rc=$?
set -e
# Kill the container before recording the final state, including timeout's KILL case.
podman rm -f "$ANAME" >/dev/null 2>&1 || true
END_EPOCH=$(date +%s)
timed_out=false
if [ "$rc" -eq 124 ] || { [ "$rc" -eq 137 ] && [ "$END_EPOCH" -ge "$DEADLINE_EPOCH" ]; }; then
  timed_out=true
fi
printf '{"agent":"codex","start_epoch":%d,"end_epoch":%d,"budget_s":%d,"elapsed_s":%d,"exit":%d,"timed_out":%s,"deadline_epoch":%d}\n' \
  "$START_EPOCH" "$END_EPOCH" "$RUN_TIMEOUT" "$((END_EPOCH - START_EPOCH))" \
  "$rc" "$timed_out" "$DEADLINE_EPOCH" >"$ROOT/$NAME.meta.json"
echo "Codex exit: $rc (elapsed $((END_EPOCH - START_EPOCH))s of $RUN_TIMEOUT s)"
exit "$rc"
