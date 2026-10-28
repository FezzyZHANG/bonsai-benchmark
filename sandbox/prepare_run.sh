#!/bin/bash

case "${BENCH_AGENT:-claude}" in
  codex) exec python3 "$(dirname "$0")/prepare_codex.py" "$@" ;;
  claude) ;;
  *) echo 'BENCH_AGENT must be claude or codex' >&2; exit 1 ;;
esac

# Prepare a fresh agent workspace and its Claude Code settings.
#
#   sandbox/prepare_run.sh RUN_NAME [RUNS_ROOT]
set -euo pipefail
BENCH="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
NAME="${1:?usage: prepare_run.sh RUN_NAME [RUNS_ROOT]}"
ROOT="${2:-/mnt/workspace/bonsai-runs}"
RUN="$ROOT/$NAME"
case "$RUN" in "$BENCH"/*) echo "RUNS_ROOT must be outside $BENCH" >&2; exit 1;; esac
[ -e "$RUN" ] && { echo "$RUN already exists" >&2; exit 1; }
mkdir -p "$ROOT"
cp -R --no-preserve=all "$BENCH/agent" "$RUN"
find "$RUN" -xdev -name __pycache__ -prune -exec rm -rf {} +
mkdir -p "$RUN/.tmp" "$RUN/.claude-home"
printf '{}\n' > "$RUN/.claude.json"
git -C "$RUN" init -q && git -C "$RUN" add -A && \
  git -C "$RUN" -c user.name=bonsai -c user.email=bonsai@localhost commit -qm "initial workspace"
mkdir -p "$BENCH/sandbox/generated"
OUT="$BENCH/sandbox/generated/$NAME.settings.json"
cp "$BENCH/sandbox/settings.template.json" "$OUT"
python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$OUT"
cat <<EOF
Workspace: $RUN
Settings:  $OUT

Run (set ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN first):
  ANTHROPIC_BASE_URL=... ANTHROPIC_AUTH_TOKEN=... $BENCH/sandbox/run_agent.sh $NAME

Grade afterwards:
  python3 $BENCH/verification/verify.py $RUN --out $ROOT/$NAME-report
EOF
