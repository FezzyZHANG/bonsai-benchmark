#!/bin/bash
# Build the agent and proxy images. Needs network for apt/npm.
set -euo pipefail
BENCH="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$BENCH"

podman build -f sandbox/Containerfile -t "${AGENT_IMAGE:-bonsai-agent:1.0}" .
podman build -f sandbox/proxy/Containerfile -t "${PROXY_IMAGE:-bonsai-proxy:1.0}" sandbox/proxy

echo "images:"
podman images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -E 'bonsai-(agent|proxy)' || true
