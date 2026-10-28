#!/usr/bin/env bash
# Reuse the streaming, credential-injecting proxy with a Responses API prefix.
# The client calls /responses; OPENAI_BASE_URL includes /v1 (or a gateway prefix).
set -euo pipefail
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (upstream credential)}"
export ANTHROPIC_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export ANTHROPIC_AUTH_TOKEN="$OPENAI_API_KEY"
export ALLOWED_PATHS=/responses,/responses/compact
export ALLOWED_METHODS=POST
export AUTH_HEADER=authorization
export LOG_BODIES="${LOG_BODIES:-0}"
exec bash "$(cd "$(dirname "$0")" && pwd)/anthropic-proxy.sh" "$@"
