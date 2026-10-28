#!/usr/bin/env python3
"""Generate an isolated Codex configuration; never copy operator credentials."""
import argparse
import json
import os
from pathlib import Path


def config(env, proxy_url):
    model = env.get("BENCH_MODEL", "").strip()
    if not model:
        raise ValueError("set BENCH_MODEL to the model ID offered by your Responses API")
    for name in ("BENCH_MAX_OUTPUT", "BENCH_MAX_TURNS", "BENCH_MAX_SUBAGENTS",
                 "BENCH_MAX_SUBAGENTS_PER_SESSION", "BENCH_MAX_SUBAGENT_DEPTH",
                 "BENCH_SUBAGENT_MODEL", "BENCH_SIMPLE_SYSTEM_PROMPT"):
        if env.get(name):
            raise ValueError(f"{name} has no supported Codex mapping; unset it. "
                             "Codex runs one agent; RUN_TIMEOUT and MAX_REQUESTS bound the run.")
    lines = [
        f"model = {json.dumps(model)}",
        'model_provider = "bonsai"',
        'approval_policy = "never"',
        # Only used by the container runner. The local smoke test overrides this.
        'sandbox_mode = "danger-full-access"',
        'web_search = "disabled"',
    ]
    effort = env.get("BENCH_EFFORT")
    if effort:
        if effort not in ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"):
            raise ValueError("invalid BENCH_EFFORT")
        lines.append(f"model_reasoning_effort = {json.dumps(effort)}")
    for variable, key in (("BENCH_MAX_CONTEXT", "model_context_window"),
                          ("BENCH_COMPACT_WINDOW", "model_auto_compact_token_limit")):
        value = env.get(variable)
        if value:
            if not value.isdecimal() or int(value) <= 0:
                raise ValueError(f"{variable} must be a positive integer")
            lines.append(f"{key} = {int(value)}")
    if env.get("BENCH_MAX_CONTEXT") and env.get("BENCH_COMPACT_WINDOW"):
        if int(env["BENCH_COMPACT_WINDOW"]) >= int(env["BENCH_MAX_CONTEXT"]):
            raise ValueError("BENCH_COMPACT_WINDOW must be smaller than BENCH_MAX_CONTEXT")
    lines += [
        "", "[features]",
        "multi_agent = false", "apps = false", "plugins = false",
        "hooks = false", "memories = false", "browser_use = false",
        "computer_use = false", "image_generation = false",
        "enable_request_compression = false",
        "", "[model_providers.bonsai]",
        'name = "Bonsai benchmark proxy"',
        f"base_url = {json.dumps(proxy_url)}",
        'wire_api = "responses"',
        'env_key = "OPENAI_API_KEY"',
        "requires_openai_auth = false",
        "supports_websockets = false",
        "request_max_retries = 2", "stream_max_retries = 2",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--proxy-url", required=True)
    args = parser.parse_args()
    try:
        text = config(os.environ, args.proxy_url)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)


if __name__ == "__main__":
    main()
