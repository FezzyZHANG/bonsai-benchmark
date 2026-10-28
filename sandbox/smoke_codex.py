#!/usr/bin/env python3
"""Test the real Codex CLI through the proxy using only a local mock API.

This is a local integration test, not a GPU benchmark or an isolation proof.
Codex uses workspace-write on the host; full access is reserved for Podman.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

from codex_config import config
from mock_openai import make_server

HERE = Path(__file__).resolve().parent


def get(url, **kwargs):
    # Local probes must not go through the operator's HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        return opener.open(urllib.request.Request(url, **kwargs), timeout=10)
    except urllib.error.HTTPError as error:
        return error


@contextlib.contextmanager
def mock_proxy(directory, max_requests=100):
    server = make_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    # The proxy is the only process receiving this sentinel upstream credential.
    env.update(HOST="127.0.0.1", PORT=str(port),
               OPENAI_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
               OPENAI_API_KEY="mock-upstream-secret", MAX_REQUESTS=str(max_requests),
               LOG_BODIES="0", NO_PROXY="127.0.0.1,localhost")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    with (directory / "proxy.jsonl").open("w") as stdout, (directory / "proxy.stderr").open("w") as stderr:
        proxy = subprocess.Popen(["bash", str(HERE / "proxy/openai-proxy.sh")],
                                 env=env, stdout=stdout, stderr=stderr)
        try:
            for _ in range(100):
                if proxy.poll() is not None:
                    raise RuntimeError((directory / "proxy.stderr").read_text())
                try:
                    with get(url + "/__proxy_health") as result:
                        if result.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(0.05)
            else:
                raise RuntimeError("proxy did not become ready")
            yield url, server
        finally:
            proxy.terminate()
            try:
                proxy.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proxy.kill()
                proxy.wait()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def smoke(output):
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="codex-smoke-", dir=output) as tmp:
        root = Path(tmp)
        workspace = root / "workspace"
        workspace.mkdir()
        config_dir = root / "codex-home"
        config_dir.mkdir()
        with mock_proxy(root) as (url, server):
            (config_dir / "config.toml").write_text(config({"BENCH_MODEL": "mock-model"}, url))
            env = os.environ.copy()
            for key in list(env):
                if key.startswith(("ANTHROPIC_", "OPENAI_", "CODEX_")) or key.lower() in (
                        "http_proxy", "https_proxy", "all_proxy"):
                    env.pop(key, None)
            env.update(CODEX_HOME=str(config_dir), OPENAI_API_KEY="benchmark-dummy-token",
                       NO_PROXY="127.0.0.1,localhost")
            with (output / "codex-smoke.transcript.jsonl").open("w") as stdout, \
                    (output / "codex-smoke.stderr.log").open("w") as stderr:
                result = subprocess.run([
                    "codex", "exec", "--strict-config", "--json", "--ephemeral", "--ignore-rules",
                    "--skip-git-repo-check", "--sandbox", "workspace-write", "--color", "never",
                    "-C", str(workspace), "-",
                ], input="Run the supplied smoke command and report completion.\n", text=True,
                    env=env, stdout=stdout, stderr=stderr, timeout=90)
            transcript = (output / "codex-smoke.transcript.jsonl").read_text()
            events = [json.loads(line) for line in transcript.splitlines() if line.strip()]
            checks = {
                "CLI exit zero": result.returncode == 0,
                "shell executed": "CODEX_TOOL_OK" in transcript,
                "workspace file written": (workspace / "codex-smoke.txt").is_file() and
                    (workspace / "codex-smoke.txt").read_text() == "written-by-codex-mock",
                "turn completed": any(e.get("type") == "turn.completed" for e in events),
                "final response": "DONE mock Codex" in transcript,
                "credential isolated": "benchmark-dummy-token" in transcript and
                    "mock-upstream-secret" not in transcript,
                "Responses URL": len(server.requests) >= 2 and
                    all(path == "/v1/responses" for path, _, _ in server.requests),
                "no hosted web search": all(
                    not any(t.get("type", "").startswith("web_search") for t in body.get("tools", []))
                    for _, _, body in server.requests),
            }
        logs = (root / "proxy.jsonl").read_text()
        (output / "codex-smoke.proxy.jsonl").write_text(logs)
        checks["proxy logs redact key"] = "mock-upstream-secret" not in logs and "<redacted>" in logs
        (output / "codex-smoke.report.json").write_text(json.dumps(checks, indent=2) + "\n")
        for name, ok in checks.items():
            print(f"{'PASS' if ok else 'FAIL'} {name}")
        if not all(checks.values()):
            print((output / "codex-smoke.stderr.log").read_text()[-6000:])
        return all(checks.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "generated/local-smoke")
    args = parser.parse_args()
    raise SystemExit(0 if smoke(args.out.resolve()) else 1)
