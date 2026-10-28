import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SANDBOX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX))
from codex_config import config
from smoke_codex import get, mock_proxy


class ConfigTests(unittest.TestCase):
    def test_settings_and_no_secret(self):
        text = config({"BENCH_MODEL": 'model"\\injection', "BENCH_EFFORT": "high",
                       "BENCH_COMPACT_WINDOW": "300000", "BENCH_MAX_CONTEXT": "1000000",
                       "OPENAI_API_KEY": "not-for-agent"}, "http://proxy:8787")
        self.assertNotIn("not-for-agent", text)
        self.assertIn('web_search = "disabled"', text)
        self.assertIn("multi_agent = false", text)
        self.assertIn("model_auto_compact_token_limit = 300000", text)
        self.assertIn('model = "model\\"\\\\injection"', text)

    def test_invalid_and_unsupported_tuning_rejected(self):
        for env in ({}, {"BENCH_MODEL": "x", "BENCH_EFFORT": "oops"},
                    {"BENCH_MODEL": "x", "BENCH_MAX_CONTEXT": "-1"},
                    {"BENCH_MODEL": "x", "BENCH_MAX_TURNS": "10"},
                    {"BENCH_MODEL": "x", "BENCH_MAX_CONTEXT": "10", "BENCH_COMPACT_WINDOW": "11"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                config(env, "http://proxy")


class PrepareTests(unittest.TestCase):
    def test_codex_preparation_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="bonsai-prepare-") as tmp:
            for agent in ("codex",):
                env = {**os.environ, "BENCH_AGENT": agent}
                command = ["bash", str(SANDBOX / "prepare_run.sh"), f"test-{agent}-{os.getpid()}", tmp]
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                run = Path(tmp) / command[-2]
                prompt = (run / "PROMPT.md").read_text()
                self.assertEqual("Claude Code's Bash" in prompt, agent == "claude")
                self.assertEqual((run / ".claude-home").exists(), agent == "claude")
                self.assertTrue((run / "TASK.md").exists())
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)

    def test_invalid_paths_and_runner(self):
        for name, root, agent in (("../escape", "/tmp", "codex"),
                                  ("test", str(SANDBOX / "../runs"), "codex"),
                                  ("test", "/tmp", "invalid")):
            result = subprocess.run(["bash", str(SANDBOX / "prepare_run.sh"), name, root],
                                    env={**os.environ, "BENCH_AGENT": agent}, capture_output=True)
            self.assertNotEqual(result.returncode, 0)


class MonitorTests(unittest.TestCase):
    def test_codex_and_claude_events(self):
        loader = importlib.machinery.SourceFileLoader("monitor", str(SANDBOX / "monitor_log"))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        events = [
            {"type": "item.started", "item": {"type": "command_execution", "command": "zig version"}},
            {"type": "item.completed", "item": {"type": "command_execution", "exit_code": 0,
                                                "aggregated_output": "0.17.0"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "finished Codex"}},
            {"type": "turn.completed", "usage": {"input_tokens": 10}},
            {"type": "turn.failed", "error": {"message": "API failed"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "finished Claude"}]}},
        ]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            for event in events:
                module.render(event, module.Style(False), {}, {}, {})
        for expected in ("zig version", "0.17.0", "finished Codex", "finished Claude", "API failed"):
            self.assertIn(expected, output.getvalue())


@unittest.skipUnless(shutil.which("deno"), "Deno is required for proxy integration tests")
class ProxyTests(unittest.TestCase):
    def test_path_auth_sse_and_compaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock_proxy(root) as (url, server):
                data = json.dumps({"model": "mock", "input": [{"type": "function_call_output"}]}).encode()
                with get(url + "/responses", data=data, headers={"Authorization": "Bearer dummy"}) as result:
                    self.assertEqual(result.status, 200)
                    self.assertIn(b"response.completed", result.read())
                with get(url + "/responses/compact", data=b"{}") as result:
                    self.assertEqual(result.status, 200)
                    self.assertEqual(json.load(result)["object"], "response.compaction")
                for path, method in (("/v1/messages", "POST"), ("/responses", "GET"), ("/files", "POST")):
                    with get(url + path, method=method) as result:
                        self.assertEqual(result.status, 403)
                self.assertEqual([r[0] for r in server.requests], ["/v1/responses", "/v1/responses/compact"])
                for _, headers, _ in server.requests:
                    self.assertEqual({k.lower(): v for k, v in headers.items()}["authorization"],
                                     "Bearer mock-upstream-secret")
            logs = (root / "proxy.jsonl").read_text()
            self.assertNotIn("mock-upstream-secret", logs)
            self.assertIn("<redacted>", logs)

    def test_request_budget(self):
        with tempfile.TemporaryDirectory() as tmp, mock_proxy(Path(tmp), max_requests=1) as (url, server):
            for expected in (200, 429):
                with get(url + "/responses/compact", data=b"{}") as result:
                    self.assertEqual(result.status, expected)
                    result.read()
            self.assertEqual(len(server.requests), 1)


class RunnerTests(unittest.TestCase):
    """Exercise the real shell runner against a recording Podman executable."""
    def run_runner(self, root, exit_code=0, internal=True):
        fakebin = root / "bin"
        fakebin.mkdir()
        run = root / "run"
        run.mkdir()
        (root / "run.agent").write_text("codex\n")
        (fakebin / "python3").write_text(f'''#!{sys.executable}
import os,sys
if sys.argv[1].endswith("doctor.py"): sys.exit(0)
os.execv({sys.executable!r}, [{sys.executable!r}]+sys.argv[1:])
''')
        (fakebin / "podman").write_text(f'''#!{sys.executable}
import json,os,sys
args=sys.argv[1:]
with open(os.environ["TEST_PODMAN_LOG"],"a") as f: f.write(json.dumps(args)+"\\n")
if args[:2]==["network","inspect"]: print({str(internal).lower()!r})
elif args[:2]==["container","exists"]: sys.exit(1)
elif args[0]=="logs": print("listening on mock")
elif args[0]=="run" and "--rm" in args:
    print('{{"type":"turn.completed"}}')
    sys.exit(int(os.environ["TEST_EXIT"]))
''')
        (fakebin / "timeout").write_text(f'''#!{sys.executable}
import os,sys
os.execvp(sys.argv[4],sys.argv[4:])
''')
        for script in fakebin.iterdir():
            script.chmod(0o755)
        env = {k: v for k, v in os.environ.items() if not k.startswith("BENCH_")}
        env.update(PATH=str(fakebin) + os.pathsep + env["PATH"], TEST_EXIT=str(exit_code),
                   TEST_PODMAN_LOG=str(root / "podman.jsonl"), BENCH_AGENT="codex",
                   BENCH_MODEL="mock", OPENAI_API_KEY="real-secret-stays-with-proxy", RUN_TIMEOUT="60")
        return subprocess.run(["bash", str(SANDBOX / "run_agent.sh"), "run", str(root)],
                              env=env, capture_output=True, text=True)

    def test_exit_and_timeout_metadata_and_isolation(self):
        for exit_code in (0, 7, 124, 137):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result = self.run_runner(root, exit_code)
                self.assertEqual(result.returncode, exit_code, result.stderr)
                meta = json.loads((root / "run.meta.json").read_text())
                # An early 137 can be OOM/SIGKILL, not a timeout.
                self.assertEqual(meta["timed_out"], exit_code == 124)
                self.assertEqual(meta["deadline_epoch"], meta["start_epoch"] + 60)
                calls = [json.loads(line) for line in (root / "podman.jsonl").read_text().splitlines()]
                agent = next(c for c in calls if c[:2] == ["run", "--rm"])
                self.assertIn("OPENAI_API_KEY=benchmark-dummy-token", agent)
                self.assertIn("--device", agent)
                self.assertEqual(sum(":/workspace/" in c and c.endswith(":ro") for c in agent), 3)
                self.assertNotIn("real-secret-stays-with-proxy", json.dumps(calls))
                self.assertNotIn("real-secret-stays-with-proxy", (root / "run.tuning").read_text())

    def test_existing_non_internal_network_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_runner(root, internal=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not an internal network", result.stderr)
            calls = [json.loads(line) for line in (root / "podman.jsonl").read_text().splitlines()]
            self.assertFalse(any(c[0] == "run" for c in calls))


if __name__ == "__main__":
    unittest.main()
