import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SANDBOX = Path(__file__).resolve().parents[1]


class PortabilityTests(unittest.TestCase):
    def test_claude_workspace_preparation_on_host(self):
        with tempfile.TemporaryDirectory(prefix="bonsai-portability-") as tmp:
            name = f"test-portability-{os.getpid()}"
            command = ["bash", str(SANDBOX / "prepare_run.sh"), name, tmp]
            env = {**os.environ, "BENCH_AGENT": "claude"}
            try:
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                run = Path(tmp) / name
                self.assertEqual((run / "PROMPT.md").read_bytes(),
                                 (SANDBOX.parent / "agent/PROMPT.md").read_bytes())
                self.assertTrue((run / ".claude-home").is_dir())
                self.assertTrue((run / ".git").is_dir())
                self.assertIn(f'"{Path(tmp).resolve()}"', result.stdout)
                repeated = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertNotEqual(repeated.returncode, 0)
            finally:
                (SANDBOX / "generated" / f"{name}.settings.json").unlink(missing_ok=True)

    def test_rejects_traversal_and_symlink_into_operator_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp) / "operator-repo"
            link.symlink_to(SANDBOX.parent, target_is_directory=True)
            for name, root in (("../escape", tmp), ("forbidden", str(link))):
                result = subprocess.run(["bash", str(SANDBOX / "prepare_run.sh"), name, root],
                                        env={**os.environ, "BENCH_AGENT": "claude"},
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)

    def test_setup_rejects_macos_before_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            uname = Path(tmp) / "uname"
            uname.write_text("#!/bin/sh\nprintf 'Darwin\\n'\n")
            uname.chmod(0o755)
            result = subprocess.run(["bash", str(SANDBOX.parent / "setup.sh")],
                                    env={**os.environ, "PATH": tmp + os.pathsep + os.environ["PATH"]},
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--prepare-only", result.stderr)
            self.assertNotIn("download", result.stdout)


if __name__ == "__main__":
    unittest.main()
