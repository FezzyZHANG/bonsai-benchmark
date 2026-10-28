#!/usr/bin/env python3
"""Prepare an isolated Codex benchmark workspace without Claude settings."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("root", nargs="?", default=os.environ.get("RUNS_ROOT", "/mnt/workspace/bonsai-runs"))
    args = parser.parse_args()
    if args.name in (".", "..") or not re.fullmatch(r"[a-zA-Z0-9_.-]+", args.name):
        parser.error("invalid RUN_NAME")
    bench = Path(__file__).resolve().parent.parent
    root = Path(args.root).resolve()
    run = root / args.name
    if run == bench or bench in run.parents:
        parser.error(f"RUNS_ROOT must be outside {bench}")
    if run.exists() or run.is_symlink():
        parser.error(f"{run} already exists")
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bench / "agent", run, copy_function=shutil.copyfile,
                    ignore=shutil.ignore_patterns("__pycache__"))
    prompt = run / "PROMPT.md"
    text = prompt.read_text()
    start = text.index("## Bash in this sandbox")
    end = text.index("Work autonomously", start)
    prompt.write_text(text[:start] +
        "The Podman container is the security boundary. Web search, apps, "
        "plugins and subagents are disabled. Use the shell and local files. "
        "The data/docs/ref mounts are read-only. Save long command output "
        "to a workspace file and inspect it there.\n\n" + text[end:])
    (run / ".tmp").mkdir(exist_ok=True)
    for command in (["init", "-q"], ["add", "-A"],
                    ["-c", "user.name=bonsai", "-c", "user.email=bonsai@localhost",
                     "commit", "-qm", "initial workspace"]):
        subprocess.run(["git", "-C", str(run), *command], check=True)
    (root / f"{args.name}.agent").write_text("codex\n")
    print(f"Workspace: {run}\nAgent: codex\n\n"
          "Set OPENAI_API_KEY and BENCH_MODEL, then run:\n"
          f'  BENCH_AGENT=codex "{bench}/sandbox/run_agent.sh" "{args.name}" "{root}"\n\n'
          f'Grade: python3 "{bench}/verification/verify.py" "{run}" --out "{root}/{args.name}-report"')


if __name__ == "__main__":
    main()
