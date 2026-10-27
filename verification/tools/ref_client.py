#!/usr/bin/env python3
"""Driver for the hidden bonsai-ref line protocol."""
import os
import subprocess


class RefError(RuntimeError):
    pass


class RefClient:
    def __init__(self, binary, stderr_path=None):
        self.stderr = open(stderr_path, "w") if stderr_path else subprocess.DEVNULL
        self.p = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )

    def cmd(self, line, prefix="ok"):
        if self.p.stdin is None or self.p.stdout is None:
            raise RefError("runner is closed")
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        reply = self.p.stdout.readline()
        if not reply:
            raise RefError(f"runner died while handling: {line[:120]}")
        reply = reply.rstrip("\n")
        if not reply.startswith(prefix):
            raise RefError(f"{line[:120]} -> {reply}")
        return reply

    def load(self, model):
        self.cmd(f"load {model}")

    def info(self):
        return self.cmd("info")

    def tokenize(self, text, add_bos=0):
        hex_text = text.encode("utf-8").hex()
        reply = self.cmd(f"tokenize {hex_text} {int(bool(add_bos))}")
        parts = reply.split()
        n = int(parts[1])
        ids = [int(x) for x in parts[2:2 + n]]
        if len(ids) != n:
            raise RefError("short tokenize reply")
        return ids

    def ctx(self, cid, n_ctx, backend="cpu"):
        self.cmd(f"ctx {cid} {n_ctx} {backend}")

    def reset(self, cid):
        self.cmd(f"reset {cid}")

    def eval(self, cid, ids):
        if not ids:
            return
        self.cmd(f"eval {cid} {len(ids)} " + " ".join(str(int(t)) for t in ids))

    def logits(self, cid, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        reply = self.cmd(f"logits {cid} {path}")
        n = int(reply.split()[1])
        size = os.path.getsize(path)
        if size != n * 4:
            raise RefError(f"logits size {size} != {n * 4}")
        return n

    def gen(self, cid, n_predict, ignore_eos=True):
        reply = self.cmd(f"gen {cid} {n_predict} {1 if ignore_eos else 0}")
        parts = reply.split()
        n = int(parts[1])
        ids = [int(x) for x in parts[2:2 + n]]
        if len(ids) != n:
            raise RefError("short gen reply")
        return ids

    def close(self):
        if self.p.poll() is None:
            try:
                self.p.stdin.write("quit\n")
                self.p.stdin.flush()
            except Exception:
                pass
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()
        if self.stderr not in (None, subprocess.DEVNULL):
            self.stderr.close()
