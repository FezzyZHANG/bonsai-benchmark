#!/usr/bin/env python3
"""Minimal Anthropic Messages API mock for sandbox plumbing tests.

Each step is a separate tool call. The Bash steps are deliberately simple:
Claude Code's command classifier can deny safe compound commands, so this
smoke test issues individual commands instead of one script.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MOCK_IP = os.environ.get("MOCK_IP", "10.90.0.4")
MOCK_PORT = os.environ.get("MOCK_PORT", "8000")

SMOKE_COMP = """#version 450
layout(local_size_x = 1) in;
void main() {}
"""

SMOKE_REF_IN = """load /workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf
info
tokenize 48656c6c6f 0
quit
"""

SMOKE_NET = f"""#!/bin/bash
if curl -m 3 -sS -o /dev/null https://example.com 2>/dev/null; then
  echo egress-OPEN
else
  echo egress-blocked
fi
if curl -m 3 -sS -o /dev/null http://{MOCK_IP}:{MOCK_PORT}/ 2>/dev/null; then
  echo egress-OPEN-egress-net
else
  echo egress-blocked-egress-net
fi
"""

SMOKE_PY = """#!/bin/bash
python3 -c "import gguf, numpy, safetensors, yaml; from safetensors import safe_open; f=safe_open('/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors', framework='numpy'); assert len(f.keys()) == 426 and not f.metadata(); print('py-ok')"
"""

BUILD_ZIG = """const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addLibrary(.{
        .name = "bonsai",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(lib);
}
"""

MAIN_ZIG = """export fn bonsai_abi_version() callconv(.c) u32 {
    return 1;
}

export fn bonsai_last_error() callconv(.c) ?[*:0]const u8 {
    return null;
}
"""

STEPS = [
    ("Write", {"file_path": "/workspace/smoke.comp", "content": SMOKE_COMP}),
    ("Write", {"file_path": "/workspace/smoke-ref.in", "content": SMOKE_REF_IN}),
    ("Write", {"file_path": "/workspace/build.zig", "content": BUILD_ZIG}),
    ("Write", {"file_path": "/workspace/src/main.zig", "content": MAIN_ZIG}),
    ("Write", {"file_path": "/workspace/mock-perm.txt", "content": "written-by-mock\n"}),
    ("Write", {"file_path": "/workspace/smoke-net.sh", "content": SMOKE_NET}),
    ("Write", {"file_path": "/workspace/smoke-py.sh", "content": SMOKE_PY}),
    ("Bash", {"command": "zig version"}),
    ("Bash", {"command": "glslc --version"}),
    ("Bash", {"command": "glslangValidator --version"}),
    ("Bash", {"command": "command -v slangc"}),
    ("Bash", {"command": "spirv-val --version"}),
    ("Bash", {"command": "vulkaninfo --summary"}),
    ("Bash", {"command": "glslc smoke.comp -o smoke.spv"}),
    ("Bash", {"command": "spirv-val smoke.spv"}),
    ("Bash", {"command": "bash smoke-py.sh"}),
    ("Bash", {"command": "./ref/bin/bonsai-ref-cpu < smoke-ref.in 2>/dev/null"}),
    ("Bash", {"command": "printenv ANTHROPIC_AUTH_TOKEN"}),
    ("Bash", {"command": "zig build -Doptimize=ReleaseFast"}),
    ("Bash", {"command": "bash smoke-net.sh"}),
    ("Read", {"file_path": "/workspace/mock-perm.txt"}),
    ("Write", {"file_path": "/workspace/data/.smoke", "content": "x"}),
    ("Write", {"file_path": "/workspace/docs/.smoke", "content": "x"}),
    ("Write", {"file_path": "/workspace/ref/.smoke", "content": "x"}),
    ("WebFetch", {"url": "https://example.com", "prompt": "Say hi"}),
    ("WebSearch", {"query": "benchmark smoke test"}),
]


def tool_stream(idx, name, inp):
    return [
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "tool_use", "id": f"toolu_mock_{idx}",
                                                   "name": name, "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "input_json_delta",
                                           "partial_json": json.dumps(inp)}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                           "usage": {"output_tokens": 12}}),
        ("message_stop", {"type": "message_stop"}),
    ]


def text_stream(text):
    return [
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": 5}}),
        ("message_stop", {"type": "message_stop"}),
    ]


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        sys.stderr.write("mock %s %s\n" % (self.command, self.path))
        sys.stderr.flush()

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._json(200, {"data": [{"id": "mock-model", "type": "model"}], "has_more": False})

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(n).decode("utf-8", "replace")
        sys.stderr.write("mock POST %s auth=%r len=%d\n" %
                         (self.path, self.headers.get("authorization"), len(raw)))
        sys.stderr.flush()
        if self.path.endswith("/count_tokens"):
            self._json(200, {"input_tokens": 42})
            return
        try:
            req = json.loads(raw)
        except Exception:
            req = {}
        results = raw.count('"tool_result"')
        step = STEPS[results] if results < len(STEPS) else None

        if not req.get("stream"):
            if step:
                name, inp = step
                msg = {"id": "msg_mock", "type": "message", "role": "assistant", "model": "mock",
                       "content": [{"type": "tool_use", "id": f"toolu_mock_{results}",
                                    "name": name, "input": inp}],
                       "stop_reason": "tool_use", "stop_sequence": None,
                       "usage": {"input_tokens": 10, "output_tokens": 12}}
            else:
                msg = {"id": "msg_mock", "type": "message", "role": "assistant", "model": "mock",
                       "content": [{"type": "text", "text": "DONE mock"}],
                       "stop_reason": "end_turn", "stop_sequence": None,
                       "usage": {"input_tokens": 10, "output_tokens": 2}}
            self._json(200, msg)
            return

        head = {"type": "message_start",
                "message": {"id": "msg_mock", "type": "message", "role": "assistant",
                            "model": "mock", "content": [], "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 10, "output_tokens": 1}}}
        if step:
            events = [("message_start", head)] + tool_stream(results, step[0], step[1])
        else:
            events = [("message_start", head)] + text_stream("DONE mock")
        body = b"".join((f"event: {ev}\ndata: {json.dumps(d)}\n\n").encode() for ev, d in events)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
