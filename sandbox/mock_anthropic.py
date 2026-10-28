#!/usr/bin/env python3
"""Minimal Anthropic Messages API mock for sandbox plumbing tests.

Turn sequence (keyed by the number of tool_result blocks in the request):
Bash -> Write -> Read -> WebFetch (expected denied) -> final text.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASH_CMD = "echo mock-tool-ok && vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName'"

STEPS = [
    ("Bash", {"command": BASH_CMD}),
    ("Write", {"file_path": "/workspace/mock-perm.txt", "content": "written-by-mock\n"}),
    ("Read", {"file_path": "/workspace/mock-perm.txt"}),
    ("WebFetch", {"url": "https://example.com", "prompt": "Say hi"}),
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
