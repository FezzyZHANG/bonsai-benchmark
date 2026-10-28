#!/usr/bin/env python3
"""Deterministic Responses API for Codex smoke tests. Never calls a real API."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except ValueError:
            self.send_error(400, "expected uncompressed JSON")
            return
        self.server.requests.append((self.path, dict(self.headers), body))
        if self.headers.get("Authorization") != "Bearer mock-upstream-secret":
            self.send_error(401)
            return
        if self.path.endswith("/responses/compact"):
            data = json.dumps({"id": "cmp_mock", "object": "response.compaction", "output": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if not self.path.endswith("/responses"):
            self.send_error(404)
            return

        inputs = body.get("input", [])
        has_output = any(i.get("type") == "function_call_output" for i in inputs if isinstance(i, dict))
        if has_output:
            item = {"id": "msg_mock", "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "DONE mock Codex", "annotations": []}]}
        else:
            def functions(tools):
                for tool in tools:
                    if tool.get("type") == "namespace":
                        yield from functions(tool.get("tools", []))
                    else:
                        yield tool

            names = {t.get("name") for t in functions(body.get("tools", []))}
            command = ("python3 -c \"import os,pathlib; "
                       "pathlib.Path('codex-smoke.txt').write_text('written-by-codex-mock'); "
                       "print('CODEX_TOOL_OK'); print(os.environ.get('OPENAI_API_KEY', ''))\"")
            if "exec_command" in names:
                name, arguments = "exec_command", {"cmd": command, "max_output_tokens": 1000}
            elif "shell_command" in names:
                name, arguments = "shell_command", {"command": command, "timeout_ms": 10000}
            elif "shell" in names:
                name, arguments = "shell", {"command": ["bash", "-c", command], "timeout_ms": 10000}
            else:
                self.send_error(400, "no supported shell tool advertised")
                return
            item = {"id": "fc_mock", "type": "function_call", "call_id": "call_mock",
                    "name": name, "arguments": json.dumps(arguments), "status": "completed"}
        response = {"id": "resp_mock", "object": "response", "model": body.get("model"),
                    "created_at": 1, "status": "completed", "output": [item],
                    "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20,
                              "input_tokens_details": {"cached_tokens": 0}}}
        events = [
            {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
            {"type": "response.output_item.added", "output_index": 0, "item": item},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": response},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for seq, event in enumerate(events):
            event["sequence_number"] = seq
            self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()


def make_server(host="127.0.0.1", port=0):
    server = ThreadingHTTPServer((host, port), Handler)
    server.requests = []
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    make_server(args.host, args.port).serve_forever()
