#!/usr/bin/env python3
"""Serve the visualizer and expose the real Twixt backend as a local JSON API.

Run from the repository root:
    python tools/nn-visualizer/server.py
Then open http://127.0.0.1:8765
"""
from __future__ import annotations

import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
VISUALIZER = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.nn_visualizer_adapter import snapshot  # type: ignore


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VISUALIZER), **kwargs)

    def do_POST(self):
        if urlparse(self.path).path != "/api/trace":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("Request body must be between 1 byte and 1 MB.")
            payload = json.loads(self.rfile.read(length))
            moves = payload.get("moves")
            if not isinstance(moves, list) or not all(isinstance(m, str) for m in moves):
                raise ValueError("'moves' must be a JSON array of move strings.")
            if len(moves) > 500:
                raise ValueError("At most 500 moves are accepted.")
            result = snapshot(moves, bool(payload.get("allow_scl", False)))
            body = json.dumps(result, separators=(",", ":")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, fmt, *args):
        print("[visualizer] " + fmt % args)


if __name__ == "__main__":
    print("Twixt NN visualizer: http://127.0.0.1:8765")
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
