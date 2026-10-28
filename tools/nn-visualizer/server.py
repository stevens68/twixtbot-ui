#!/usr/bin/env python3
"""Serve the visualizer and expose the real Twixt backend as a local JSON API.

Run from the repository root:
    python tools/nn-visualizer/server.py
Then open http://127.0.0.1:8765
"""
from __future__ import annotations

import json
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
VISUALIZER = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "tools"))
from nn_visualizer_adapter import snapshot
from src.files import parse_t1_file


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VISUALIZER), **kwargs)

    def do_POST(self):
        endpoint = urlparse(self.path).path
        if endpoint not in ("/api/trace", "/api/t1"):
            self.send_error(404)
            return
        try:
            started = time.perf_counter()
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("Request body must be between 1 byte and 1 MB.")
            raw = self.rfile.read(length)
            read_done = time.perf_counter()

            snapshot_ms = 0.0
            if endpoint == "/api/t1":
                _, parsed_moves = parse_t1_file(raw.decode("utf-8-sig").splitlines())
                moves = [str(move) for move in parsed_moves]
                if len(moves) > 500:
                    raise ValueError("At most 500 moves are accepted.")
                result = {"moves": moves}
            else:
                payload = json.loads(raw)
                moves = payload.get("moves")
                if not isinstance(moves, list) or not all(isinstance(m, str) for m in moves):
                    raise ValueError("'moves' must be a JSON array of move strings.")
                if len(moves) > 500:
                    raise ValueError("At most 500 moves are accepted.")
                snapshot_started = time.perf_counter()
                result = snapshot(
                    moves,
                    bool(payload.get("allow_scl", False)),
                    bool(payload.get("allow_swap", True)),
                )
                snapshot_ms = (time.perf_counter() - snapshot_started) * 1000
            parsed_done = time.perf_counter()

            body = json.dumps(result, separators=(",", ":")).encode("utf-8")
            serialized_done = time.perf_counter()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            written_done = time.perf_counter()

            print(
                "[NN perf server] "
                f"endpoint={endpoint} moves={len(moves)} "
                f"read_ms={(read_done - started) * 1000:.1f} "
                f"request_parse_ms={(parsed_done - read_done) * 1000 - snapshot_ms:.1f} "
                f"snapshot_ms={snapshot_ms:.1f} "
                f"serialize_ms={(serialized_done - parsed_done) * 1000:.1f} "
                f"write_ms={(written_done - serialized_done) * 1000:.1f} "
                f"response_bytes={len(body)} "
                f"total_ms={(written_done - started) * 1000:.1f}",
                flush=True,
            )
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
