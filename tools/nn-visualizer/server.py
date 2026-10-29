#!/usr/bin/env python3
"""Serve the visualizer and expose the real Twixt backend as a local JSON API.

Run from the repository root:
    python tools/nn-visualizer/server.py
Then open http://127.0.0.1:8765
"""
from __future__ import annotations

import gzip
import json
import sys
import time

import torch
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
from src.constants import MODEL_FOLDER


class Handler(SimpleHTTPRequestHandler):
    # Keep local browser fetches on one TCP connection instead of paying a new
    # connection setup for every step through a long game. Every POST response
    # below includes Content-Length, as required for persistent HTTP/1.1.
    protocol_version = "HTTP/1.1"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VISUALIZER), **kwargs)

    def do_GET(self):
        endpoint = urlparse(self.path).path
        if endpoint != "/api/weights":
            return super().do_GET()
        try:
            checkpoint = torch.load(MODEL_FOLDER, map_location="cpu", weights_only=True)
            state_dict = checkpoint["state_dict"]
            def read_conv_layer(name):
                weight_key = name + ".weight"
                if weight_key not in state_dict:
                    raise KeyError(f"Model checkpoint does not contain {weight_key}.")
                # PyTorch Conv2d: [out_channels, in_channels, height, width].
                weights = state_dict[weight_key].detach().cpu().permute(0, 2, 3, 1).tolist()
                layer = {
                    "name": name,
                    "weight_key": weight_key,
                    "filter_count": len(weights),
                    "kernel_height": len(weights[0]),
                    "kernel_width": len(weights[0][0]),
                    "channels": len(weights[0][0][0]),
                    "has_bias": name + ".bias" in state_dict,
                    "source": str(Path(MODEL_FOLDER).relative_to(ROOT)),
                }
                if layer["has_bias"]:
                    layer["bias"] = state_dict[name + ".bias"].detach().cpu().tolist()
                return layer, weights

            links_layer, link_weights = read_conv_layer("links")
            pegs_layer, peg_weights = read_conv_layer("pegs")
            locations_layer, locations_weights = read_conv_layer("location")
            body = json.dumps({
                "layers": {
                    "links": links_layer,
                    "pegs": pegs_layer,
                    "locations": locations_layer,
                },
                "weights": {
                    "links": link_weights,
                    "pegs": peg_weights,
                    "locations": locations_weights,
                },
            }, separators=(",", ":")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"error": f"Could not load trained model weights from {MODEL_FOLDER}: {exc}"}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

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
                # The visualizer needs the authoritative board and input tensors,
                # not terminal-result detection. Draw detection performs an
                # expensive recursive path search and is deliberately skipped
                # for interactive snapshots; the CLI retains the default check.
                result = snapshot(
                    moves,
                    bool(payload.get("allow_scl", False)),
                    bool(payload.get("allow_swap", True)),
                    check_draw=False,
                )
                snapshot_ms = (time.perf_counter() - snapshot_started) * 1000
            parsed_done = time.perf_counter()

            body = json.dumps(result, separators=(",", ":")).encode("utf-8")
            uncompressed_bytes = len(body)
            serialized_done = time.perf_counter()

            # Snapshot responses are JSON and compress well. Fetch transparently
            # decompresses gzip responses, so the browser-side JSON handling is
            # unchanged. Keep a Vary header for any intermediary caches.
            accepts_gzip = "gzip" in self.headers.get("Accept-Encoding", "").lower()
            if accepts_gzip:
                compressed_body = gzip.compress(body, compresslevel=5)
                if len(compressed_body) < len(body):
                    body = compressed_body
                    content_encoding = "gzip"
                else:
                    content_encoding = None
            else:
                content_encoding = None

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Vary", "Accept-Encoding")
            if content_encoding:
                self.send_header("Content-Encoding", content_encoding)
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
                f"response_bytes={uncompressed_bytes} "
                f"wire_bytes={len(body)} "
                f"compression={'gzip' if content_encoding else 'none'} "
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
