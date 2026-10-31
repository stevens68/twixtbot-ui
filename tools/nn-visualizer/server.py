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
import torch.nn.functional as F
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
VISUALIZER = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "tools"))
from nn_visualizer_adapter import snapshot
from src.files import parse_t1_file
from src.constants import MODEL_FOLDER


_CHECKPOINT_CACHE = {"mtime_ns": None, "value": None}


def load_checkpoint():
    """Load the model once per checkpoint file version for on-demand block requests."""
    modified = Path(MODEL_FOLDER).stat().st_mtime_ns
    if _CHECKPOINT_CACHE["value"] is None or _CHECKPOINT_CACHE["mtime_ns"] != modified:
        _CHECKPOINT_CACHE["value"] = torch.load(
            MODEL_FOLDER, map_location="cpu", weights_only=True
        )
        _CHECKPOINT_CACHE["mtime_ns"] = modified
    return _CHECKPOINT_CACHE["value"]


def residual_block_payload(checkpoint, block_index):
    state_dict = checkpoint["state_dict"]
    config = checkpoint.get("config", {})
    block_count = int(config.get("blocks", 0))
    if block_index < 0 or block_index >= block_count:
        raise ValueError(
            f"Residual block index must be between 0 and {block_count - 1}."
        )

    layers = {}
    weights = {}
    for name in ("conv1", "conv2"):
        weight_key = f"blocks.{block_index}.{name}.weight"
        if weight_key not in state_dict:
            raise KeyError(f"Model checkpoint does not contain {weight_key}.")
        bank = (
            state_dict[weight_key]
            .detach()
            .cpu()
            .permute(0, 2, 3, 1)
            .tolist()
        )
        weights[name] = bank
        layers[name] = {
            "name": f"blocks.{block_index}.{name}",
            "weight_key": weight_key,
            "filter_count": len(bank),
            "kernel_height": len(bank[0]),
            "kernel_width": len(bank[0][0]),
            "channels": len(bank[0][0][0]),
            "has_bias": f"blocks.{block_index}.{name}.bias" in state_dict,
            "source": str(Path(MODEL_FOLDER).relative_to(ROOT)),
        }
    return {
        "index": block_index,
        "config": {"channels": int(config.get("channels", 0))},
        "layers": layers,
        "weights": weights,
    }


def first_residual_output(checkpoint, tensors):
    """Evaluate the trained stem and residual block 0 in inference mode."""
    state = checkpoint["state_dict"]
    config = checkpoint.get("config", {})
    if int(config.get("channels", 0)) != 48 or int(config.get("blocks", 0)) < 1:
        raise ValueError("The checkpoint must contain a 48-channel residual block 0.")

    expected = {
        "pegs": (24, 24, 2),
        "links": (24, 24, 8),
        "locations": (24, 24, 2),
    }
    inputs = {}
    for name, shape in expected.items():
        if name not in tensors:
            raise ValueError(f"Missing {name} input tensor.")
        value = torch.as_tensor(tensors[name], dtype=torch.float32)
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be a finite tensor with shape {shape}.")
        inputs[name] = value.unsqueeze(0).permute(0, 3, 1, 2).contiguous()

    def parameter(key):
        if key not in state:
            raise KeyError(f"Model checkpoint does not contain {key}.")
        return state[key].detach().cpu()

    def batch_norm(value, prefix):
        return F.batch_norm(
            value,
            parameter(prefix + ".running_mean"),
            parameter(prefix + ".running_var"),
            parameter(prefix + ".weight"),
            parameter(prefix + ".bias"),
            training=False,
            momentum=0.1,
            eps=float(config.get("bn_eps", 1e-3)),
        )

    with torch.inference_mode():
        pegs = inputs["pegs"]
        links = F.pad(inputs["links"], (1, 2, 1, 2))
        locations = inputs["locations"]

        # Exact stem from TorchTwixtNet.forward(): sum the three branches,
        # apply primary BN and abs, then retain this tensor for the skip path.
        stem = (
            F.conv2d(locations, parameter("location.weight")) +
            F.conv2d(pegs, parameter("pegs.weight"), padding=2) +
            F.conv2d(links, parameter("links.weight"))
        )
        x = torch.abs(batch_norm(stem, "primary_bn"))

        h = F.conv2d(x, parameter("blocks.0.conv1.weight"), padding=2)
        h = torch.abs(batch_norm(h, "blocks.0.bn1"))
        h = F.conv2d(h, parameter("blocks.0.conv2.weight"), padding=2)
        h = torch.abs(batch_norm(h, "blocks.0.bn2"))
        output = h + x

        output = output[0].permute(1, 2, 0).contiguous()
        return {
            "shape": list(output.shape),
            "values": output.tolist(),
            "min": float(output.min()),
            "max": float(output.max()),
        }


class Handler(SimpleHTTPRequestHandler):
    # Keep local browser fetches on one TCP connection instead of paying a new
    # connection setup for every step through a long game. Every POST response
    # below includes Content-Length, as required for persistent HTTP/1.1.
    protocol_version = "HTTP/1.1"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VISUALIZER), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        endpoint = parsed.path
        if endpoint == "/api/residual-block":
            try:
                values = parse_qs(parsed.query).get("index", [])
                if len(values) != 1:
                    raise ValueError("Exactly one residual block index is required.")
                block_index = int(values[0])
                payload = residual_block_payload(load_checkpoint(), block_index)
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
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
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            return
        if endpoint != "/api/weights":
            return super().do_GET()
        try:
            checkpoint = load_checkpoint()
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

            # Expose both convolutions from the first residual block. The
            # checkpoint stores Conv2d weights as [out, in, height, width];
            # the browser renderer consumes [filter, row, column, channel].
            config = checkpoint.get("config", {})
            if int(config.get("blocks", 0)) < 1:
                raise ValueError("The checkpoint does not contain a residual block.")
            residual_conv1_layer, residual_conv1_weights = read_conv_layer("blocks.0.conv1")
            residual_conv2_layer, residual_conv2_weights = read_conv_layer("blocks.0.conv2")

            # The first three convolution branches are summed before this
            # BatchNorm2d layer, then the model applies torch.abs().
            bn_prefix = "primary_bn."
            bn_keys = {
                "weight": bn_prefix + "weight",
                "bias": bn_prefix + "bias",
                "running_mean": bn_prefix + "running_mean",
                "running_var": bn_prefix + "running_var",
            }
            missing_bn = [key for key in bn_keys.values() if key not in state_dict]
            if missing_bn:
                raise KeyError("Model checkpoint is missing primary batch-normalization values: " + ", ".join(missing_bn))
            primary_bn = {
                name: state_dict[key].detach().cpu().tolist()
                for name, key in bn_keys.items()
            }
            primary_bn["eps"] = 1e-3

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
                "primary_bn": primary_bn,
                "residual_block_count": int(config.get("blocks", 0)),
                "residual_block_0": {
                    "index": 0,
                    "config": {"channels": int(config.get("channels", 0))},
                    "layers": {
                        "conv1": residual_conv1_layer,
                        "conv2": residual_conv2_layer,
                    },
                    "weights": {
                        "conv1": residual_conv1_weights,
                        "conv2": residual_conv2_weights,
                    },
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
        if endpoint not in ("/api/trace", "/api/t1", "/api/residual-output"):
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
            if endpoint == "/api/residual-output":
                payload = json.loads(raw)
                result = first_residual_output(load_checkpoint(), payload.get("tensors", {}))
                moves = []
            elif endpoint == "/api/t1":
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
