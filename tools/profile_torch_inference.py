#! /usr/bin/env python

"""Profile PyTorch inference for the converted Twixt model on CPU."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.backend.torchmodel import TwixtNet


def load_model(model_path):
    checkpoint = torch.load(
        model_path, map_location="cpu", weights_only=True
    )
    model = TwixtNet(**checkpoint["config"]).eval()
    model.load_state_dict(checkpoint["state_dict"])
    return model


def make_inputs(model, batch_size):
    loc_channels = 3 if model.use_recents else 2
    return (
        torch.randn(batch_size, 24, 24, 2),
        torch.randn(batch_size, 24, 24, 8),
        torch.randn(batch_size, 24, 24, loc_channels),
    )


def benchmark(model, inputs, warmup, iterations):
    with torch.inference_mode():
        for _ in range(warmup):
            model(*inputs)

        started = time.perf_counter()
        for _ in range(iterations):
            model(*inputs)
        elapsed = time.perf_counter() - started

    return elapsed / iterations


def benchmark_parts(model, np_inputs, warmup, iterations):
    inputs = [torch.from_numpy(x).unsqueeze(0).float() for x in np_inputs]
    with torch.inference_mode():
        for _ in range(warmup):
            model(*inputs)
        started = time.perf_counter()
        for _ in range(iterations):
            [torch.from_numpy(x).unsqueeze(0).float() for x in np_inputs]
        input_time = (time.perf_counter() - started) / iterations
        started = time.perf_counter()
        for _ in range(iterations):
            pwin, movelogits = model(*inputs)
        forward_time = (time.perf_counter() - started) / iterations
        started = time.perf_counter()
        for _ in range(iterations):
            pwin.cpu().numpy()
            movelogits.cpu().numpy()
        output_time = (time.perf_counter() - started) / iterations
    return input_time, forward_time, output_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parents[1] / "model" / "torch.pt"),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    print("PyTorch:", torch.__version__)
    print("CPU threads:", torch.get_num_threads())
    print("Interop threads:", torch.get_num_interop_threads())
    print("MKLDNN available:", torch.backends.mkldnn.is_available())
    print("MKLDNN enabled:", torch.backends.mkldnn.enabled)

    model = load_model(args.model)
    print("Model config:", model)
    print()

    for batch_size in (1, 2, 4, 8):
        inputs = make_inputs(model, batch_size)
        avg = benchmark(model, inputs, args.warmup, args.iterations)
        print(
            f"batch={batch_size:2d}  "
            f"{avg * 1000:8.3f} ms/inference  "
            f"{1.0 / avg:8.1f} inferences/s"
        )

    print()
    np_inputs = tuple(x.numpy() for x in make_inputs(model, 1))
    input_time, forward_time, output_time = benchmark_parts(
        model, np_inputs, args.warmup, args.iterations
    )
    print("Evaluator parts (batch=1):")
    print(f"input conversion:  {input_time * 1000:8.3f} ms")
    print(f"model forward:     {forward_time * 1000:8.3f} ms")
    print(f"output conversion: {output_time * 1000:8.3f} ms")
    print(f"sum:               {(input_time + forward_time + output_time) * 1000:8.3f} ms")

    print()
    print("Thread-count sweep (batch=1):")
    original_threads = torch.get_num_threads()

    for threads in (1, 2, 4, 6, 8):
        if threads > torch.get_num_threads() and threads > 8:
            continue
        torch.set_num_threads(threads)
        inputs = make_inputs(model, 1)
        avg = benchmark(model, inputs, args.warmup, args.iterations)
        print(
            f"threads={threads:2d}  "
            f"{avg * 1000:8.3f} ms/inference  "
            f"{1.0 / avg:8.1f} inferences/s"
        )

    torch.set_num_threads(original_threads)

    print()
    print("Note: random inputs are used; this measures model runtime,")
    print("not TF/PyTorch numerical parity.")


if __name__ == "__main__":
    main()
