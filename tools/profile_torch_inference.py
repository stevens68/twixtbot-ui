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


def benchmark_sections(model, inputs, warmup, iterations):
    pegs = inputs[0].permute(0, 3, 1, 2)
    links = inputs[1].permute(0, 3, 1, 2)
    locs = inputs[2].permute(0, 3, 1, 2)

    def forward_sections():
        links_padded = torch.nn.functional.pad(links, (1, 2, 1, 2))
        h = model.location(locs) + model.pegs(pegs) + model.links(links_padded)
        h = torch.abs(model.primary_bn(h))

        for block in model.blocks:
            h = block(h)

        policy = torch.abs(model.policy_bn(model.policy_conv1(h)))
        policy = model.policy_conv2(policy)
        policy = policy[:, :, 1:-1, :].reshape(policy.shape[0], -1)

        v = h
        for conv, bn in zip(model.value_conv, model.value_bn):
            if model.value_padding == "SAME":
                height, width = v.shape[-2:]
                out_h = (height + 1) // 2
                out_w = (width + 1) // 2
                pad_h = max((out_h - 1) * 2 + 5 - height, 0)
                pad_w = max((out_w - 1) * 2 + 5 - width, 0)
                v = torch.nn.functional.pad(
                    v,
                    (
                        pad_w // 2,
                        pad_w - pad_w // 2,
                        pad_h // 2,
                        pad_h - pad_h // 2,
                    ),
                )
            v = torch.abs(bn(conv(v)))

        v = v.permute(0, 2, 3, 1).flatten(1)
        v = torch.abs(model.value_bn_fc(model.value_fc(v)))
        model.value_out(v)

    with torch.inference_mode():
        for _ in range(warmup):
            forward_sections()

        def timed(fn):
            started = time.perf_counter()
            for _ in range(iterations):
                fn()
            return (time.perf_counter() - started) / iterations

        primary = timed(
            lambda: (
                model.location(locs)
                + model.pegs(pegs)
                + model.links(
                    torch.nn.functional.pad(links, (1, 2, 1, 2))
                )
            )
        )

        h = model.location(locs) + model.pegs(pegs) + model.links(
            torch.nn.functional.pad(links, (1, 2, 1, 2))
        )
        h = torch.abs(model.primary_bn(h))

        residual = timed(
            lambda: run_residual_blocks(model.blocks, h)
        )

        h = run_residual_blocks(model.blocks, h)
        policy = timed(
            lambda: run_policy(model, h)
        )
        value = timed(
            lambda: run_value(model, h)
        )

    return primary, residual, policy, value


def run_residual_blocks(blocks, h):
    for block in blocks:
        h = block(h)
    return h


def run_policy(model, h):
    policy = torch.abs(model.policy_bn(model.policy_conv1(h)))
    policy = model.policy_conv2(policy)
    return policy[:, :, 1:-1, :].reshape(policy.shape[0], -1)


def run_value(model, h):
    v = h
    for conv, bn in zip(model.value_conv, model.value_bn):
        if model.value_padding == "SAME":
            height, width = v.shape[-2:]
            out_h = (height + 1) // 2
            out_w = (width + 1) // 2
            pad_h = max((out_h - 1) * 2 + 5 - height, 0)
            pad_w = max((out_w - 1) * 2 + 5 - width, 0)
            v = torch.nn.functional.pad(
                v,
                (
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h // 2,
                    pad_h - pad_h // 2,
                ),
            )
        v = torch.abs(bn(conv(v)))
    v = v.permute(0, 2, 3, 1).flatten(1)
    v = torch.abs(model.value_bn_fc(model.value_fc(v)))
    return model.value_out(v)


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
    section_inputs = make_inputs(model, 1)
    primary, residual, policy, value = benchmark_sections(
        model, section_inputs, args.warmup, args.iterations
    )
    print("Model sections (batch=1):")
    print(f"primary:   {primary * 1000:8.3f} ms")
    print(f"residual:  {residual * 1000:8.3f} ms")
    print(f"policy:    {policy * 1000:8.3f} ms")
    print(f"value:     {value * 1000:8.3f} ms")

    print()
    np_inputs = tuple(x.numpy()[0] for x in make_inputs(model, 1))
    input_time, forward_time, output_time = benchmark_parts(
        model, np_inputs, args.warmup, args.iterations
    )
    print("Evaluator parts (batch=1):")
    print(f"input conversion:  {input_time * 1000:8.3f} ms")
    print(f"model forward:     {forward_time * 1000:8.3f} ms")
    print(f"output conversion: {output_time * 1000:8.3f} ms")
    print(f"sum:               {(input_time + forward_time + output_time) * 1000:8.3f} ms")

    print()
    print("Channels-last benchmark (batch=1):")
    inputs = make_inputs(model, 1)
    channels_last_model = model.to(memory_format=torch.channels_last)
    channels_last_inputs = tuple(
        x.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        for x in inputs
    )
    avg = benchmark(channels_last_model, channels_last_inputs, args.warmup, args.iterations)
    print(
        f"channels_last  {avg * 1000:8.3f} ms/inference  "
        f"{1.0 / avg:8.1f} inferences/s"
    )
    channels_last_model.to(memory_format=torch.contiguous_format)

    print()
    print("torch.compile benchmark (batch=1):")
    inputs = make_inputs(model, 1)
    try:
        compiled = torch.compile(model)
        avg = benchmark(compiled, inputs, args.warmup, args.iterations)
        print(
            f"compiled  {avg * 1000:8.3f} ms/inference  "
            f"{1.0 / avg:8.1f} inferences/s"
        )
    except Exception as exc:
        print(f"torch.compile failed: {type(exc).__name__}: {exc}")

    print()
    print("MKLDNN sweep (batch=1):")
    original_mkldnn = torch.backends.mkldnn.enabled
    mkldnn_results = []
    for enabled in (True, False):
        torch.backends.mkldnn.enabled = enabled
        inputs = make_inputs(model, 1)
        avg = benchmark(model, inputs, args.warmup, args.iterations)
        mkldnn_results.append((enabled, avg))
        print(
            f"mkldnn={str(enabled):5s}  "
            f"{avg * 1000:8.3f} ms/inference  "
            f"{1.0 / avg:8.1f} inferences/s"
        )
    torch.backends.mkldnn.enabled = original_mkldnn

    print()
    print("Residual block sweep (batch=1):")
    block_inputs = make_inputs(model, 1)
    block_pegs = block_inputs[0].permute(0, 3, 1, 2)
    block_links = torch.nn.functional.pad(
        block_inputs[1].permute(0, 3, 1, 2), (1, 2, 1, 2)
    )
    block_locs = block_inputs[2].permute(0, 3, 1, 2)
    with torch.inference_mode():
        block_h = torch.abs(
            model.primary_bn(
                model.location(block_locs)
                + model.pegs(block_pegs)
                + model.links(block_links)
            )
        )
        for index, block in enumerate(model.blocks):
            for _ in range(args.warmup):
                block(block_h)
            started = time.perf_counter()
            for _ in range(args.iterations):
                block(block_h)
            elapsed = (time.perf_counter() - started) / args.iterations
            print(f"block={index:2d}  {elapsed * 1000:8.3f} ms")

    print()
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
