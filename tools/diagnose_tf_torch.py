#! /usr/bin/env python

"""Diagnose TensorFlow/PyTorch layer parity for the converted Twixt model."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow.compat.v1 as tf  # noqa: E402
import torch  # noqa: E402

from src.backend.torchmodel import TwixtNet  # noqa: E402


MODEL = "model/pb"
REFERENCE = "reference/tf_reference.npz"
CHECKPOINT = "model/torch.pt"


def tf_conv_ops(graph, prefix):
    scope = prefix + "/"
    return [
        op for op in graph.get_operations()
        if op.type == "Conv2D" and op.name.startswith(scope)
    ]


def main():
    ref = np.load(REFERENCE)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    net = TwixtNet(**checkpoint["config"])
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()

    captured = {}

    def hook(name):
        def save(_module, _inputs, output):
            captured[name] = output.detach().cpu().numpy()
        return save

    net.location.register_forward_hook(hook("location"))
    net.pegs.register_forward_hook(hook("pegs"))
    net.links.register_forward_hook(hook("links"))
    for i, block in enumerate(net.blocks):
        block.conv1.register_forward_hook(hook(f"block{i}.conv1"))
        block.conv2.register_forward_hook(hook(f"block{i}.conv2"))
    net.policy_conv1.register_forward_hook(hook("policy.conv1"))
    net.policy_conv2.register_forward_hook(hook("policy.conv2"))
    for i, layer in enumerate(net.value_conv):
        layer.register_forward_hook(hook(f"value.conv{i}"))

    with torch.no_grad():
        net(
            torch.from_numpy(ref["pegs"]).float(),
            torch.from_numpy(ref["links"]).float(),
            torch.from_numpy(ref["locs"]).float(),
        )

    tf.disable_v2_behavior()
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], MODEL
        )

        tensors = {}
        groups = [
            ("primary_location", "location"),
            ("primary_pegs", "pegs"),
            ("primary_links", "links"),
        ]
        for i in range(checkpoint["config"]["blocks"]):
            groups.extend([
                (f"block{i}", f"block{i}.conv1"),
                (f"block{i}", f"block{i}.conv2"),
            ])
        groups.extend([
            ("pwin", None),
            ("movelogits", None),
        ])

        for prefix, torch_name in groups:
            ops = tf_conv_ops(graph, prefix)
            if prefix == "pwin":
                for i, op in enumerate(ops):
                    tensors[f"value.conv{i}"] = op.outputs[0]
            elif prefix == "movelogits":
                for i, op in enumerate(ops):
                    tensors[f"policy.conv{i + 1}"] = op.outputs[0]
            elif prefix.startswith("block"):
                if len(ops) != 2:
                    raise ValueError(
                        f"expected two Conv2D ops under {prefix}, found "
                        f"{[op.name for op in ops]}"
                    )
                tensors[torch_name] = ops[0].outputs[0]
                block_index = prefix[len("block"):]
                tensors[f"block{block_index}.conv2"] = ops[1].outputs[0]
            elif len(ops) != 1:
                raise ValueError(
                    f"expected one Conv2D under {prefix}, found "
                    f"{[op.name for op in ops]}"
                )
            else:
                tensors[torch_name] = ops[0].outputs[0]

        value_filters = [
            op.inputs[1] for op in tf_conv_ops(graph, "pwin")
        ]
        value_filter_values = sess.run(value_filters)

        values = sess.run(tensors, feed_dict={
            graph.get_tensor_by_name("pegx:0"): ref["pegs"],
            graph.get_tensor_by_name("linkx:0"): ref["links"],
            graph.get_tensor_by_name("locx:0"): ref["locs"],
            graph.get_tensor_by_name("is_training:0"): False,
        })

    print("Value-conv weight parity:")
    for i, tf_filter in enumerate(value_filter_values):
        torch_filter = net.value_conv[i].weight.detach().cpu().numpy()
        tf_oihw = np.transpose(tf_filter, (3, 2, 0, 1))
        diff = np.max(np.abs(torch_filter - tf_oihw))
        print(f"value.conv{i} weights max_abs={diff:.6g}")

    print()
    print("Layer convolution parity:")
    for name, torch_value in captured.items():
        tf_value = values[name]
        torch_nhwc = np.transpose(torch_value, (0, 2, 3, 1))
        diff = np.max(np.abs(torch_nhwc - tf_value))
        print(f"{name:20} max_abs={diff:.6g}")

    # Compare the actual model outputs too. This catches post-convolution
    # differences such as the policy crop/reshape and value-head BN/FC path.
    with torch.no_grad():
        torch_pwin, torch_policy = net(
            torch.from_numpy(ref["pegs"]).float(),
            torch.from_numpy(ref["links"]).float(),
            torch.from_numpy(ref["locs"]).float(),
        )

    print()
    print("Final output parity:")
    for name, torch_value, tf_name in (
        ("pwin", torch_pwin.numpy(), "pwin:0"),
        ("movelogits", torch_policy.numpy(), "movelogits:0"),
    ):
        tf_value = sess.run(graph.get_tensor_by_name(tf_name), feed_dict={
            graph.get_tensor_by_name("pegx:0"): ref["pegs"],
            graph.get_tensor_by_name("linkx:0"): ref["links"],
            graph.get_tensor_by_name("locx:0"): ref["locs"],
            graph.get_tensor_by_name("is_training:0"): False,
        })
        diff = np.max(np.abs(torch_value - tf_value))
        print(f"{name:20} max_abs={diff:.6g}")

    print()
    print("TensorFlow convolution ops:")
    for prefix, _ in groups:
        for op in tf_conv_ops(graph, prefix):
            print(f"{op.name:50} {op.outputs[0].shape}")


if __name__ == "__main__":
    main()
