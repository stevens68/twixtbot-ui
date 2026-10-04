#! /usr/bin/env python

"""Convert a TensorFlow SavedModel into a PyTorch checkpoint."""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow.compat.v1 as tf  # noqa: E402
import torch  # noqa: E402

from src.backend.torchmodel import TwixtNet  # noqa: E402


def values(sess):
    return {v.name.split(":")[0]: sess.run(v) for v in tf.global_variables()}


def conv(module, data, name):
    module.weight.data.copy_(torch.from_numpy(data[name]).permute(3, 2, 0, 1))


def linear(module, data, name):
    module.weight.data.copy_(torch.from_numpy(data[name]).T)


def bn(module, data, scope):
    module.weight.data.copy_(torch.from_numpy(data[scope + "/gamma"]))
    module.bias.data.copy_(torch.from_numpy(data[scope + "/beta"]))
    module.running_mean.data.copy_(
        torch.from_numpy(data[scope + "/moving_mean"])
    )
    module.running_var.data.copy_(
        torch.from_numpy(data[scope + "/moving_variance"])
    )


def infer_config(data, loc_channels, pwin_shape):
    channels = int(data["primary_location/Variable"].shape[-1])
    blocks = {
        int(match.group(1))
        for name in data
        if (match := re.fullmatch(r"block(\\d+)/Variable", name))
    }
    if blocks and blocks != set(range(max(blocks) + 1)):
        raise ValueError(f"non-contiguous residual block names: {sorted(blocks)}")

    block_count = max(blocks) + 1 if blocks else 0
    value_fc_shape = data["pwin/Variable_2"].shape
    value_hidden = int(value_fc_shape[1])
    value_reductions = len([
        name for name in data
        if re.fullmatch(r"pwin/Variable(?:_\\d+)?", name)
    ]) - 2
    if value_reductions < 0:
        raise ValueError("could not infer value-head convolution count")

    fc_inputs = int(value_fc_shape[0])
    if fc_inputs % channels:
        raise ValueError(
            f"value FC input size {fc_inputs} is not divisible by channels {channels}"
        )
    spatial_sq = fc_inputs // channels
    spatial = int(spatial_sq ** 0.5)
    if spatial * spatial != spatial_sq:
        raise ValueError(f"unsupported value-head spatial size: {spatial_sq}")

    if value_reductions:
        # mkbig.py's 5x5/stride-2 VALID and SAME modes have distinct final sizes.
        valid_spatial = 24
        for _ in range(value_reductions):
            valid_spatial = (valid_spatial - 5) // 2 + 1
        same_spatial = 24
        for _ in range(value_reductions):
            same_spatial = (same_spatial + 1) // 2
        if spatial == valid_spatial:
            value_padding = "VALID"
        elif spatial == same_spatial:
            value_padding = "SAME"
        else:
            raise ValueError(
                f"cannot infer value padding from final spatial size {spatial}"
            )
    else:
        value_padding = "VALID"

    config = {
        "use_recents": int(loc_channels) == 3,
        "channels": channels,
        "blocks": block_count,
        "value_hidden": value_hidden,
        "value_triple": int(pwin_shape[1]) == 3,
        "value_reductions": value_reductions,
        "value_padding": value_padding,
    }
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model/pb")
    parser.add_argument("--output", default="model/torch.pt")
    args = parser.parse_args()

    tf.disable_v2_behavior()
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], args.model
        )
        data = values(sess)
        locx = graph.get_tensor_by_name("locx:0")
        pwin = graph.get_tensor_by_name("pwin:0")
        config = infer_config(data, locx.shape[3], pwin.shape)

    net = TwixtNet(**config)

    conv(net.location, data, "primary_location/Variable")
    conv(net.pegs, data, "primary_pegs/Variable")
    conv(net.links, data, "primary_links/Variable")
    bn(net.primary_bn, data, "primary/BatchNorm")

    for i, block in enumerate(net.blocks):
        scope = f"block{i}"
        conv(block.conv1, data, f"{scope}/Variable")
        bn(block.bn1, data, f"{scope}/BatchNorm")
        conv(block.conv2, data, f"{scope}/Variable_1")
        bn(block.bn2, data, f"{scope}/BatchNorm_1")

    for i, (conv_layer, bn_layer) in enumerate(zip(net.value_conv, net.value_bn)):
        suffix = "" if i == 0 else f"_{i}"
        conv(conv_layer, data, f"pwin/Variable{suffix}")
        bn(bn_layer, data, f"pwin/BatchNorm{suffix}")

    linear(net.value_fc, data, "pwin/Variable_2")
    bn(net.value_bn_fc, data, "pwin/BatchNorm_2")
    linear(net.value_out, data, "pwin/Variable_3")

    conv(net.policy_conv1, data, "movelogits/Variable")
    bn(net.policy_bn, data, "movelogits/BatchNorm")
    conv(net.policy_conv2, data, "movelogits/Variable_1")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(
        {"config": config, "state_dict": net.state_dict()},
        args.output,
    )
    print(f"wrote {args.output} with config={config}")


if __name__ == "__main__":
    main()
