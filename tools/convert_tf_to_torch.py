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


def numbered_names(data, pattern):
    def suffix(name):
        match = re.fullmatch(pattern, name)
        return 0 if match.group(1) is None else int(match.group(1))

    return sorted(
        (name for name in data if re.fullmatch(pattern, name)),
        key=suffix,
    )


def bn_scopes(data):
    """Return BN scopes in TensorFlow variable creation order.

    The SavedModel in use does not preserve the Python scopes from mkbig.py
    for these variables, so matching names such as primary/BatchNorm is not
    reliable. TensorFlow's global-variable order does preserve layer creation
    order, which is exactly the order needed for this architecture.
    """
    scopes = []
    seen = set()
    for name in data:
        if not name.endswith("/gamma"):
            continue
        scope = name[:-len("/gamma")]
        required = (
            scope + "/beta",
            scope + "/moving_mean",
            scope + "/moving_variance",
        )
        if scope not in seen and all(item in data for item in required):
            seen.add(scope)
            scopes.append(scope)
    return scopes


def infer_config(data, loc_channels, pwin_shape):
    channels = int(data["primary_location/Variable"].shape[-1])
    blocks = {
        int(match.group(1))
        for name in data
        if (match := re.fullmatch(r"block(\d+)/Variable", name))
    }
    if blocks and blocks != set(range(max(blocks) + 1)):
        raise ValueError(f"non-contiguous residual block names: {sorted(blocks)}")

    block_count = max(blocks) + 1 if blocks else 0

    pwin_weights = numbered_names(data, r"pwin/Variable(?:_(\d+))?")
    conv_names = [name for name in pwin_weights if data[name].ndim == 4]
    linear_names = [name for name in pwin_weights if data[name].ndim == 2]

    output_names = [
        name for name in linear_names if int(data[name].shape[1]) == int(pwin_shape[1])
    ]
    if len(output_names) != 1:
        raise ValueError(
            "could not identify value-head output weight; "
            f"2D pwin weights={[(n, data[n].shape) for n in linear_names]}"
        )
    value_out_name = output_names[0]

    fc_names = [name for name in linear_names if name != value_out_name]
    if len(fc_names) != 1:
        raise ValueError(
            "could not identify value-head FC weight; "
            f"2D pwin weights={[(n, data[n].shape) for n in linear_names]}"
        )
    value_fc_name = fc_names[0]

    value_reductions = len(conv_names)
    value_fc_shape = data[value_fc_name]
    value_hidden = int(value_fc_shape.shape[1])

    fc_inputs = int(value_fc_shape.shape[0])
    if fc_inputs % channels:
        raise ValueError(
            f"value FC input size {fc_inputs} is not divisible by channels {channels}"
        )
    spatial_sq = fc_inputs // channels
    spatial = int(spatial_sq ** 0.5)
    if spatial * spatial != spatial_sq:
        raise ValueError(f"unsupported value-head spatial size: {spatial_sq}")

    if value_reductions:
        valid_spatial = 24
        for _ in range(value_reductions):
            if valid_spatial < 5:
                valid_spatial = -1
                break
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
    return config, conv_names, value_fc_name, value_out_name


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
        config, value_conv_names, value_fc_name, value_out_name = infer_config(
            data, locx.shape[3], pwin.shape
        )

    net = TwixtNet(**config)

    all_bn_scopes = bn_scopes(data)
    expected_bn_count = 1 + 2 * config["blocks"] + config["value_reductions"] + 1 + 1
    if len(all_bn_scopes) != expected_bn_count:
        raise ValueError(
            f"expected {expected_bn_count} BatchNorm groups, "
            f"found {len(all_bn_scopes)}: {all_bn_scopes}"
        )

    pos = 0
    primary_bn_scope = all_bn_scopes[pos]
    pos += 1
    conv(net.location, data, "primary_location/Variable")
    conv(net.pegs, data, "primary_pegs/Variable")
    conv(net.links, data, "primary_links/Variable")
    bn(net.primary_bn, data, primary_bn_scope)

    for i, block in enumerate(net.blocks):
        scope = f"block{i}"
        conv(block.conv1, data, f"{scope}/Variable")
        bn(block.bn1, data, all_bn_scopes[pos])
        pos += 1
        conv(block.conv2, data, f"{scope}/Variable_1")
        bn(block.bn2, data, all_bn_scopes[pos])
        pos += 1

    for conv_layer, bn_layer, weight_name in zip(
        net.value_conv,
        net.value_bn,
        value_conv_names,
    ):
        conv(conv_layer, data, weight_name)
        bn(bn_layer, data, all_bn_scopes[pos])
        pos += 1

    linear(net.value_fc, data, value_fc_name)
    bn(net.value_bn_fc, data, all_bn_scopes[pos])
    pos += 1
    linear(net.value_out, data, value_out_name)

    conv(net.policy_conv1, data, "movelogits/Variable")
    bn(net.policy_bn, data, all_bn_scopes[pos])
    pos += 1
    conv(net.policy_conv2, data, "movelogits/Variable_1")

    if pos != len(all_bn_scopes):
        raise ValueError(f"unused BatchNorm groups: {all_bn_scopes[pos:]}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(
        {"config": config, "state_dict": net.state_dict()},
        args.output,
    )
    print(f"wrote {args.output} with config={config}")


if __name__ == "__main__":
    main()
