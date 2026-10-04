#! /usr/bin/env python

"""Convert the TensorFlow SavedModel weights into a PyTorch checkpoint."""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow.compat.v1 as tf  # noqa: E402
import torch  # noqa: E402

from src.backend.torchmodel import TwixtNet  # noqa: E402


def values(sess):
    return {
        v.name.split(":")[0]: sess.run(v)
        for v in tf.global_variables()
    }


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
        use_recents = int(locx.shape[3]) == 3
        value_triple = int(pwin.shape[1]) == 3

    net = TwixtNet(
        use_recents=use_recents,
        value_triple=value_triple,
    )

    conv(net.location, data, "primary_location/Variable")
    conv(net.pegs, data, "primary_pegs/Variable")
    conv(net.links, data, "primary_links/Variable")
    bn(net.primary_bn, data, "BatchNorm")

    for i, block in enumerate(net.blocks):
        bn1 = f"BatchNorm_{1 + 2 * i}"
        bn2 = f"BatchNorm_{2 + 2 * i}"
        conv(block.conv1, data, f"block{i}/Variable")
        bn(block.bn1, data, bn1)
        conv(block.conv2, data, f"block{i}/Variable_1")
        bn(block.bn2, data, bn2)

    # Value head is created before the policy head in mkbig.py.
    bn(net.value_bn[0], data, "BatchNorm_25")
    bn(net.value_bn[1], data, "BatchNorm_26")
    bn(net.value_bn_fc, data, "BatchNorm_27")
    conv(net.value_conv[0], data, "pwin/Variable")
    conv(net.value_conv[1], data, "pwin/Variable_1")
    linear(net.value_fc, data, "pwin/Variable_2")
    linear(net.value_out, data, "pwin/Variable_3")

    bn(net.policy_bn, data, "BatchNorm_28")
    conv(net.policy_conv1, data, "movelogits/Variable")
    conv(net.policy_conv2, data, "movelogits/Variable_1")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(
        {
            "config": {
                "use_recents": use_recents,
                "value_triple": value_triple,
            },
            "state_dict": net.state_dict(),
        },
        args.output,
    )
    print(
        f"wrote {args.output} "
        f"(recent_moves={use_recents}, value_triple={value_triple})"
    )


if __name__ == "__main__":
    main()    conv(net.location, data, "primary_location/Variable")
    conv(net.pegs, data, "primary_pegs/Variable")
    conv(net.links, data, "primary_links/Variable")
    bn(net.primary_bn, data, "primary/BatchNorm")

    for i, block in enumerate(net.blocks):
        scope = f"block{i}"
        conv(block.conv1, data, f"{scope}/Variable")
        bn(block.bn1, data, f"{scope}/BatchNorm")
        conv(block.conv2, data, f"{scope}/Variable_1")
        bn(block.bn2, data, f"{scope}/BatchNorm_1")

    # mkbig.py creates the value head before the policy head.
    conv(net.value_conv[0], data, "pwin/Variable")
    bn(net.value_bn[0], data, "pwin/BatchNorm")
    conv(net.value_conv[1], data, "pwin/Variable_1")
    bn(net.value_bn[1], data, "pwin/BatchNorm_1")
    linear(net.value_fc, data, "pwin/Variable_2")
    bn(net.value_bn_fc, data, "pwin/BatchNorm_2")
    linear(net.value_out, data, "pwin/Variable_3")

    conv(net.policy_conv1, data, "movelogits/Variable")
    bn(net.policy_bn, data, "movelogits/BatchNorm")
    conv(net.policy_conv2, data, "movelogits/Variable_1")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(
        {
            "config": {
                "use_recents": use_recents,
                "value_triple": value_triple,
            },
            "state_dict": net.state_dict(),
        },
        args.output,
    )
    print(
        f"wrote {args.output} "
        f"(recent_moves={use_recents}, value_triple={value_triple})"
    )


if __name__ == "__main__":
    main()
