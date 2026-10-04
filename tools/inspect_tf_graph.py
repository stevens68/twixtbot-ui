import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow.compat.v1 as tf


MODEL = "model/pb"


tf.disable_v2_behavior()

with tf.Session() as sess:
    tf.saved_model.loader.load(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        MODEL,
    )

    graph = tf.get_default_graph()

    print("\nNETWORK OPERATIONS")
    print("==================")

    for op in graph.get_operations():
        name = op.name

        # Only show the actual network, avoiding optimizer/training noise.
        if not (
            name.startswith("primary_")
            or name.startswith("block")
            or name.startswith("BatchNorm")
            or name.startswith("pwin")
            or name.startswith("movelogits")
        ):
            continue

        inputs = []
        for tensor in op.inputs:
            inputs.append(
                f"{tensor.name} {tensor.shape}"
            )

        outputs = []
        for tensor in op.outputs:
            outputs.append(
                f"{tensor.name} {tensor.shape}"
            )

        print(f"\n{name}")
        print(f"  type: {op.type}")

        if inputs:
            print("  inputs:")
            for x in inputs:
                print(f"    {x}")

        if outputs:
            print("  outputs:")
            for x in outputs:
                print(f"    {x}")
