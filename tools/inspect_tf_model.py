#! /usr/bin/env python

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

MODEL = "model/pb"

with tf.Session() as sess:
    tf.saved_model.loader.load(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        MODEL,
    )

    graph = tf.get_default_graph()

    print("=" * 100)
    print("INPUT / OUTPUT TENSORS")
    print("=" * 100)

    for name in (
        "pegx:0",
        "linkx:0",
        "locx:0",
        "is_training:0",
        "pwin:0",
        "movelogits:0",
    ):
        tensor = graph.get_tensor_by_name(name)
        print(
            f"{name:20} "
            f"shape={tensor.shape} "
            f"dtype={tensor.dtype}"
        )

    print()
    print("=" * 100)
    print("GRAPH OPERATIONS")
    print("=" * 100)

    for op in graph.get_operations():
        outputs = ", ".join(
            f"{tensor.name}:{tensor.shape}"
            for tensor in op.outputs
        )

        inputs = ", ".join(
            f"{tensor.name}:{tensor.shape}"
            for tensor in op.inputs
        )

        print()
        print(f"{op.name}")
        print(f"  type:    {op.type}")
        print(f"  inputs:  {inputs}")
        print(f"  outputs: {outputs}")

    print()
    print("=" * 100)
    print("TRAINABLE VARIABLES")
    print("=" * 100)

    for variable in tf.trainable_variables():
        print(
            f"{variable.name:60} "
            f"{variable.shape}"
        )
