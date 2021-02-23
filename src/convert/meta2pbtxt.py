import tensorflow.compat.v1 as tf
import os

# Python 2.7 only!
# this script reads ./six-917000 and writes ./newmodel/saved_model.pbtxt + ./newmodel/variables
# make sure ./newmodel exists and is empty

# FIX nodes in saved_model.pbtxt:
# manually edit saved_model.pvtxt and adjust 94 nodes named *FusedBatchNormGrad as follows
# * check if _output_shapes.list.shape #3 and #4 dim[0] have a size property
# * if so, remove the size property from dim[0]
# note: half of the nodes (47) have to be adjusted, the other half is fine.


loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Restore from checkpoint
    loader = tf.train.import_meta_graph('./six-917000.meta')
    loader.restore(sess, './six-917000')

    # Export checkpoint to SavedModel
    builder = tf.saved_model.builder.SavedModelBuilder('./newmodel')
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.SERVING],
                                         strip_default_attrs=True)
    builder.save(as_text=True)
