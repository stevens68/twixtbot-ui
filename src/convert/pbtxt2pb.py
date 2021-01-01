import tensorflow.compat.v1 as tf


with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(
        sess, [tf.saved_model.SERVING], '../../model/pbtxt')
    graph = tf.get_default_graph()

    builder = tf.saved_model.builder.SavedModelBuilder('../../model/pb')
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.SERVING],
                                         strip_default_attrs=True)
    builder.save()
    print("model saved as *.pb file.")
