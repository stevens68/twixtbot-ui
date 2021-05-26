import os
import logging
import constants as ct

# Suppress Tensorflow info messages and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import tensorflow.compat.v1 as tf  # noqa E402
tf.get_logger().setLevel(ct.K_LOG_LEVEL[3])
tf.disable_v2_behavior()


class NNEvaluater:

    def __init__(self, model):

        export_dir = os.path.join(os.getcwd(), model)
        self.logger = logging.getLogger(ct.LOGGER)
        self.sess = tf.Session()
        tf.get_logger().setLevel(self.logger.getEffectiveLevel())
        tf.saved_model.loader.load(
            sess=self.sess,
            tags=[tf.saved_model.tag_constants.SERVING],
            export_dir=export_dir)
        graph = tf.get_default_graph()

        self.pegx_t = graph.get_tensor_by_name("pegx:0")
        self.linkx_t = graph.get_tensor_by_name("linkx:0")
        self.locx_t = graph.get_tensor_by_name("locx:0")
        self.is_training_t = graph.get_tensor_by_name("is_training:0")

        self.pwin_t = graph.get_tensor_by_name("pwin:0")
        self.movelogits_t = graph.get_tensor_by_name("movelogits:0")

        self.use_recents = (int(self.locx_t.shape[3]) == 3)

    def eval_one(self, nip):

        pegs, links, locs = nip.to_input_arrays(self.use_recents)

        feed_dict = {
            self.pegx_t: [pegs],
            self.linkx_t: [links],
            self.locx_t: [locs],
            self.is_training_t: None
        }

        return self.sess.run(
            [self.pwin_t, self.movelogits_t],
            feed_dict=feed_dict
        )
