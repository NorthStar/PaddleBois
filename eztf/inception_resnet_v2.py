from os.path import join, isfile
import urllib
from datasets import dataset_utils

import numpy as np

from PIL import Image

import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope

class Inception_Resnet_V2(object):

    def __init__(self):
        url = "http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz"
        checkpoints_dir = '/tmp/checkpoints'
        checkpoints_filename = 'inception_resnet_v2_2016_08_30.ckpt'

        if not tf.gfile.Exists(checkpoints_dir):
            tf.gfile.MakeDirs(checkpoints_dir)

        if not tf.gfile.Exists(join(checkpoints_dir, checkpoints_filename)):
            dataset_utils.download_and_uncompress_tarball(url,
                checkpoints_dir)

        self.checkpoints_filename = join(
                checkpoints_dir,
                checkpoints_filename)

        self.model_name = 'InceptionResnetV2'

    def run(self, images):
        sess = tf.InteractiveSession()

        input_tensor = tf.placeholder(tf.float32, shape=(None,299,299,3), name='input_image')
        scaled_input_tensor = tf.scalar_mul((1.0/255), input_tensor)
        scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
        scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)

        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(scaled_input_tensor, num_classes=1001, is_training=False)

        saver = tf.train.Saver()
        saver.restore(sess, self.checkpoints_filename)

        for image in images:
            im = Image.open(image).resize((299,299))
            im = np.array(im)
            im = im.reshape(-1, 299, 299, 3)

            predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_tensor: im})
            print (np.max(predict_values), np.max(logit_values))
            print (np.argmax(predict_values), np.argmax(logit_values))

