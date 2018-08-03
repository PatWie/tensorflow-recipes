# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

import os
import argparse
import tensorflow as tf

from tensorpack import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils.scope_utils import under_name_scope


enable_argscope_for_module(tf.layers)

"""
Re-implementation of
SSH: Single Stage Headless Face Detector

<https://arxiv.org/abs/1708.03979>
"""

BATCH_SIZE = 16
SHAPE = 28
CHANNELS = 3


def upsample(x, factor=2):
    _, h, w, _ = x.get_shape().as_list()
    x = tf.image.resize_bilinear(x, [factor * h, factor * w])
    return x


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, (None, SHAPE, SHAPE, CHANNELS), 'input1'),
                tf.placeholder(tf.int32, (None,), 'input2')]

    @under_name_scope
    def detection_module(self, x, channels, name):
        # see Figure 3 (SSH Detection Module)
        yc = self.context_module(x, channels, 'context_%s' % name)

        with argscope([tf.layers.conv2d], padding='same'):
            y = tf.layers.conv2d(x, channels, kernel_size=3, activation=tf.nn.relu, name='conv1')
            y = tf.concatenate([yc, y], axis=-1)
            logits = tf.layers.conv2d(x, 2, kernel_size=1, activation=tf.identity, name='conv2')
            reg = tf.layers.conv2d(x, 8, kernel_size=1, activation=tf.identity, name='conv2')

        return logits, reg

    @under_name_scope
    def context_module(self, x, channels, name):
        # see Figure 4 (SSH Context Module)
        with tf.variable_scope(name):
            with argscope([tf.layers.conv2d], kernel_size=3, activation=tf.nn.relu, padding='same'):
                c1 = tf.layers.conv2d(x, channels // 2, name='conv1')
                # upper path
                c2 = tf.layers.conv2d(c1, channels // 2, name='conv2')
                # lower path
                c3 = tf.layers.conv2d(c1, channels // 2, name='conv3a')
                c3 = tf.layers.conv2d(c3, channels // 2, name='conv3b')
                return tf.concatenate([c2, c3], axis=-1)

    @under_name_scope
    def vgg16(self, x):
        with argscope([tf.layers.conv2d], kernel_size=3, activation=tf.nn.relu, padding='same'):
            x = tf.layers.conv2d(x, 64, name='conv1_1')
            x = tf.layers.conv2d(x, 64, name='conv1_2')
            x = tf.layers.max_pooling2d(x, 2, 2, name='pool1')

            x = tf.layers.conv2d(x, 128, name='conv2_1')
            x = tf.layers.conv2d(x, 128, name='conv2_2')
            x = tf.layers.max_pooling2d(x, 2, 2, name='pool2')

            x = tf.layers.conv2d(x, 256, name='conv3_1')
            x = tf.layers.conv2d(x, 256, name='conv3_2')
            x = tf.layers.conv2d(x, 256, name='conv3_3')
            x = tf.layers.max_pooling2d(x, 2, 2, name='pool3')

            x = tf.layers.conv2d(x, 512, name='conv4_1')
            x = tf.layers.conv2d(x, 512, name='conv4_2')
            c43 = tf.layers.conv2d(x, 512, name='conv4_3')
            x = tf.layers.max_pooling2d(c43, 2, 2, name='pool4')

            x = tf.layers.conv2d(x, 512, name='conv5_1')
            x = tf.layers.conv2d(x, 512, name='conv5_2')
            c53 = tf.layers.conv2d(x, 512, name='conv5_3')

            return c43, c53

    @under_name_scope
    def rpn_reg_loss(self, actual, expected, anchors, epsilon=1e-4):
        diff = actual - expected
        x = tf.abs(diff)
        alpha = tf.less_equal(x, 1.0)

        cost = tf.reduce_sum(expected * alpha * (0.5 * diff * diff) + (1 - alpha) * (x - 0.5))
        cost /= tf.reduce_sum(eps + expected)
        return cost

    def build_graph(self, img, input2):

        c43, c53 = self.vvg16(img)

        p3 = tf.layers.max_pooling2d(c53, 2, 2, name='pool1')
        p3_logits, p3_reg = self.detection_module(p3, 512, 'm3')

        p2_logits, p2_reg = self.detection_module(c53, 512, 'm2')

        with argscope([tf.layers.conv2d], kernel_size=3, activation=tf.nn.relu, padding='same'):
            p1a = tf.layers.conv2d(c53, 128, kernel_size=1, name='conv_p1_1')
            p1a = upsample(p1a)
            p1b = tf.layers.conv2d(c43, 128, kernel_size=1, name='conv_p1_2')
            p1 = tf.add(p1a, p1b)
            p1 = tf.layers.conv2d(p1, 128, kernel_size=3, name='conv_p1_2')
            p1_logits, p1_reg = self.detection_module(p1, 128, 'm1')

        cost = tf.identity(input1 - input2, name='total_costs')
        summary.add_moving_summary(cost)
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=5e-3, trainable=False)
        summary.add_moving_summary(lr)
        return tf.train.AdamOptimizer(lr)


def get_data(subset):
    # something that yields [[SHAPE, SHAPE, CHANNELS], [1]]
    ds = FakeData([[SHAPE, SHAPE, CHANNELS], [1]], 1000, random=False,
                  dtype=['float32', 'uint8'], domain=[(0, 255), (0, 10)])
    ds = PrefetchDataZMQ(ds, 2)
    ds = BatchData(ds, BATCH_SIZE)
    return ds


def get_config():
    global BATCH
    nr_tower = max(get_nr_gpu(), 1)
    BATCH = TOTAL_BATCH_SIZE // nr_tower
    logger.set_logger_dir()

    ds_train = get_data('train')
    ds_test = get_data('test')

    return TrainConfig(
        model=Model(),
        data=QueueInput(ds_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(ds_test, [ScalarStats('total_costs')]),
        ],
        extra_callbacks=[
            MovingAverageSummary(),
            ProgressBar(['']),
            MergeAllSummaries(),
            RunUpdateOps()
        ],
        steps_per_epoch=ds_train.size(),
        max_epoch=100,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()

    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    if args.load:
        config.session_init = SaverRestore(args.load)

    trainer = SyncMultiGPUTrainerParameterServer(max(get_nr_gpu(), 1))
    launch_train_with_config(config, trainer)
