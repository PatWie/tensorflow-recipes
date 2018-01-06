#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: PatWie <mail@patwie.com>

"""
Split-Brain Autoencoders: Unsupervised Learning by Cross-Channel Prediction
Richard Zhang et al. (https://arxiv.org/abs/1611.09842)
"""

import argparse
import tensorflow as tf
from tensorpack import * # noqa
import tensorpack.tfutils.symbolic_functions as symbf

BATCH_SIZE = 64
INPUT_SHAPE = 256


def lab2rgb(lab, stack=True):
    """Convert Lab input to RGB space.

    Args:
        lab (tf.tensor): lab representation of image
        stack (bool, optional): glue channels together

    Returns:
        tf.tensor: rgb image with values within range [0, 255]
    """
    l, a, b = tf.unstack(lab, 3, axis=3)

    y = (l + 16.) / 116.
    x = a / 500. + y
    z = y - b / 200.

    def scale0(x):
        return tf.where(x * x * x > 0.008856, x * x * x, (x - 16. / 116.) / 7.787)

    x = 0.95047 * scale0(x)
    y = 1.00000 * scale0(y)
    z = 1.08883 * scale0(z)

    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570

    def scale1(x):
        return tf.where(x > 0.0031308, (1.055 * (x ** (1 / 2.4)) - 0.055), 12.92 * x)

    r = tf.clip_by_value(scale1(r), 0., 1.)
    g = tf.clip_by_value(scale1(g), 0., 1.)
    b = tf.clip_by_value(scale1(b), 0., 1.)

    if stack:
        rgb = tf.stack((r, g, b), axis=3) * 255
    else:
        rgb = (r, g, b)
    return rgb


def rgb2lab(rgb, stack=True):
    """Convert RGB input to Lab space.

    l: 0 to 100
    a: -127 to 127
    b: -127 to 127

    Args:
        rgb (tf.Tensor): rgb image with values within range [0, 255]
        stack (bool, optional): glue all channels together

    Returns:
        tf.tensor: image in Lab colorspace
    """

    rgb = rgb / 255.
    r, g, b = tf.unstack(rgb, 3, axis=3)

    def scale0(x):
        return tf.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)

    r = scale0(r)
    g = scale0(g)
    b = scale0(b)

    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

    def scale1(x):
        return tf.where(x > 0.008856, x**(1. / 3.), (7.787 * x) + 16. / 116.)

    x = scale1(x)
    y = scale1(y)
    z = scale1(z)

    l = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)

    if stack:
        lab = tf.stack((l, a, b), axis=3)
    else:
        lab = (l, a, b)
    return lab


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'rgb')]

    def _build_graph(self, inputs):

        def net(x, scope_name, out_ch=3):
            # kind of alexnet but with batchnorm
            with tf.variable_scope(scope_name):
                with argscope(BatchNorm, use_local_stat=True), argscope(Dropout, is_training=True):
                    with argscope(Deconv2D, nl=tf.nn.relu, kernel_shape=4, stride=2):
                        return (LinearWrap(x)
                                .Conv2D('conv1', 96, 11, stride=4, padding='VALID')
                                .MaxPooling('pool1', 3, stride=2)
                                .Conv2D('conv2', 256, 5, stride=1, padding='VALID')
                                .MaxPooling('pool2', 3, stride=2)
                                .Conv2D('conv3', 384, 3)
                                .Conv2D('conv4', 384, 3)
                                .Conv2D('conv5', 256, 3)
                                .Conv2D('result', out_ch, 1, nl=tf.tanh)())

        rgb = inputs[0]

        # convert to LAB
        lab = rgb2lab(rgb)
        l = tf.expand_dims(lab[:, :, :, 0], 3)  # [0, 100]
        ab = lab[:, :, :, 1:2]  # [-127, 127]

        l_expected = tf.image.resize_images(l, (12, 12))
        ab_expected = tf.image.resize_images(ab, (12, 12))

        # encoder part + mapping from  [0,100]-50 --> ([-1, 1] + 1) * 50
        l_prediction = (net(l - 50, "l_part", out_ch=1) + 1.) * 50.
        # [-127, 127] -> [-1, 1] * 127
        ab_prediction = net(ab, "ab_part", out_ch=2) * 127.

        # loss
        loss_1 = tf.nn.l2_loss(l_expected - l_prediction, name="l_loss")
        loss_2 = tf.nn.l2_loss(ab_expected - ab_prediction, name="ab_loss")
        self.cost = tf.add(loss_1, loss_2, name='total_loss')
        summary.add_moving_summary(loss_1, loss_2, self.cost)

        # visualization
        lab_prediction = tf.concat((l_prediction, ab_prediction), 3)
        rgb_prediction = lab2rgb(lab_prediction)

        rgb_expected = tf.image.resize_images(rgb, (12, 12))
        viz = tf.concat([rgb_expected, rgb_prediction], 2)
        viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
        tf.summary.image('rgb, rgb_pred', viz, max_outputs=max(30, BATCH_SIZE))

        viz2 = tf.concat([l_expected, l_prediction], 2) / 100. * 255.
        viz2 = tf.cast(tf.clip_by_value(viz2, 0, 255), tf.uint8, name='viz2')
        tf.summary.image('l, l_pred', viz2, max_outputs=max(30, BATCH_SIZE))

    def _get_optimizer(self):
        lr = symbf.get_scalar_var('learning_rate', 1e-4, summary=True)
        return tf.train.GradientDescentOptimizer(lr)


def get_data(train_or_test):
    # return FakeData([[64, 224,224,3],[64]], 1000, random=False, dtype='uint8')
    isTrain = train_or_test == 'train'
    ds = dataset.ILSVRC12(args.data, train_or_test,
                          shuffle=True if isTrain else False, dir_structure='original')
    # drop label and BGR -> RGB
    ds = MapData(ds, lambda d: [d[0][:, :, [2, 1, 0]]])
    # resize to 256x256
    ds = AugmentImageComponent(ds, [imgaug.Resize((INPUT_SHAPE, INPUT_SHAPE))])
    # ensure parallel pre-fetching
    ds = PrefetchDataZMQ(ds, 5)
    ds = BatchData(ds, BATCH_SIZE)
    return ds


def get_config():
    logger.auto_set_dir()
    dataset = get_data("train")
    return TrainConfig(
        model=Model(),
        dataflow=dataset,
        callbacks=[ModelSaver()],
        steps_per_epoch=1000,
        max_epoch=100,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir', default='/scratch/imagenet/')
    parser.add_argument('--load', help='load model', default='')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)

    with change_gpu(args.gpu):
        nr_gpu = get_nr_gpu()
        config.nr_tower = nr_gpu
        SyncMultiGPUTrainer(config).train()
