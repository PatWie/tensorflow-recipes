# -*- coding: utf-8 -*-
# Author: Your Name <your@email.com>

import os
import argparse
import tensorflow as tf

from tensorpack import *

enable_argscope_for_module(tf.layers)

"""
Re-implementation of Learning to See in the Dark in by Chen et al.
<https://arxiv.org/abs/1805.01934>
"""

BATCH_SIZE = 2
SHAPE = 512


def visualize_images(name, imgs):
    """Generate tensor for TensorBoard (casting, clipping)
    Args:
        name: name for visualiaztion operation
        *imgs: multiple images in *args style
    Example:
        visualize_images('viz1', [img1])
        visualize_images('viz2', [img1, img2, img3])
    """
    xy = (tf.concat(imgs, axis=2)) * 256.
    xy = tf.cast(tf.clip_by_value(xy, 0, 255), tf.uint8, name='viz')
    tf.summary.image(name, xy, max_outputs=30)


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, (None, 2 * SHAPE, 2 * SHAPE, 3), 'long_exposure'),
                tf.placeholder(tf.float32, (None, SHAPE, SHAPE, 4), 'short_exposure')]

    def build_graph(self, long_expo, short_expo):

        # unet
        with argscope([tf.layers.conv2d], activation=tf.nn.leaky_relu, padding='same'):
            with argscope([tf.layers.conv2d_transpose], padding='valid', use_bias=False):
                conv1 = tf.layers.conv2d(short_expo, 32, 3, name='conv1_1')
                conv1 = tf.layers.conv2d(conv1, 32, 3, name='conv1_2')
                pool1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='same', name='maxpool1')

                conv2 = tf.layers.conv2d(pool1, 64, 3, name='conv2_1')
                conv2 = tf.layers.conv2d(conv2, 64, 3, name='conv2_2')
                pool2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='same', name='maxpool2')

                conv3 = tf.layers.conv2d(pool2, 128, 3, name='conv3_1')
                conv3 = tf.layers.conv2d(conv3, 128, 3, name='conv3_2')
                pool3 = tf.layers.max_pooling2d(conv3, 2, 2, padding='same', name='maxpool3')

                conv4 = tf.layers.conv2d(pool3, 256, 3, name='conv4_1')
                conv4 = tf.layers.conv2d(conv4, 256, 3, name='conv4_2')
                pool4 = tf.layers.max_pooling2d(conv4, 2, 2, padding='same', name='maxpool4')

                conv5 = tf.layers.conv2d(pool4, 512, 3, name='conv5_1')
                conv5 = tf.layers.conv2d(conv5, 512, 3, name='conv5_2')

                deconv5 = tf.layers.conv2d_transpose(conv5, 256, 2, strides=2, name='deconv5')
                up6 = tf.concat([conv4, deconv5], name='concat6', axis=-1)
                conv6 = tf.layers.conv2d(up6, 256, 3, name='conv6_1')
                conv6 = tf.layers.conv2d(conv6, 256, 3, name='conv6_2')

                deconv6 = tf.layers.conv2d_transpose(conv6, 128, 2, strides=2, name='deconv6')
                up7 = tf.concat([conv3, deconv6], name='concat7', axis=-1)
                conv7 = tf.layers.conv2d(up7, 128, 3, name='conv7_1')
                conv7 = tf.layers.conv2d(conv7, 128, 3, name='conv7_2')

                deconv7 = tf.layers.conv2d_transpose(conv7, 64, 2, strides=2, name='deconv7')
                up8 = tf.concat([conv2, deconv7], name='concat8', axis=-1)
                conv8 = tf.layers.conv2d(up8, 64, 3, name='conv8_1')
                conv8 = tf.layers.conv2d(conv8, 64, 3, name='conv8_2')

                deconv8 = tf.layers.conv2d_transpose(conv8, 32, 2, strides=2, name='deconv8')
                up9 = tf.concat([conv1, deconv8], name='concat9', axis=-1)
                conv9 = tf.layers.conv2d(up9, 32, 3, name='conv9_1')
                conv9 = tf.layers.conv2d(conv9, 32, 3, name='conv9_2')

                conv10 = tf.layers.conv2d(conv9, 12, 1, activation=None, name='conv10')
                out = tf.depth_to_space(conv10, 2)

        visualize_images('info', [out, long_expo])

        cost = tf.reduce_mean(tf.abs(out - long_expo), name='l1')
        summary.add_moving_summary(cost)
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)
        return tf.train.AdamOptimizer(lr)


def get_data(subset):
    from sony_dataset import SonyDataset, RandomCropRaw
    ds = SonyDataset('/scratch/wieschol/seeindark/dataset/Sony', subset=subset)
    ds = RandomCropRaw(ds)
    aus = [imgaug.Flip(horiz=True), imgaug.Flip(vert=True), imgaug.Transpose()]
    ds = AugmentImageComponents(ds, aus, index=(0, 1), copy=False)
    ds = PrefetchDataZMQ(ds, nr_proc=10)
    ds = BatchData(ds, 8)
    return ds


def get_config():
    logger.set_logger_dir('/scratch/wieschol/seeindark/train_log')

    ds_train = get_data('train')

    return TrainConfig(
        model=Model(),
        data=QueueInput(ds_train),
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter(
                'learning_rate', [(2000, 1e-5)]),
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

    launch_train_with_config(config, SimpleTrainer())
