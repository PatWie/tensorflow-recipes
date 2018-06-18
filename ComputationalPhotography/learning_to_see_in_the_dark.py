#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

import os
import argparse
import tensorflow as tf

from tensorpack import *
from tensorpack.utils.gpu import get_nr_gpu
from sony_dataset import SonyDataset, RandomCropRaw, CenterCropRaw

enable_argscope_for_module(tf.layers)

"""
Re-implementation of Learning to See in the Dark in by Chen et al.
<https://arxiv.org/abs/1805.01934>


train:
    python learning_to_see_in_the_dark.py --gpu 0

apply:

    python learning_to_see_in_the_dark.py --gpu 0 \
        --apply --load /scratch/wieschol/seeindark/train_log2/checkpoint \
        --short /scratch/wieschol/seeindark/dataset/Sony/short/00001_00_0.1s.ARW --exposure_time 10

    python learning_to_see_in_the_dark.py --gpu 0 \
        --apply --load /scratch/wieschol/seeindark/train_log2/checkpoint \
        --short "/scratch/wieschol/seeindark/dataset/Sony/short/1*_00_*.ARW" --exposure_time 10

"""

BATCH_SIZE = 1
SHAPE = 512
DATA_ROOT_DIR = '/scratch/wieschol/seeindark/dataset/Sony'
EPOCH_MULTIPLIER = 10


def visualize_images(name, imgs):
    """Generate tensor for TensorBoard (casting, clipping)
    Args:
        name: name for visualiaztion operation
        *imgs: multiple images in *args style
    Example:
        visualize_images('viz1', [img1])
        visualize_images('viz2', [img1, img2, img3])
    """
    xy = (tf.concat(imgs, axis=2)) * 255.
    xy = tf.cast(tf.clip_by_value(xy, 0, 255), tf.uint8, name='viz')
    tf.summary.image(name, xy, max_outputs=30)


class Model(ModelDesc):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def inputs(self):
        return [tf.placeholder(tf.float32, (None, 2 * self.H, 2 * self.W, 3), 'long_exposure'),
                tf.placeholder(tf.float32, (None, self.H, self.W, 4), 'short_exposure')]

    def build_graph(self, long_expo, short_expo):
        NF = 32
        with argscope([tf.layers.conv2d], activation=tf.nn.leaky_relu, padding='same'):
            with argscope([tf.layers.conv2d_transpose], padding='valid', use_bias=False,
                          kernel_initializer=tf.truncated_normal_initializer(0.02)):
                conv1 = tf.layers.conv2d(short_expo, NF, 3, name='conv1_1')
                conv1 = tf.layers.conv2d(conv1, NF, 3, name='conv1_2')
                pool1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='same', name='maxpool1')

                conv2 = tf.layers.conv2d(pool1, 2 * NF, 3, name='conv2_1')
                conv2 = tf.layers.conv2d(conv2, 2 * NF, 3, name='conv2_2')
                pool2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='same', name='maxpool2')

                conv3 = tf.layers.conv2d(pool2, 4 * NF, 3, name='conv3_1')
                conv3 = tf.layers.conv2d(conv3, 4 * NF, 3, name='conv3_2')
                pool3 = tf.layers.max_pooling2d(conv3, 2, 2, padding='same', name='maxpool3')

                conv4 = tf.layers.conv2d(pool3, 8 * NF, 3, name='conv4_1')
                conv4 = tf.layers.conv2d(conv4, 8 * NF, 3, name='conv4_2')
                pool4 = tf.layers.max_pooling2d(conv4, 2, 2, padding='same', name='maxpool4')

                conv5 = tf.layers.conv2d(pool4, 16 * NF, 3, name='conv5_1')
                conv5 = tf.layers.conv2d(conv5, 16 * NF, 3, name='conv5_2')

                deconv5 = tf.layers.conv2d_transpose(conv5, 8 * NF, 2, strides=2, name='deconv5')
                up6 = tf.concat([conv4, deconv5], name='concat6', axis=-1)
                conv6 = tf.layers.conv2d(up6, 8 * NF, 3, name='conv6_1')
                conv6 = tf.layers.conv2d(conv6, 8 * NF, 3, name='conv6_2')

                deconv6 = tf.layers.conv2d_transpose(conv6, 4 * NF, 2, strides=2, name='deconv6')
                up7 = tf.concat([conv3, deconv6], name='concat7', axis=-1)
                conv7 = tf.layers.conv2d(up7, 4 * NF, 3, name='conv7_1')
                conv7 = tf.layers.conv2d(conv7, 4 * NF, 3, name='conv7_2')

                deconv7 = tf.layers.conv2d_transpose(conv7, 2 * NF, 2, strides=2, name='deconv7')
                up8 = tf.concat([conv2, deconv7], name='concat8', axis=-1)
                conv8 = tf.layers.conv2d(up8, 2 * NF, 3, name='conv8_1')
                conv8 = tf.layers.conv2d(conv8, 2 * NF, 3, name='conv8_2')

                deconv8 = tf.layers.conv2d_transpose(conv8, NF, 2, strides=2, name='deconv8')
                up9 = tf.concat([conv1, deconv8], name='concat9', axis=-1)
                conv9 = tf.layers.conv2d(up9, NF, 3, name='conv9_1')
                conv9 = tf.layers.conv2d(conv9, NF, 3, name='conv9_2')

                conv10 = tf.layers.conv2d(conv9, 12, 1, activation=None, name='conv10')
                out = tf.depth_to_space(conv10, 2, name='prediction')

        visualize_images('info', [out, long_expo])

        cost = tf.reduce_mean(tf.abs(out - long_expo), name='l1_cost')
        summary.add_moving_summary(cost)
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)
        return tf.train.AdamOptimizer(lr)


class VisualizeTestImages(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['long_exposure', 'short_exposure'], ['viz'])

    def _before_train(self):
        self.val_ds = get_test_data()
        self.val_ds.reset_state()

    def _trigger(self):
        idx = 0
        for long_expo, short_expo in self.val_ds.get_data():
            prediction = self.pred(long_expo, short_expo)[0]
            self.trainer.monitors.put_image('test-{}'.format(idx), prediction)
            idx += 1


def get_train_data():
    ds = SonyDataset(DATA_ROOT_DIR, subset='train')
    ds = RandomCropRaw(ds)
    aus = [imgaug.Flip(horiz=True), imgaug.Flip(vert=True), imgaug.Transpose()]
    ds = AugmentImageComponents(ds, aus, index=(0, 1), copy=False)
    ds = PrefetchDataZMQ(ds, nr_proc=10)
    ds = BatchData(ds, 8)
    return ds


def get_test_data():
    ds = SonyDataset(DATA_ROOT_DIR, subset='test', num=50)
    ds = CenterCropRaw(ds)
    ds = BatchData(ds, 10)
    ds = CacheData(ds)
    return ds


def get_config():
    logger.set_logger_dir('/scratch/wieschol/seeindark/train_log3')

    ds_train = get_train_data()
    ds_test = get_test_data()

    return TrainConfig(
        model=Model(SHAPE, SHAPE),
        data=QueueInput(ds_train),
        callbacks=[
            PeriodicTrigger(ModelSaver(), every_k_epochs=5),
            ScheduledHyperParamSetter(
                'learning_rate', [(2000 // EPOCH_MULTIPLIER, 1e-5)]),
            PeriodicTrigger(VisualizeTestImages(), every_k_epochs=5),
            InferenceRunner(ds_test,
                            ScalarStats(['l1_cost'])),
        ],
        steps_per_epoch=ds_train.size() * EPOCH_MULTIPLIER,
        max_epoch=4000 // EPOCH_MULTIPLIER,
    )


def apply(model_path, filenames_short, exposure_time_sec):
    import rawpy
    import cv2
    import numpy as np
    from prepare_sony import pack_raw
    import glob
    import tqdm

    if os.path.isfile(filenames_short):
        filenames_short = [filenames_short]
    else:
        filenames_short = glob.glob(filenames_short)

    # assume all images have same shape to avoid multiple loads of the checkpoint
    short_raw = rawpy.imread(filenames_short[0])
    short_uint16 = pack_raw(short_raw)
    short_float = short_uint16.astype(np.float32)
    short_float = np.maximum(short_float - 512, 0) / (16383 - 512)
    H, W, _ = short_float.shape

    predict_func = OfflinePredictor(PredictConfig(
        model=Model(H, W),
        session_init=get_model_loader(model_path),
        input_names=['short_exposure'],
        output_names=['prediction']))

    # apply model to all images
    for filename_short in tqdm.tqdm(filenames_short):
        in_exposure_time = float(filename_short.split('_')[-1].replace('s.ARW', ''))
        factor = exposure_time_sec / in_exposure_time

        short_raw = rawpy.imread(filename_short)
        short_uint16 = pack_raw(short_raw)
        short_float = short_uint16.astype(np.float32)
        short_float = np.maximum(short_float - 512, 0) / (16383 - 512)
        short_float *= factor
        pred = predict_func(short_float[None, ...])[0][0, ...]
        p = np.clip(pred * 255, 0, 255)
        out_fn = 'outputs/{}-{}-prediction.jpg'.format(os.path.basename(filename_short)[:-4], exposure_time_sec)
        cv2.imwrite(out_fn, p[:, :, ::-1])

        naive_uint16 = short_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        naive_float = np.float32(naive_uint16 / 65535.0)
        # naive_float *= factor
        naive_float *= pred.mean() / naive_float.mean()

        p = np.clip(naive_float * 255, 0, 255)
        out_fn = 'outputs/{}-{}-noise.jpg'.format(os.path.basename(filename_short)[:-4], exposure_time_sec)
        cv2.imwrite(out_fn, p[:, :, ::-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--short', type=str, help='path to short exposure file')
    parser.add_argument('--exposure_time', type=float, help='final exposure time', default=1.)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.apply:
        apply(args.load, args.short, args.exposure_time)
    else:
        config = get_config()

        if args.load:
            config.session_init = SaverRestore(args.load)

        nr_tower = max(get_nr_gpu(), 1)
        trainer = SyncMultiGPUTrainerParameterServer(nr_tower)
        launch_train_with_config(config, trainer)
