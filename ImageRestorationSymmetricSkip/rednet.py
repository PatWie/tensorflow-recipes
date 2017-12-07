#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

"""
Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections
<https://arxiv.org/abs/1606.08921>

- experiments with STRIDE 3 gives PSNR 18.3
- experiments with STRIDE 1 gives PSNR 23.4
"""

import argparse
import numpy as np
from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.tfutils.summary import add_moving_summary
import tensorflow as tf
import tensorpack.tfutils.symbolic_functions as symbf


BATCH = 32
SHAPE = 243
CHANNELS = 1
STRIDE = 1
NF = 64


class Model(ModelDesc):

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, SHAPE, SHAPE, CHANNELS), 'noisy'),
                InputDesc(tf.float32, (None, SHAPE, SHAPE, CHANNELS), 'gt')]

    def _build_graph(self, inputs):
        noisy, gt = inputs
        noisy = noisy / 128.0 - 1
        gt = gt / 128.0 - 1

        def ReluConv2D(name, x, out_channels, use_relu=True):
            if use_relu:
                x = tf.nn.relu(x, name='%s_relu' % name)
            x = Conv2D('%s_conv' % name, x, out_channels)
            return x

        def ReluDeconv2D(name, x, out_channels):
            x = tf.nn.relu(x, name='%s_relu' % name)
            x = Deconv2D('%s_deconv' % name, x, out_channels)
            return x

        with argscope([Conv2D, Deconv2D], nl=tf.identity, stride=STRIDE, kernel_shape=3):
            # encoder
            e1 = ReluConv2D('enc1', noisy, NF, use_relu=False)
            e2 = ReluConv2D('enc2', e1, NF)
            e3 = ReluConv2D('enc3', e2, NF)
            e4 = ReluConv2D('enc4', e3, NF)
            e5 = ReluConv2D('enc5', e4, NF)
            # decoder
            e6 = ReluDeconv2D('dec1', e5, NF)
            e6 = tf.add(e6, e4, name='skip1')
            e7 = ReluDeconv2D('dec2', e6, NF)
            e8 = ReluDeconv2D('dec3', e7, NF)
            e8 = tf.add(e8, e2, name='skip2')
            e9 = ReluDeconv2D('dec4', e8, NF)
            e10 = ReluDeconv2D('dec5', e9, CHANNELS)

        estimate = tf.add(e10, noisy, name="estimate")
        self.cost = tf.reduce_mean(tf.squared_difference(estimate, gt), name="mse")

        estimate_scaled = 128.0 * (1.0 + estimate)
        gt_scaled = 128.0 * (1.0 + gt)

        psnr = symbf.psnr(estimate_scaled, gt_scaled, 255, name="psnr")
        print psnr.name

        # use tensorboard for visualization
        with tf.name_scope("visualization"):
            viz = (tf.concat([noisy, estimate, gt], 2) + 1.0) * 128.0
            viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
        tf.summary.image('noisy, estimate, gt', viz, max_outputs=max(30, BATCH))

        add_moving_summary(self.cost, psnr)

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 1e-3, summary=True)
        return tf.train.AdamOptimizer(lr)


class RandomGaussianNoise(imgaug.ImageAugmentor):
    # see: https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/dataflow/imgaug/noise.py
    # paper uses sigma randomly within range [0, 70]
    def __init__(self, sigma_range):
        super(RandomGaussianNoise, self).__init__()
        self._init(locals())
        self.sigma_min = sigma_range[0]
        self.sigma_size = sigma_range[1] - sigma_range[0]

    def _get_augment_params(self, img):
        return self.rng.randn(*img.shape)

    def _augment(self, img, noise):
        old_dtype = img.dtype
        sigma = self.sigma_min + self.rng.rand() * self.sigma_size
        return np.clip(img + noise * sigma, 0, 255).astype(old_dtype)


def get_data(subset='train'):

    assert subset in ['train', 'test']

    isTrain = (subset == 'train')
    ds = dataset.BSDS500(subset, shuffle=True)
    ds = MapData(ds, lambda dp: [np.dot(dp[0], [0.299, 0.587, 0.114])])
    ds = MapData(ds, lambda dp: [np.expand_dims(dp[0], axis=-1)])
    ds = MapData(ds, lambda dp: [dp[0], dp[0]])
    if isTrain:
        augmentors = [
            imgaug.RandomCrop((SHAPE, SHAPE)),
            imgaug.Brightness(63, clip=False),
            imgaug.Contrast((0.4, 1.5)),
            imgaug.Flip(horiz=True),
            imgaug.Flip(vert=True)
        ]
        ds = AugmentImageComponents(ds, augmentors, index=(0, 1))

        ds = AugmentImageComponent(ds, [RandomGaussianNoise(sigma_range=(10, 70))], index=0)
        ds = BatchDataByShape(ds, 8, idx=0)
        ds = PrefetchDataZMQ(ds, 1)
    else:
        augmentors = [
            imgaug.CenterCrop((SHAPE, SHAPE)),
        ]
        ds = AugmentImageComponents(ds, augmentors, index=(0, 1))
        ds = AugmentImageComponent(ds, [RandomGaussianNoise(sigma_range=(10, 70))], index=0)
        ds = BatchData(ds, 1)
    return ds


def get_config(batch):
    logger.auto_set_dir()
    dataset_train = get_data('train')
    dataset_val = get_data('test')

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            MinSaver('validation_mse'),
            InferenceRunner(dataset_val, [ScalarStats('mse'), ScalarStats('psnr')]),
        ],
        extra_callbacks=[
            MovingAverageSummary(),
            ProgressBar(['tower0/mse:0', 'tower0/psnr:0']),
            MergeAllSummaries(),
        ],
        model=Model(),
        steps_per_epoch=500,
        max_epoch=110,
    )


if __name__ == '__main__':
    # python train.py --gpu 2,3
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load model')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--batch', help='batch size', type=int, default=32)
    args = parser.parse_args()

    with change_gpu(args.gpu):
        BATCH = args.batch
        NR_GPU = len(args.gpu.split(','))
        config = get_config(args.batch)
        if args.load:
            config.session_init = SaverRestore(args.load)
        config.nr_tower = NR_GPU
        SyncMultiGPUTrainer(config).train()
