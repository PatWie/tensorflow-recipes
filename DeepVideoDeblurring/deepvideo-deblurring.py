#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

import argparse
import tensorflow as tf
from tensorpack import *  # noqa

"""
Re-implementation of Deep-Video Deblurring, Su et.
<https://arxiv.org/abs/1611.08387>
"""


class Model(ModelDesc):

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, SEQ_LEN, None, None, 3), 'blurry'),
                InputDesc(tf.float32, (None, SEQ_LEN, None, None, 3), 'sharp')]

    def _build_graph(self, input_vars):

        blurry, sharp = input_vars
        blurry = blurry / 128.0 - 1
        sharp = sharp / 128.0 - 1

        # the original paper predicts a sharp version of the middle-frame
        # target_id = (self.num_frames // 2 + 1)
        target_id = self.num_frames - 1
        gt = sharp[:, target_id, :, :, :]
        inputs = tf.transpose(blurry, [0, 2, 3, 1, 4])
        inputs = tf.reshape(inputs, [-1, self.img_h, self.img_w, 3 * self.num_frames])

        with argscope([Conv2D, Deconv2D], nl=lambda x, name: BatchNorm(name, x)):
            d01 = blurry[:, target_id, :, :, :]                                   # -->
            d02 = Conv2D('F0', inputs, 64, stride=1, kernel_shape=5)              # -->

            # H x W -> H/2 x W/2
            d11 = Conv2D('c1_0', tf.nn.relu(d02), 64, stride=2, kernel_shape=3)
            d12 = Conv2D('c1_1', tf.nn.relu(d11), 128, stride=1, kernel_shape=3)
            d13 = Conv2D('c1_2', tf.nn.relu(d12), 128, stride=1, kernel_shape=3)  # -->

            # H/2 x W/2 -> H/4 x W/4
            d21 = Conv2D('c2_0', tf.nn.relu(d13), 256, stride=2, kernel_shape=3)
            d22 = Conv2D('c2_1', tf.nn.relu(d21), 256, stride=1, kernel_shape=3)
            d23 = Conv2D('c2_2', tf.nn.relu(d22), 256, stride=1, kernel_shape=3)
            d24 = Conv2D('c2_3', tf.nn.relu(d23), 256, stride=1, kernel_shape=3)  # -->

            # H/4 x W/4 -> H/8 x W/8
            d31 = Conv2D('c3_0', tf.nn.relu(d24), 512, stride=2, kernel_shape=3)
            d32 = Conv2D('c3_1', tf.nn.relu(d31), 512, stride=1, kernel_shape=3)
            d33 = Conv2D('c3_2', tf.nn.relu(d32), 512, stride=1, kernel_shape=3)
            d34 = Conv2D('c3_3', tf.nn.relu(d33), 512, stride=1, kernel_shape=3)

            # H/8 x W/8 -> H/4 x W/4
            u11 = Deconv2D('U1', tf.nn.relu(d34), 256, stride=2, kernel_shape=4) + d24
            u12 = Conv2D('c4_1', tf.nn.relu(u11), 256, stride=1, kernel_shape=3)
            u13 = Conv2D('c4_2', tf.nn.relu(u12), 256, stride=1, kernel_shape=3)
            u14 = Conv2D('c4_3', tf.nn.relu(u13), 256, stride=1, kernel_shape=3)

            # H/4 x W/4 -> H/2 x W/2
            u21 = Deconv2D('U2', tf.nn.relu(u14), 128, stride=2, kernel_shape=4) + d13
            u22 = Conv2D('c5_1', tf.nn.relu(u21), 128, stride=1, kernel_shape=3)
            u23 = Conv2D('c5_2', tf.nn.relu(u22), 64, stride=1, kernel_shape=3)

            # H/2 x W/2 -> H x W
            u31 = Deconv2D('U3', tf.nn.relu(u23), 64, stride=2, kernel_shape=4) + d02
            u32 = Conv2D('c6_1', tf.nn.relu(u31), 15, stride=1, kernel_shape=3)
            u33 = Conv2D('c6_2', tf.nn.relu(u32), 3, stride=1, kernel_shape=3) + d01

        pred = u33
        tf.identity((pred + 1.0) * 128., name='predicted_img')

        self.cost = tf.nn.l2_loss(pred - gt, name="cost")


def get_config(batch_size):
    logger.auto_set_dir()
    dataset_train = YoutubeData(imgsize=128, batch=BATCH_SIZE)
    steps_per_epoch = 1000

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
        ],
        extra_callbacks=[
            MovingAverageSummary(),
            ProgressBar(['tower0/psnr_%i' % (SEQ_LEN - 1), 'tower0/psnr_input_%i' % (SEQ_LEN - 1),
                         'tower0/psnr_improv_%i' % (SEQ_LEN - 1)]),
            MergeAllSummaries(),
            RunUpdateOps()
        ],
        model=Model(),
        steps_per_epoch=steps_per_epoch,
        max_epoch=400,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', required=True)
    parser.add_argument('--batch', help='batch-size', type=int, default=32)
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    NR_GPU = len(args.gpu.split(','))
    with change_gpu(args.gpu):
        config = get_config(args.batch)
        config.nr_tower = NR_GPU
        SyncMultiGPUTrainer(config).train()
