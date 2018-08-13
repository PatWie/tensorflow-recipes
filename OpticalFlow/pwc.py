#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

import os
import numpy as np
from helper import Flow
import cv2
import argparse
import tensorflow as tf

from tensorpack import *
from user_ops import correlation
from flownet_models import pad

enable_argscope_for_module(tf.layers)

""""
This is a tensorpack script re-implementation of
PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume
https://arxiv.org/abs/1709.02371

This is not an attempt to reproduce the lengthly training protocol,
but to rely on tensorpack's "OfflinePredictor" for easier inference.

The ported pre-trained Caffe-model are here
http://files.patwie.com/recipes/models/pwc.npz


To run it on actual data:

    python flownet2.py --gpu 0 \
        --left 00001_img1.ppm \
        --right 00001_img2.ppm \
        --load pwc.npz

"""

CHANNELS = 3


def resample(img, flow):
    """
    flow an image/tensor (im2) back to im1, according to the optical flow

    img: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, c, h, w = [tf.shape(img)[i] for i in range(4)]

    img_flat = tf.reshape(tf.transpose(img, [0, 2, 3, 1]), [-1, c])

    dx = flow[:, 0:1, :, :]
    dy = flow[:, 1:, :, :]

    xf = tf.reshape(tf.tile(tf.range(w), [h]), [h, w])
    yf = tf.transpose(tf.reshape(tf.tile(tf.range(h), [w]), [w, h]), [1, 0])

    xf = tf.cast(xf, dx.dtype)
    yf = tf.cast(yf, dy.dtype)

    xf = xf + dx
    yf = yf + dy

    alpha = tf.transpose(xf - tf.floor(xf), [0, 2, 3, 1])
    beta = tf.transpose(yf - tf.floor(yf), [0, 2, 3, 1])

    # https://www.tensorflow.org/api_docs/python/tf/gather
    # On GPU, if an out of bound index is found, a 0 is stored in the corresponding output value.
    # no clipping here
    xL = tf.cast(tf.floor(xf), dtype=tf.int32)
    xR = tf.cast(tf.floor(xf) + 1, dtype=tf.int32)
    yT = tf.cast(tf.floor(yf), dtype=tf.int32)
    yB = tf.cast(tf.floor(yf) + 1, dtype=tf.int32)

    batch_ids = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(B), axis=-1), axis=-1), [1, h, w])

    def bilinear_interpolate(flat):

        def get(y, x):
            idx = tf.reshape(batch_ids * h * w + y * w + x, [-1])
            idx = tf.cast(idx, tf.int32)
            return tf.gather(flat, idx)

        val = tf.zeros_like(alpha)
        val += (1 - alpha) * (1 - beta) * tf.reshape(get(yT, xL), [-1, h, w, c])
        val += (0 + alpha) * (1 - beta) * tf.reshape(get(yT, xR), [-1, h, w, c])
        val += (1 - alpha) * (0 + beta) * tf.reshape(get(yB, xL), [-1, h, w, c])
        val += (0 + alpha) * (0 + beta) * tf.reshape(get(yB, xR), [-1, h, w, c])
        return val

    val = bilinear_interpolate(img_flat)

    # we need to enforce the channel_dim known during compile-time here
    shp = img.shape.as_list()
    ans = tf.reshape(val, [-1, h, w, shp[1]])

    # same for mask (NCHW --> NHWC)
    mask = tf.transpose(img, [0, 2, 3, 1]) * 0 + 1

    mask_flat = tf.reshape(mask, [-1, c])
    mask = bilinear_interpolate(mask_flat)

    # vanish mask entries which are warped out-of the image area (we clipe the coordinates)
    ans = tf.transpose(ans, [0, 3, 1, 2])
    mask = tf.transpose(mask, [0, 3, 1, 2])
    z = tf.zeros_like(mask)
    o = tf.ones_like(mask)

    # filter mask
    mask = tf.where(tf.less(mask, 0.9999 * o), z, mask)
    mask = tf.where(tf.greater(mask, z), o, mask)

    # return mask
    return ans * mask


class PWCModel(ModelDesc):

    def inputs(self):
        return [tf.placeholder(tf.float32, (None, CHANNELS, None, None), 'left'),
                tf.placeholder(tf.float32, (None, CHANNELS, None, None), 'right')]

    def build_graph(self, im1, im2):

        def corr_func(x, y):
            return correlation(x, y, kernel_size=1, max_displacement=4,
                               stride_1=1, stride_2=1, pad=4, data_format='NCHW')

        with argscope([tf.layers.conv2d], activation=lambda x: tf.nn.leaky_relu(x, 0.1),
                      padding='valid', strides=1, kernel_size=3,
                      data_format='channels_first'), \
            argscope([tf.layers.conv2d_transpose], padding='same', activation=tf.identity,
                     data_format='channels_first', strides=2, kernel_size=4):

            x = tf.concat([im1, im2], axis=0)
            x = x / 255.

            # create feature pyramids
            pyramid = []
            for k, nf in enumerate([16, 32, 64, 96, 128, 196]):
                x = tf.layers.conv2d(pad(x, 1), nf, name='conv%ia' % (k + 1), strides=2)
                x = tf.layers.conv2d(pad(x, 1), nf, name='conv%iaa' % (k + 1), strides=1)
                x = tf.layers.conv2d(pad(x, 1), nf, name='conv%ib' % (k + 1), strides=1)
                pyramid.append(tf.split(x, 2, axis=0))

            x = tf.nn.leaky_relu(corr_func(pyramid[5][0], pyramid[5][1]), 0.1)

            # warping between left and right features
            base_warp_mulp = 0.625
            for kk, stage in enumerate([6, 5, 4, 3, 2]):
                for k, nf in enumerate([128, 128, 96, 64, 32]):
                    y = tf.layers.conv2d(pad(x, 1), nf, name='conv%i_%i' % (stage, k), strides=1)
                    x = tf.concat([y, x], axis=1)

                flow = tf.layers.conv2d(pad(x, 1), 2, name='predict_flow%i' % (stage), strides=1,
                                        activation=tf.identity)
                if stage == 2:
                    break
                flow_up = tf.layers.conv2d_transpose(flow, 2, name='up_flow%i' % (stage))
                feat_up = tf.layers.conv2d_transpose(x, 2, name='up_feat%i' % (stage))
                fac = base_warp_mulp * (2**kk)
                warp = resample(pyramid[4 - kk][1], flow_up * fac)
                corr = tf.nn.leaky_relu(corr_func(pyramid[4 - kk][0], warp), 0.1)
                x = tf.concat([corr, pyramid[4 - kk][0], flow_up, feat_up], axis=1)

            nfs = [128, 128, 128, 96, 64, 32, 2]
            pads = [1, 2, 4, 8, 16, 1, 1]

            # "decoder"
            for k, (n, p) in enumerate(zip(nfs, pads)):
                x = tf.layers.conv2d(pad(x, p), n, name='dc_conv%i' % (k + 1), strides=1, dilation_rate=(p, p))

            with tf.name_scope('resize_back'):
                flow2 = (flow + x) * 20.0

            tf.identity(flow2, name='prediction')


def apply(model_path, left, right):
    left = cv2.imread(left).astype(np.float32)
    right = cv2.imread(right).astype(np.float32)

    assert left.shape == right.shape
    h_in, w_in = left.shape[:2]

    # images needs to be divisible by 64
    h = int(np.ceil(h_in / 64.) * 64.)
    w = int(np.ceil(w_in / 64.) * 64.)
    print('resize inputs (%i, %i) to (%i, %i)' % (h_in, w_in, h, w))
    left = cv2.resize(left, (w, h)).transpose(2, 0, 1)[None, ...]
    right = cv2.resize(right, (w, h)).transpose(2, 0, 1)[None, ...]

    predict_func = OfflinePredictor(PredictConfig(
        model=PWCModel(),
        session_init=get_model_loader(model_path),
        input_names=['left', 'right'],
        output_names=['prediction']))

    output = predict_func(left, right)[0].transpose(0, 2, 3, 1)[0]

    dx = cv2.resize(output[:, :, 0], (w_in, h_in)) * w_in / float(w)
    dy = cv2.resize(output[:, :, 1], (w_in, h_in)) * h_in / float(h)
    output = np.dstack((dx, dy))

    flow = Flow()
    img = flow.visualize(output)

    cv2.imwrite('pwc_output.png', img * 255)
    cv2.imshow('flow output', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--left', help='input', type=str)
    parser.add_argument('--right', help='input', type=str)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    apply(args.load, args.left, args.right)
