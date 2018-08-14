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

    def sample(img, coords):
        """
        Args:
            img: bxhxwxc
            coords: bxh2xw2x2. each coordinate is (y, x) integer.
                Out of boundary coordinates will be clipped.
        Return:
            bxh2xw2xc image
        """
        shape = img.get_shape().as_list()[1:]   # h, w, c
        batch = tf.shape(img)[0]
        shape2 = coords.get_shape().as_list()[1:3]  # h2, w2
        assert None not in shape2, coords.get_shape()
        max_coor = tf.constant([shape[0] - 1, shape[1] - 1], dtype=tf.float32)

        coords = tf.clip_by_value(coords, 0., max_coor)  # borderMode==repeat
        coords = tf.to_int32(coords)

        batch_index = tf.range(batch, dtype=tf.int32)
        batch_index = tf.reshape(batch_index, [-1, 1, 1, 1])
        batch_index = tf.tile(batch_index, [1, shape2[0], shape2[1], 1])    # bxh2xw2x1
        indices = tf.concat([batch_index, coords], axis=3)  # bxh2xw2x3
        sampled = tf.gather_nd(img, indices)
        return sampled

    def ImageSample(inputs, borderMode='repeat'):
        """
        Sample the images using the given coordinates, by bilinear interpolation.
        This was described in the paper:
        `Spatial Transformer Networks <http://arxiv.org/abs/1506.02025>`_.

        Args:
            inputs (list): [images, coords]. images has shape NHWC.
                coords has shape (N, H', W', 2), where each pair of the last dimension is a (y, x) real-value
                coordinate.
            borderMode: either "repeat" or "constant" (zero-filled)

        Returns:
            tf.Tensor: a tensor named ``output`` of shape (N, H', W', C).
        """
        image, mapping = inputs
        assert image.get_shape().ndims == 4 and mapping.get_shape().ndims == 4
        input_shape = image.get_shape().as_list()[1:]
        assert None not in input_shape, \
            "Images in ImageSample layer must have fully-defined shape"
        assert borderMode in ['repeat', 'constant']

        orig_mapping = mapping
        mapping = tf.maximum(mapping, 0.0)
        lcoor = tf.floor(mapping)
        ucoor = lcoor + 1

        diff = mapping - lcoor
        neg_diff = 1.0 - diff  # bxh2xw2x2

        lcoory, lcoorx = tf.split(lcoor, 2, 3)
        ucoory, ucoorx = tf.split(ucoor, 2, 3)

        lyux = tf.concat([lcoory, ucoorx], 3)
        uylx = tf.concat([ucoory, lcoorx], 3)

        diffy, diffx = tf.split(diff, 2, 3)
        neg_diffy, neg_diffx = tf.split(neg_diff, 2, 3)

        ret = tf.add_n([sample(image, lcoor) * neg_diffx * neg_diffy,
                        sample(image, ucoor) * diffx * diffy,
                        sample(image, lyux) * neg_diffy * diffx,
                        sample(image, uylx) * diffy * neg_diffx], name='sampled')
        if borderMode == 'constant':
            max_coor = tf.constant([input_shape[0] - 1, input_shape[1] - 1], dtype=tf.float32)
            mask = tf.greater_equal(orig_mapping, 0.5)
            mask2 = tf.less_equal(orig_mapping, max_coor+0.5)
            mask = tf.logical_and(mask, mask2)  # bxh2xw2x2
            mask = tf.reduce_all(mask, [3])  # bxh2xw2 boolean
            mask = tf.expand_dims(mask, 3)
            ret = ret * tf.cast(mask, tf.float32)
        return tf.identity(ret, name='output')

    B, c, h, w = [tf.shape(img)[i] for i in range(4)]

    img_flat = tf.reshape(tf.transpose(img, [0, 2, 3, 1]), [-1, c])

    xf = tf.reshape(tf.tile(tf.range(w), [h]), [h, w])
    yf = tf.transpose(tf.reshape(tf.tile(tf.range(h), [w]), [w, h]), [1, 0])
    grid = tf.stack([yf, xf], axis=0)
    grid = tf.expand_dims(grid, axis=0)
    grid = tf.cast(grid, flow.dtype)

    vgrid = grid + tf.reverse(flow, axis=[1])
    shp2 = tf.to_float(tf.stack([h, w]))
    vgrid = vgrid * 2.0 / tf.reshape(shp2 - 1., [1, 2, 1, 1]) - 1.

    def interpolate(x, start, end):
        return (x + 1) * (end - start) / 2. + start

    # start = tf.constant([0.5, 0.5], dtype=tf.float32)
    # end = tf.constant([int(h) - 0.5, int(w) - 0.5], dtype=tf.float32)
    start = tf.constant([0.5, 0.5], dtype=tf.float32)
    end = tf.stack([tf.cast(h, tf.float32) - 0.5, tf.cast(w, tf.float32) - 0.5], axis=0)

    coords = interpolate(vgrid, tf.reshape(start, [1, 2, 1, 1]), tf.reshape(end, [1, 2, 1, 1]))
    coords = coords - 0.5

    img_trans = tf.transpose(img, [0, 2, 3, 1])
    coords = tf.transpose(coords, [0, 2, 3, 1])
    output = ImageSample([img_trans, coords], borderMode='constant')
    ans = tf.transpose(output, [0, 3, 1, 2])
    return ans


class PWCModel(ModelDesc):

    def __init__(self, batch=None, height=None, width=None):
        self.batch = batch
        self.height = height
        self.width = width

    def inputs(self):
        return [tf.placeholder(tf.float32, (None, CHANNELS, self.height, self.width), 'left'),
                tf.placeholder(tf.float32, (None, CHANNELS, self.height, self.width), 'right')]

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
        model=PWCModel(batch=1, height=h, width=w),
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
