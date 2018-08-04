#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

import os
import cv2
from helper import Flow
import argparse
import tensorflow as tf
import numpy as np

from tensorpack import *


enable_argscope_for_module(tf.layers)

"""
This is a tensorpack script re-implementation of
FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks
https://arxiv.org/abs/1612.01925

This is not an attempt to reproduce the lengthly training protocol,
but to rely on tensorpack's "OfflinePredictor" for easier inference.

The ported pre-trained Caffe-model is here
http://files.patwie.com/recipes/models/flownet2-s.npz

It has the original license:

```
    Pre-trained weights are provided for research purposes only and without any warranty.

    Any commercial use of the pre-trained weights requires FlowNet2 authors consent.
    When using the the pre-trained weights in your research work, please cite the following paper:

    @InProceedings{IMKDB17,
      author       = "E. Ilg and N. Mayer and T. Saikia and M. Keuper and A. Dosovitskiy and T. Brox",
      title        = "FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
      month        = "Jul",
      year         = "2017",
      url          = "http://lmb.informatik.uni-freiburg.de//Publications/2017/IMKDB17"
    }
```

To verify this against the Caffe-model, please use the test-case:

    http://files.patwie.com/recipes/models/flownet2-s-expected-values.npy (containing item 07446 of flying-chairs)

and run

    python flownet2s.py --gpu 0 --verify --load flownet2-s.npz --expect flownet2-s-expected-values.npy

To run it on actual data:

    python flownet2s.py --gpu 0 \
        --left left_img.ppm \
        --right right_img.ppm \
        --apply --load flownet2-s.npz

"""

BATCH_SIZE = 16
SHAPE = None
CHANNELS = 3


def pad(x, p=3):
    """Pad tensor in H, W

    Remarks:
        TensorFlow uses "ceil(input_spatial_shape[i] / strides[i])" rather than explicit padding
        like Caffe, pyTorch does. Hence, we need to pad here beforehand.

    Args:
        x (tf.tensor): incoming tensor
        p (int, optional): padding for H, W

    Returns:
        tf.tensor: padded tensor
    """
    return tf.pad(x, [[0, 0], [0, 0], [p, p], [p, p]])


def resize(x, factor=4):
    """Resize input tensor with unkown input-shape by a factor

    Args:
        x (tf.Tensor): tensor NCHW
        factor (int, optional): resize factor for H, W

    Returns:
        tf.Tensor: resized tensor NCHW
    """
    shp = tf.shape(x)[2:] * factor
    shp = tf.Print(shp, [shp])
    # NCHW -> NHWC
    x = tf.transpose(x, [0, 2, 3, 1])
    x = tf.image.resize_bilinear(x, shp, align_corners=True)
    # NHWC -> NCHW
    return tf.transpose(x, [0, 3, 1, 2])


class FlowNet2S(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, (None, CHANNELS, SHAPE, SHAPE), 'left'),
                tf.placeholder(tf.float32, (None, CHANNELS, SHAPE, SHAPE), 'right')]

    def graph_structure(self, left, right):
        # this is refactored into a separate function for a potential re-implementation of the entire FlowNet2
        x = tf.stack([left, right], axis=2)
        rgb_mean = tf.reduce_mean(x, axis=[0, 2, 3, 4], keep_dims=True)
        x = (x - rgb_mean) / 255.

        x = tf.concat(tf.unstack(x, axis=2), axis=1)

        with argscope([tf.layers.conv2d], activation=lambda x: tf.nn.leaky_relu(x, 0.1),
                      padding='valid', strides=2, kernel_size=3,
                      data_format='channels_first'), \
            argscope([tf.layers.conv2d_transpose], padding='same', activation=tf.identity,
                     data_format='channels_first', strides=2, kernel_size=4):
            x = tf.layers.conv2d(pad(x, 3), 64, kernel_size=7, name='conv1')
            conv2 = tf.layers.conv2d(pad(x, 2), 128, kernel_size=5, name='conv2')
            x = tf.layers.conv2d(pad(conv2, 2), 256, kernel_size=5, name='conv3')
            conv3 = tf.layers.conv2d(pad(x, 1), 256, name='conv3_1', strides=1)
            x = tf.layers.conv2d(pad(conv3, 1), 512, name='conv4')
            conv4 = tf.layers.conv2d(pad(x, 1), 512, name='conv4_1', strides=1)
            x = tf.layers.conv2d(pad(conv4, 1), 512, name='conv5')
            conv5 = tf.layers.conv2d(pad(x, 1), 512, name='conv5_1', strides=1)
            x = tf.layers.conv2d(pad(conv5, 1), 1024, name='conv6')
            conv6 = tf.layers.conv2d(pad(x, 1), 1024, name='conv6_1', strides=1)

            flow6 = tf.layers.conv2d(pad(conv6, 1), 2, name='predict_flow6', strides=1, activation=tf.identity)
            flow6_up = tf.layers.conv2d_transpose(flow6, 2, name='upsampled_flow6_to_5', use_bias=False)
            x = tf.layers.conv2d_transpose(conv6, 512, name='deconv5', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat5 = tf.concat([conv5, x, flow6_up], axis=1)
            flow5 = tf.layers.conv2d(pad(concat5, 1), 2, name='predict_flow5', strides=1, activation=tf.identity)
            flow5_up = tf.layers.conv2d_transpose(flow5, 2, name='upsampled_flow5_to_4', use_bias=False)
            x = tf.layers.conv2d_transpose(concat5, 256, name='deconv4', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat4 = tf.concat([conv4, x, flow5_up], axis=1)
            flow4 = tf.layers.conv2d(pad(concat4, 1), 2, name='predict_flow4', strides=1, activation=tf.identity)
            flow4_up = tf.layers.conv2d_transpose(flow4, 2, name='upsampled_flow4_to_3', use_bias=False)
            x = tf.layers.conv2d_transpose(concat4, 128, name='deconv3', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat3 = tf.concat([conv3, x, flow4_up], axis=1)
            flow3 = tf.layers.conv2d(pad(concat3, 1), 2, name='predict_flow3', strides=1, activation=tf.identity)
            flow3_up = tf.layers.conv2d_transpose(flow3, 2, name='upsampled_flow3_to_2', use_bias=False)
            x = tf.layers.conv2d_transpose(concat3, 64, name='deconv2', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat2 = tf.concat([conv2, x, flow3_up], axis=1)
            flow2 = tf.layers.conv2d(pad(concat2, 1), 2, name='predict_flow2', strides=1, activation=tf.identity)

        flow2 = tf.identity(flow2, name='flow2')
        return resize(flow2 / 20.)

    def build_graph(self, left, right):
        prediction = self.graph_structure(left, right)
        tf.identity(prediction, name="prediction")

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=5e-3, trainable=False)
        summary.add_moving_summary(lr)
        return tf.train.AdamOptimizer(lr)


def apply(model_path, left, right, ground_truth=None):

    left = cv2.imread(left).astype(np.float32).transpose(2, 0, 1)[None, ...]
    right = cv2.imread(right).astype(np.float32).transpose(2, 0, 1)[None, ...]

    predict_func = OfflinePredictor(PredictConfig(
        model=FlowNet2S(),
        session_init=get_model_loader(model_path),
        input_names=['left', 'right'],
        output_names=['prediction']))

    output = predict_func(left, right)[0].transpose(0, 2, 3, 1)
    print output.shape
    flow = Flow()

    img = flow.visualize(output[0])
    if ground_truth is not None:
        img = np.concatenate([img, flow.visualize(Flow.read(ground_truth))], axis=1)

    cv2.imshow('flow output', img)
    cv2.waitKey(0)


def verify(model_path, expect_path):

    predict_func = OfflinePredictor(PredictConfig(
        model=FlowNet2S(),
        session_init=get_model_loader(model_path),
        input_names=['left', 'right'],
        output_names=['flow2']))

    expected = np.load(expect_path)
    left = expected.item().get('inputs')[:, :, 0, :, :]
    right = expected.item().get('inputs')[:, :, 1, :, :]

    output = predict_func(left, right)[0]
    assert np.allclose(output, expected.item().get('flow2'), atol=1e-5), "Network output does not match expected output"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--apply', action='store_true', help='apply network to inputs')
    parser.add_argument('--verify', action='store_true', help='verify network outputs')
    parser.add_argument('--left', help='input', type=str)
    parser.add_argument('--right', help='input', type=str)
    parser.add_argument('--gt', help='ground_truth', type=str, default=None)
    parser.add_argument('--expect', help='expected_values', type=str, default=None)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.apply:
        apply(args.load, args.left, args.right, args.gt)
    elif args.verify:
        verify(args.load, args.expect)
