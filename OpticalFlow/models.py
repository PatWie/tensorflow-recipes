#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>


import tensorflow as tf
from tensorpack import *
from user_ops import correlation

enable_argscope_for_module(tf.layers)


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


SHAPE = None
CHANNELS = 3


class FlowNetBase(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, (None, CHANNELS, SHAPE, SHAPE), 'left'),
                tf.placeholder(tf.float32, (None, CHANNELS, SHAPE, SHAPE), 'right')]

    def graph_structure(self, left, right):
        raise NotImplementedError()

    def build_graph(self, left, right):
        prediction = self.graph_structure(left, right)
        prediction = resize(prediction / 20.)
        tf.identity(prediction, name="prediction")


class FlowNet2S(FlowNetBase):
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

        return tf.identity(flow2, name='flow2')


class FlowNet2C(FlowNetBase):
    def graph_structure(self, left, right):
        # this is refactored into a separate function for a potential re-implementation of the entire FlowNet2
        x = tf.stack([left, right], axis=2)
        rgb_mean = tf.reduce_mean(x, axis=[0, 2, 3, 4], keep_dims=True)
        x = (x - rgb_mean) / 255.

        x1, x2 = tf.unstack(x, axis=2)
        x1x2 = tf.concat([x1, x2], axis=0)

        with argscope([tf.layers.conv2d], activation=lambda x: tf.nn.leaky_relu(x, 0.1),
                      padding='valid', strides=2, kernel_size=3,
                      data_format='channels_first'), \
            argscope([tf.layers.conv2d_transpose], padding='same', activation=tf.identity,
                     data_format='channels_first', strides=2, kernel_size=4):
            x = tf.layers.conv2d(pad(x1x2, 3), 64, kernel_size=7, name='conv1')
            out_conv2 = tf.layers.conv2d(pad(x, 2), 128, kernel_size=5, name='conv2')
            out_conv3 = tf.layers.conv2d(pad(out_conv2, 2), 256, kernel_size=5, name='conv3')

            out_conv2a, _ = tf.split(out_conv2, 2, axis=0)
            out_conv3a, out_conv3b = tf.split(out_conv3, 2, axis=0)

            out_corr = correlation(out_conv3a, out_conv3b,
                                   kernel_size=1,
                                   max_displacement=20,
                                   stride_1=1,
                                   stride_2=2,
                                   pad=20)
            out_corr = tf.nn.leaky_relu(out_corr, 0.1)

            out_conv_redir = tf.layers.conv2d(out_conv3a, 32, kernel_size=1, strides=1, name='conv_redir')

            x = tf.concat([out_conv_redir, out_corr], axis=1)

            in_conv3_1 = tf.concat([out_conv_redir, out_corr], axis=1)
            out_conv3_1 = tf.layers.conv2d(pad(in_conv3_1, 1), 256, name='conv3_1', strides=1)

            x = tf.layers.conv2d(pad(out_conv3_1, 1), 512, name='conv4')
            conv4 = tf.layers.conv2d(pad(x, 1), 512, name='conv4_1', strides=1)
            x = tf.layers.conv2d(pad(conv4, 1), 512, name='conv5')
            conv5 = tf.layers.conv2d(pad(x, 1), 512, name='conv5_1', strides=1)
            x = tf.layers.conv2d(pad(conv5, 1), 1024, name='conv6')
            conv6 = tf.layers.conv2d(pad(x, 1), 1024, name='conv6_1', strides=1)

            flow6 = tf.layers.conv2d(pad(conv6, 1), 2, name='predict_flow6', strides=1, activation=tf.identity)
            flow6_up = tf.layers.conv2d_transpose(flow6, 2, name='upsampled_flow6_to_5', use_bias=False)
            x = tf.layers.conv2d_transpose(conv6, 512, name='deconv5', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            # return flow6
            concat5 = tf.concat([conv5, x, flow6_up], axis=1)
            flow5 = tf.layers.conv2d(pad(concat5, 1), 2, name='predict_flow5', strides=1, activation=tf.identity)
            flow5_up = tf.layers.conv2d_transpose(flow5, 2, name='upsampled_flow5_to_4', use_bias=False)
            x = tf.layers.conv2d_transpose(concat5, 256, name='deconv4', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat4 = tf.concat([conv4, x, flow5_up], axis=1)
            flow4 = tf.layers.conv2d(pad(concat4, 1), 2, name='predict_flow4', strides=1, activation=tf.identity)
            flow4_up = tf.layers.conv2d_transpose(flow4, 2, name='upsampled_flow4_to_3', use_bias=False)
            x = tf.layers.conv2d_transpose(concat4, 128, name='deconv3', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat3 = tf.concat([out_conv3_1, x, flow4_up], axis=1)
            flow3 = tf.layers.conv2d(pad(concat3, 1), 2, name='predict_flow3', strides=1, activation=tf.identity)
            flow3_up = tf.layers.conv2d_transpose(flow3, 2, name='upsampled_flow3_to_2', use_bias=False)
            x = tf.layers.conv2d_transpose(concat3, 64, name='deconv2', activation=lambda x: tf.nn.leaky_relu(x, 0.1))

            concat2 = tf.concat([out_conv2a, x, flow3_up], axis=1)
            flow2 = tf.layers.conv2d(pad(concat2, 1), 2, name='predict_flow2', strides=1, activation=tf.identity)
            flow2 = tf.identity(flow2, name='flow2')
            return flow2

