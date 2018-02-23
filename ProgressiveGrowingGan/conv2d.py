#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: conv2d.py


import tensorflow as tf
import numpy as np
from tensorpack.models.registry import layer_register
from tensorpack.models.utils import VariableHolder
from tensorpack.utils.argtools import shape2d, shape4d

__all__ = ['MyConv2D']

"""
Taken from the Tensorpack implementation to include the initializer from the paper.

Remark:

    This should not make any difference, just to silence the conscience. See:

        sess = tf.InteractiveSession()

        shape = [3, 3, 128, 32]
        fan_in = 3 * 3 * 128
        c = tf.constant(np.sqrt(2. / fan_in), dtype=tf.float32)
        default_init = tf.variance_scaling_initializer(scale=2.0)
        paper_init = tf.random_normal_initializer(stddev=1.)

        default_inits = [np.abs(sess.run(tf.reduce_sum(default_init(shape)))) for _ in range(100)]
        paper_inits = [np.abs(sess.run(tf.reduce_sum(c * paper_init(shape)))) for _ in range(100)]

        print np.sum(default_inits)
        print np.sum(paper_inits)
"""

@layer_register(log_shape=True)
def MyConv2D(x, out_channel, kernel_shape,
             padding='SAME', stride=1,
             W_init=None, b_init=None,
             activation=tf.identity, split=1, use_bias=True,
             data_format='NHWC', wscale=True):

    in_shape = x.get_shape().as_list()
    channel_axis = 3 if data_format == 'NHWC' else 1
    in_channel = in_shape[channel_axis]
    assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
    assert in_channel % split == 0
    assert out_channel % split == 0

    kernel_shape = shape2d(kernel_shape)
    padding = padding.upper()
    filter_shape = kernel_shape + [in_channel / split, out_channel]
    stride = shape4d(stride, data_format=data_format)

    # see paper details -- START
    fan_in = kernel_shape[0] * kernel_shape[1] * in_channel
    c = tf.constant(np.sqrt(2. / fan_in), dtype=tf.float32)

    # "use a trivial N(0,1)"
    W_init = tf.random_normal_initializer(stddev=c)
    W = tf.get_variable('W', filter_shape, initializer=W_init)

    # "scale weights at runtime"
    scale = tf.sqrt(tf.reduce_mean(W ** 2))

    W = W / scale

    if b_init is None:
        b_init = tf.constant_initializer()

    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=b_init)

    if split == 1:
        conv = tf.nn.conv2d(x, W, stride, padding, data_format=data_format)
    else:
        inputs = tf.split(x, split, channel_axis)
        kernels = tf.split(W, split, 3)
        outputs = [tf.nn.conv2d(i, k, stride, padding, data_format=data_format)
                   for i, k in zip(inputs, kernels)]
        conv = tf.concat(outputs, channel_axis)

    conv = conv * scale

    ret = activation(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')
    ret.variables = VariableHolder(W=W)
    if use_bias:
        ret.variables.b = b
    return ret
