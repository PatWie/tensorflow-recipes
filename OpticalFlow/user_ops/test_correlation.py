#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from __init__ import correlation_cost
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.framework import constant_op


def python_correlation(A, B, kernel_size, max_displacement, stride_1, stride_2, pad, data_format):
    """ This is a fallback option for the correlation cost layer
    """
    assert kernel_size == 1
    assert data_format == 'NCHW'

    b, c, h, w = A.shape.as_list()

    border = max_displacement
    dr = int(max_displacement / stride_2)

    Hout = int(np.ceil((h + 2 * (pad - border)) / stride_1))
    Wout = int(np.ceil((w + 2 * (pad - border)) / stride_1))

    Apad = tf.pad(A, [[0, 0], [0, 0], [pad, pad], [pad, pad]])
    Bpad = tf.pad(B, [[0, 0], [0, 0], [pad, pad], [pad, pad]])

    res = []

    for tj in range(-dr, dr + 1):
        for ti in range(-dr, dr + 1):
            res_h = []
            for h in range(0, Hout):
                h1 = int(h * stride_1 + max_displacement)
                res_w = []
                for w in range(0, Wout):
                    w1 = int(w * stride_1 + max_displacement)

                    patchA = Apad[:, :, h1:h1 + 1, w1:w1 + 1]

                    w2 = w1 + ti * stride_2
                    h2 = h1 + tj * stride_2

                    patchB = Bpad[:, :, h2:h2 + 1, w2:w2 + 1]

                    ans = tf.reduce_mean(patchA * patchB, axis=[1], keepdims=True)
                    res_w.append(ans)

                res_h.append(tf.concat(res_w, axis=3))
            res.append(tf.concat(res_h, axis=2))
    res = tf.concat(res, axis=1)
    with tf.Session() as sess:
        return sess.run(res)


class CorrelationCostTest(test.TestCase):

    def _forward(self, input_a, input_b,
                 kernel_size,
                 max_displacement,
                 stride_1,
                 stride_2,
                 pad,
                 data_format,
                 use_gpu=False):
        with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu) as sess:

            input_a_op = ops.convert_to_tensor(input_a)
            input_b_op = ops.convert_to_tensor(input_b)

            kernel_size = 1
            max_displacement = 2
            stride_1 = 1
            stride_2 = 2
            pad = 4

            call_op = correlation_cost
            actual_op = call_op(input_a_op, input_b_op,
                                kernel_size=kernel_size,
                                max_displacement=max_displacement,
                                stride_1=stride_1,
                                stride_2=stride_2,
                                pad=pad,
                                data_format=data_format)

            return sess.run(actual_op)

    def _forward_both(self, shape, data_format='NCHW', dtype=dtypes.float32):
        # some shape to test uneven number of channels
        input_a = np.random.randn(*shape)
        input_b = np.random.randn(*shape)

        input_a = constant_op.constant(input_a, dtype=dtype)
        input_b = constant_op.constant(input_b, dtype=dtype)

        kernel_size = 1
        max_displacement = 2
        stride_1 = 1
        stride_2 = 2
        pad = 4

        if data_format == 'NHWC':
            input_a = array_ops.transpose(input_a, [0, 2, 3, 1])
            input_b = array_ops.transpose(input_b, [0, 2, 3, 1])

        actual_cpu = self._forward(input_a, input_b,
                                   kernel_size=kernel_size,
                                   max_displacement=max_displacement,
                                   stride_1=stride_1,
                                   stride_2=stride_2,
                                   pad=pad,
                                   data_format=data_format,
                                   use_gpu=False)

        actual_gpu = self._forward(input_a, input_b,
                                   kernel_size=kernel_size,
                                   max_displacement=max_displacement,
                                   stride_1=stride_1,
                                   stride_2=stride_2,
                                   pad=pad,
                                   data_format=data_format,
                                   use_gpu=False)

        if data_format == 'NCHW':
            actual_python = python_correlation(input_a, input_b,
                                               kernel_size=kernel_size,
                                               max_displacement=max_displacement,
                                               stride_1=stride_1,
                                               stride_2=stride_2,
                                               pad=pad,
                                               data_format=data_format)

            self.assertEqual(actual_cpu.shape, actual_python.shape)
            self.assertAllClose(actual_cpu, actual_python)

        self.assertEqual(actual_cpu.shape, actual_gpu.shape)
        self.assertAllClose(actual_cpu, actual_gpu)

    def _gradients(self, data_format='NCHW', use_gpu=False):

        batch, channels, height, width = 2, 3, 5, 6
        input_a = np.random.randn(batch, channels, height, width)
        input_b = np.random.randn(batch, channels, height, width)

        kernel_size = 1
        max_displacement = 2
        stride_1 = 1
        stride_2 = 2
        pad = 4

        if data_format == 'NHWC':
            input_a = input_a.transpose(0, 2, 3, 1)
            input_b = input_b.transpose(0, 2, 3, 1)

        with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu):

            input_a_op = ops.convert_to_tensor(input_a, dtype=dtypes.float32)
            input_b_op = ops.convert_to_tensor(input_b, dtype=dtypes.float32)

            call_op = correlation_cost
            actual_op = call_op(input_a_op, input_b_op,
                                kernel_size=kernel_size,
                                max_displacement=max_displacement,
                                stride_1=stride_1,
                                stride_2=stride_2,
                                pad=pad,
                                data_format=data_format)

            err_a = test.compute_gradient_error(
                [input_a_op, input_b_op],
                [input_a.shape, input_b.shape],
                actual_op, actual_op.shape.as_list())

            self.assertLess(err_a, 1e-4)

    def testForwardSameFloatLarge(self):
        # to test channel_num larger than 1 warp
        self._forward_both((1, 65, 3, 4), data_format='NCHW', dtype=dtypes.float32)
        self._forward_both((1, 65, 3, 4), data_format='NHWC', dtype=dtypes.float32)

    def testForwardSameDoubleLarge(self):
        # to test channel_num larger than 1 warp
        self._forward_both((1, 65, 3, 4), data_format='NCHW', dtype=dtypes.float64)
        self._forward_both((1, 65, 3, 4), data_format='NHWC', dtype=dtypes.float64)

    def testForwardSameFloatSmall(self):
        # to test channel_num smaller than 1 warp
        self._forward_both((1, 15, 3, 4), data_format='NCHW', dtype=dtypes.float32)
        self._forward_both((1, 15, 3, 4), data_format='NHWC', dtype=dtypes.float32)

    def testForwardSameDoubleSmall(self):
        # to test channel_num smaller than 1 warp
        self._forward_both((1, 15, 3, 4), data_format='NCHW', dtype=dtypes.float64)
        self._forward_both((1, 15, 3, 4), data_format='NHWC', dtype=dtypes.float64)

    def testBackwardNCHW(self):
        self._gradients(data_format='NCHW', use_gpu=False)
        self._gradients(data_format='NCHW', use_gpu=True)

    def testBackwardNHWC(self):
        self._gradients(data_format='NHWC', use_gpu=False)
        self._gradients(data_format='NHWC', use_gpu=True)


if __name__ == "__main__":
    test.main()
