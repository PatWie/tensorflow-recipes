#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>, 2018

import numpy as np
import tensorflow as tf

from __init__ import correlation

np.random.seed(42)
tf.set_random_seed(42)


class CorrelationTest(tf.test.TestCase):

    def forward(self):
        values = np.load('forward.npy')

        input_a_op = tf.convert_to_tensor(values.item()['input_a'])
        input_b_op = tf.convert_to_tensor(values.item()['input_b'])

        kernel_size = values.item()['kernel_size']
        max_displacement = values.item()['max_displacement']
        stride_1 = values.item()['stride_1']
        stride_2 = values.item()['stride_2']
        pad = values.item()['pad_size']

        print('kernel_size', kernel_size)
        print("#+#+#+#+#+#+##+#+#+#")

        with self.test_session(use_gpu=True, force_gpu=True) as sess:
            actual_op = correlation(input_a_op, input_b_op,
                                    kernel_size=kernel_size,
                                    max_displacement=max_displacement,
                                    stride_1=stride_1,
                                    stride_2=stride_2,
                                    pad=pad)
            actual = sess.run(actual_op)

            self.assertEqual(actual.shape, values.item()['expected'].shape)
            self.assertAllClose(actual, values.item()['expected'])

    def test_forward_float(self):
        self.forward()

    def backward(self):
        values = np.load('forward.npy')

        input_a_op = tf.convert_to_tensor(values.item()['input_a'])
        input_b_op = tf.convert_to_tensor(values.item()['input_b'])

        kernel_size = values.item()['kernel_size']
        max_displacement = values.item()['max_displacement']
        stride_1 = values.item()['stride_1']
        stride_2 = values.item()['stride_2']
        pad = values.item()['pad_size']

        with self.test_session(use_gpu=True, force_gpu=True) as sess:
            actual_op = correlation(input_a_op, input_b_op,
                                    kernel_size, max_displacement, stride_1, stride_2, pad)
            actual = sess.run(actual_op)

            err_a = tf.test.compute_gradient_error(
                input_a_op, values.item()['input_a'].shape,
                actual_op, actual.shape)

            self.assertLess(err_a, 1e-2)

            err_b = tf.test.compute_gradient_error(
                input_b_op, values.item()['input_b'].shape,
                actual_op, actual.shape)

            self.assertLess(err_b, 1e-2)

    # def test_backward_float(self):
    #     self.backward()


if __name__ == '__main__':
    tf.test.main()
