#! /usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
import os


__all__ = ['correlation']

path = os.path.join(os.path.dirname(__file__), 'correlation_op.so')
_correlation_module = tf.load_op_library(path)
correlation = _correlation_module.correlation
correlation_grad = _correlation_module.correlation_grad


@tf.RegisterGradient("Correlation")
def _correlation_grad(corr_op, gradients):
    kernel_size = corr_op.get_attr("kernel_size")
    max_displacement = corr_op.get_attr("max_displacement")
    stride_1 = corr_op.get_attr("stride_1")
    stride_2 = corr_op.get_attr("stride_2")
    pad = corr_op.get_attr("pad")

    corr_grads = correlation_grad(gradients,
                                  corr_op.inputs[0],
                                  corr_op.inputs[1],
                                  kernel_size,
                                  max_displacement,
                                  stride_1,
                                  stride_2,
                                  pad)

    # Return the gradients with respect to input_a and input_b
    return corr_grads[0], corr_grads[1]
