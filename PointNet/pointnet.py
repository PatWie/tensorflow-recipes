#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
from tensorpack import *
import os
import tensorflow as tf
import numpy as np
from tensorpack.utils import logger
import tensorpack.tfutils.symbolic_functions as symbf


BATCH_SIZE = 32
NUM_POINTS = 1024
N_TRANSFORM = True


@layer_register()
def TransformPoints(x, K, in_dim=3):
    W = tf.get_variable('W', [x.get_shape()[1], in_dim * K], initializer=tf.constant_initializer(0.0))
    b = tf.get_variable('b', [in_dim * K], initializer=tf.constant_initializer(0.0))
    # prior transf is identity matrix
    b += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
    transform = tf.matmul(x, W)
    transform = tf.nn.bias_add(transform, b)
    transform = tf.reshape(transform, [-1, in_dim, K])
    return transform


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, NUM_POINTS, 3), 'point'),
                InputDesc(tf.int32, (None, ), 'label')]

    def input_transform(self, points, k=3):
        # [B,N,3] --> [3, k]
        num_point = points.get_shape()[1]
        points = tf.expand_dims(points, -1)
        with argscope(Conv2D, nl=BNReLU, padding='VALID'), \
                argscope(FullyConnected, nl=BNReLU):
            transmat = (LinearWrap(points)
                        .Conv2D('tconv0', 64, kernel_shape=[1, 3])
                        .Conv2D('tconv1', 128, kernel_shape=1)
                        .Conv2D('tconv2', 1024, kernel_shape=1)
                        .MaxPooling('tpool0', [num_point, 1])
                        .FullyConnected('tfc0', 512, nl=BNReLU)
                        .FullyConnected('tfc1', 256, nl=BNReLU)
                        .TransformPoints('transf_xyz', 3, in_dim=3)())
        logger.info('transformation matrix: {}\n\n'.format(transmat.get_shape()))
        return transmat

    def feature_transform(self, points, k=64):
        # [B,N,1,K] --> [k, k]
        num_point = points.get_shape()[1]
        with argscope(Conv2D, nl=BNReLU, kernel_shape=1, padding='VALID'), \
                argscope(FullyConnected, nl=BNReLU):
            transmat = (LinearWrap(points)
                        .Conv2D('tfconv0', 64)
                        .Conv2D('tfconv1', 128)
                        .Conv2D('tfconv2', 1024)
                        .MaxPooling('tfpool0', [num_point, 1])
                        .FullyConnected('tffc0', 512)
                        .FullyConnected('tffc1', 256)
                        .TransformPoints('transf_features', k, in_dim=k)())
        logger.info('transformation matrix: {}\n\n'.format(transmat.get_shape()))
        return transmat

    def _build_graph(self, inputs):
        points, label = inputs
        num_point = points.get_shape()[1]

        logger.info('input transform')
        trans_mat = self.input_transform(points)
        points = tf.matmul(points, trans_mat)
        points = tf.expand_dims(points, -1)

        logger.info('mlp(64, 64)')
        with argscope(Conv2D, nl=BNReLU, padding='VALID'):
            points = (LinearWrap(points)
                      .Conv2D('conv0', 64, kernel_shape=[1, 3])
                      .Conv2D('conv1', 64, kernel_shape=[1, 1])())
        if N_TRANSFORM:
            logger.info('feature transform')
            trans_mat2 = self.feature_transform(points)
            points = tf.reshape(points, [-1, NUM_POINTS, 64])
            points = tf.matmul(points, trans_mat2)
            points = tf.expand_dims(points, [2])

        logger.info('mlp(64, 128, 1024)')
        with argscope(Conv2D, nl=BNReLU, kernel_shape=[1, 1], padding='VALID'):
            points = (LinearWrap(points)
                      .Conv2D('conv2', 64)
                      .Conv2D('conv3', 128)
                      .Conv2D('conv4', 1024)())

        logger.info('global feature')
        points = MaxPooling('tpool0', points, [num_point, 1])

        logger.info('output scores')
        with argscope([Conv2D, FullyConnected], nl=BNReLU):
            logits = (LinearWrap(points)
                      .FullyConnected('fc0', 512)
                      .Dropout('drop0', 0.5)
                      .FullyConnected('fc1', 256)
                      .Dropout('drop1', 0.5)
                      .FullyConnected('fc2', 40, nl=tf.identity)())

        # vanilla classification loss
        cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cls_loss = tf.reduce_mean(cls_loss, name="cls_costs")

        accuracy = symbf.accuracy(logits, label, name='accuracy')

        if N_TRANSFORM:
            # orthogonality
            mat_diff = tf.matmul(trans_mat2, tf.transpose(trans_mat2, perm=[0, 2, 1]))
            mat_diff = tf.subtract(mat_diff, tf.constant(np.eye(64), dtype=tf.float32), name="mat_diff")
            mat_diff_loss = tf.nn.l2_loss(mat_diff)
            self.cost = tf.add(cls_loss, mat_diff_loss, name="total_costs")
            summary.add_moving_summary(mat_diff_loss)
        else:
            self.cost = tf.identity(cls_loss, name="total_costs")
        summary.add_moving_summary(cls_loss, self.cost, accuracy)

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 0.0001, summary=True)
        return tf.train.MomentumOptimizer(lr, momentum=0.9)


class RandomPoints(MapDataComponent):
    def __init__(self, ds, nr, index=0):
        self.rng = get_rng(self)

        def func(points):
            num = points.shape[0]
            idxs = list(range(num))
            self.rng.shuffle(idxs)
            idxs = idxs[:nr]
            return points[idxs]
        super(RandomPoints, self).__init__(ds, func, index=index)

    def reset_state(self):
        self.rng = get_rng(self)
        self.ds.reset_state()


class JitterPoints(MapDataComponent):
    def __init__(self, ds, index=0, sigma=0.01, clip=0.05):
        self.rng = get_rng(self)

        def func(points):
            noise = self.rng.rand(*points.shape)
            noise = np.clip(sigma * noise, -clip, clip)
            return points + noise
        super(JitterPoints, self).__init__(ds, func, index=index)

    def reset_state(self):
        self.rng = get_rng(self)
        self.ds.reset_state()


class RandomRotatePoints(MapDataComponent):
    def __init__(self, ds, index=0, sigma=0.01, clip=0.05):
        self.rng = get_rng(self)

        def func(points):
            rotation_angle = self.rng.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])

            return np.dot(points.reshape((-1, 3)), rotation_matrix)
        super(RandomRotatePoints, self).__init__(ds, func, index=index)

    def reset_state(self):
        self.rng = get_rng(self)
        self.ds.reset_state()


def get_data(hdf5s):
    root = '/datasets/ModelNet40/modelnet40_ply_hdf5_2048/'
    ds = []
    for hdf5 in hdf5s:
        ds.append(HDF5Data(os.path.join(root, hdf5), ['data', 'label']))
    ds = RandomChooseData(ds)
    ds = MapData(ds, lambda dp: [dp[0], dp[1][0]])
    ds = RandomPoints(ds, NUM_POINTS)
    ds = RandomRotatePoints(ds)
    ds = JitterPoints(ds)
    ds = PrefetchDataZMQ(ds, 2)
    ds = BatchData(ds, BATCH_SIZE)
    return ds


def get_config():
    logger.auto_set_dir()

    ds_train = get_data(['ply_data_train%i.h5' % i for i in range(5)])
    ds_train = FixedSizeData(ds_train, 2048 * 5)
    ds_test = get_data(['ply_data_test%i.h5' % i for i in range(2)])
    ds_test = FixedSizeData(ds_test, 250)

    return TrainConfig(
        model=Model(),
        dataflow=ds_train,
        callbacks=[
            ModelSaver(),
            MaxSaver('validation_accuracy'),
            InferenceRunner(ds_test, [ScalarStats('total_costs'), ScalarStats('accuracy'),
                            ScalarStats('cls_costs')]),
        ],
        extra_callbacks=[
            MovingAverageSummary(),
            ProgressBar(['tower0/total_costs', 'tower0/accuracy']),
            MergeAllSummaries(),
            RunUpdateOps()
        ],
        steps_per_epoch=ds_train.size(),
        max_epoch=100,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', required=True)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--transform', help='with transform [default: 1]', action='store_true')
    args = parser.parse_args()

    NR_GPU = len(args.gpu.split(','))
    BATCH_SIZE = BATCH_SIZE // NR_GPU
    N_TRANSFORM = args.transform
    with change_gpu(args.gpu):
        config = get_config()
        config.nr_tower = NR_GPU

        if args.load:
            config.session_init = SaverRestore(args.load)

        SyncMultiGPUTrainer(config).train()
