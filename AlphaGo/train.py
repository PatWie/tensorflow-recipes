#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

import argparse
import os
from tensorpack import *
from go_db import GameDecoder, DihedralGroup
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
import tensorflow as tf
import multiprocessing
from tensorpack.utils.stats import RatioCounter
import sys

"""
Re-Implementation of the Policy-Network (SL) from AlphaGo:

"Mastering the Game of Go with Deep Neural Networks and Tree Search"
<https://gogameguru.com/i/2016/03/deepmind-mastering-go.pdf>
"""

BATCH_SIZE = 16
SHAPE = 19
NUM_PLANES = 47


class Model(ModelDesc):

    def __init__(self, k=128, add_wrong=False):
        self.k = k  # match version was 192
        self.add_wrong = add_wrong  # match version was 192

    def _get_inputs(self):
        return [InputDesc(tf.int32, (None, 8 * NUM_PLANES, SHAPE, SHAPE), 'feature_planes'),
                InputDesc(tf.int32, (None, 8), 'labels'),
                InputDesc(tf.int32, (None, 8, SHAPE, SHAPE), 'labels_2d')]

    def _build_graph(self, inputs):
        feature_planes, labels, labels_2d = inputs

        feature_planes = tf.cast(feature_planes, tf.float32)
        feature_planes = tf.reshape(feature_planes, [-1, NUM_PLANES, SHAPE, SHAPE])
        feature_planes = tf.placeholder_with_default(feature_planes, [None, NUM_PLANES, SHAPE, SHAPE],
                                                     name='board_plhdr')

        labels = tf.reshape(labels, [-1])
        labels_2d = tf.reshape(labels_2d, [-1, SHAPE, SHAPE])

        def pad(x, p, name):
            return tf.pad(x, [[0, 0], [0, 0], [p, p], [p, p]], mode='CONSTANT', name=name)

        net = feature_planes
        with argscope([Conv2D], nl=tf.nn.relu, kernel_shape=3, padding='VALID',
                      stride=1, use_bias=False, data_format='NCHW', out_channel=self.k):
            net = pad(net, p=2, name='pad1')
            net = Conv2D('conv1', net, kernel_shape=5)

            net = Conv2D('conv2', pad(net, p=1, name='pad2'))
            net = Conv2D('conv3', pad(net, p=1, name='pad3'))
            net = Conv2D('conv4', pad(net, p=1, name='pad4'))
            net = Conv2D('conv5', pad(net, p=1, name='pad5'))
            net = Conv2D('conv6', pad(net, p=1, name='pad6'))
            net = Conv2D('conv7', pad(net, p=1, name='pad7'))
            net = Conv2D('conv8', pad(net, p=1, name='pad8'))
            net = Conv2D('conv9', pad(net, p=1, name='pad9'))
            net = Conv2D('conv10', pad(net, p=1, name='pad10'))
            net = Conv2D('conv11', pad(net, p=1, name='pad11'))
            net = Conv2D('conv12', pad(net, p=1, name='pad12'))
            net = Conv2D('conv_final', net, out_channel=1, kernel_shape=1, use_bias=True, nl=tf.identity)

        prob = tf.nn.softmax(net, name='probabilities')
        logits = tf.reshape(net, [-1, 19 * 19], name='logits')
        # logits = tf.identity(batch_flatten(net), name='logits')
        labels_2d_flat = tf.reshape(labels_2d, [-1, 19 * 19], name='labels_2d_flat')
        # labels_2d_flat = tf.identity(batch_flatten(labels_2d), name='labels_2d_flat')

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_2d_flat)
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        self.cost = tf.reduce_mean(loss, name='total_costs')

        acc_top1 = accuracy(logits, labels, 1, name='accuracy-top1')
        acc_top5 = accuracy(logits, labels, 5, name='accuracy-top5')

        summary.add_moving_summary(acc_top1, acc_top5, self.cost)

        if self.add_wrong:
            wrong_top1 = prediction_incorrect(logits, labels, 1, name='wrong-top1')
            wrong_top1 = tf.reduce_mean(wrong_top1, name='train-error-top1')

            wrong_top5 = prediction_incorrect(logits, labels, 5, name='wrong-top5')
            wrong_top5 = tf.reduce_mean(wrong_top5, name='train-error-top1')

            summary.add_moving_summary(wrong_top1, wrong_top5)

        # visualization
        with tf.name_scope('visualization'):
            # show the board
            vis_pos = tf.expand_dims(feature_planes[:, 0, :, :] - feature_planes[:, 1, :, :], axis=-1)
            vis_pos = (vis_pos + 1) * 128
            vis_pos = tf.image.grayscale_to_rgb(vis_pos)

            # show logits
            vis_logits = net[:, 0, :, :]
            vis_logits -= tf.reduce_min(vis_logits)
            vis_logits /= tf.reduce_max(vis_logits)
            vis_logits = tf.reshape(vis_logits * 256, [-1, SHAPE, SHAPE, 1])
            vis_logits = tf.image.grayscale_to_rgb(vis_logits)

            vis_prob = prob[:, 0, :, :]
            # just for visualization
            vis_prob -= tf.reduce_min(vis_prob)
            vis_prob /= tf.reduce_max(vis_prob)
            vis_prob = tf.reshape(vis_prob * 256, [-1, SHAPE, SHAPE, 1])
            vis_prob = tf.image.grayscale_to_rgb(vis_prob)

            # convert labels to board representation
            viz_label = labels_2d[:, :, :]
            viz_label = tf.cast(viz_label, tf.float32)
            viz_label = tf.reshape(viz_label * 256, [-1, SHAPE, SHAPE, 1])
            viz_label = tf.image.grayscale_to_rgb(viz_label)

            viz = tf.concat([vis_pos, vis_logits, vis_prob, viz_label], axis=2)
            viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')

        tf.summary.image('pos, logits, prob, labels', viz, BATCH_SIZE)

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 0.003, summary=True)
        return tf.train.GradientDescentOptimizer(lr)
        # return tf.train.AdamOptimizer(lr)


def get_data(lmdb, shuffle=False, isTrain=False):
    df = LMDBDataPoint(lmdb, shuffle=isTrain)
    df = PrefetchData(df, 5000, 1)
    df = GameDecoder(df, random_move=True)
    df = DihedralGroup(df)
    if isTrain:
        df = PrefetchDataZMQ(df, min(20, multiprocessing.cpu_count()))
    df = BatchData(df, BATCH_SIZE, remainder=not isTrain)
    return df


def get_config(path, k, max_eval=None):
    logger.set_logger_dir(
        os.path.join('train_log',
                     'tfgo-policy_net-{}'.format(k)))

    df_train = get_data(os.path.join(path, 'go_train.lmdb'), shuffle=True, isTrain=True)
    df_val = get_data(os.path.join(path, 'go_val.lmdb'), shuffle=True, isTrain=False)
    if max_eval:
        df_val = FixedSizeData(df_val, max_eval)

    return TrainConfig(
        model=Model(k),
        dataflow=df_train,
        callbacks=[
            ModelSaver(),
            MaxSaver('validation_accuracy-top1'),
            MaxSaver('validation_accuracy-top5'),
            InferenceRunner(df_val, [ScalarStats('total_costs'),
                                     ScalarStats('accuracy-top1'),
                                     ScalarStats('accuracy-top5')]),
            # Use train_log/tfgo-policy_net-128/hyper.txt to control your parameters
            HumanHyperParamSetter('learning_rate'),
        ],
        extra_callbacks=[
            MovingAverageSummary(),
            ProgressBar(['tower0/total_costs:0', 'learning_rate:0',
                         'tower0/accuracy-top1:0', 'tower0/accuracy-top5:0']),
            MergeAllSummaries(),
            RunUpdateOps()
        ],
        steps_per_epoch=df_train.size(),
        max_epoch=1000,
    )


def eval(model_file, path, k, max_eval=None):
    df_val = get_data(os.path.join(path, 'go_val.lmdb'), shuffle=True, isTrain=False)
    if max_eval:
        df_val = FixedSizeData(df_val, max_eval)
    pred_config = PredictConfig(
        model=Model(k, add_wrong=True),
        session_init=get_model_loader(model_file),
        input_names=['feature_planes', 'labels', 'labels_2d'],
        output_names=['wrong-top1', 'wrong-top5']
    )
    pred = SimpleDatasetPredictor(pred_config, df_val)
    acc1, acc5 = RatioCounter(), RatioCounter()
    try:
        for o in pred.get_result():
            batch_size = o[0].shape[0]
            acc1.feed(o[0].sum(), batch_size)
            acc5.feed(o[1].sum(), batch_size)
    except Exception as e:
        print e
        from IPython import embed
        embed()
    err1 = (acc1.ratio) * 100
    err5 = (acc5.ratio) * 100
    print("Top1 Accuracy: {0:.2f}% Error: {1:.2f}% Random-Guess: ~0.44%".format(100 - err1, err1))
    print("Top5 Accuracy: {0:.2f}% Error: {1:.2f}% Random-Guess: ~2.00%".format(100 - err5, err5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', required=True)
    parser.add_argument('--load', help='path to checkpoint of model', default='train_log/tfgo-policy_net-1280625-181619/checkpoint')
    parser.add_argument('--path', help='path to directory containing "go_train.lmdb" and "go_val.lmdb"', default='/scratch_shared/wieschol/pro')
    parser.add_argument('--k', type=int, help='number_of_filters for network', choices=xrange(1, 256), default=128)
    parser.add_argument('--eval', help='evaluate accuracy on held-out games', action='store_true')
    parser.add_argument('--max_eval', help='number of games to evaluate on (optional)')
    args = parser.parse_args()

    NR_GPU = len(args.gpu.split(','))
    with change_gpu(args.gpu):

        if args.eval:
            BATCH_SIZE = 64
            eval(args.load, args.path, args.k, max_eval=args.max_eval)
            sys.exit()

        config = get_config(args.path, args.k, max_eval=args.max_eval)
        config.nr_tower = NR_GPU

        if args.load:
            config.session_init = SaverRestore(args.load, ignore=['learning_rate'])

        SyncMultiGPUTrainer(config).train()

