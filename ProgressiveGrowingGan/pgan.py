#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Florian Grimm
#          Vanessa Kirchner
#          Andreas Specker
#          Patrick Wieschollek <mail@patwie.com>


import os
import cv2
import argparse
import functools
import numpy as np
from termcolor import colored

from tensorpack.utils.gpu import get_nr_gpu
from tensorpack import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import tensorflow as tf
from tensorpack.tfutils.common import get_global_step_var
from tensorpack.tfutils.summary import add_moving_summary

from GAN import GANTrainer, GANModelDesc, MultiGPUGANTrainer
from conv2d import MyConv2D

"""an updated version of train.py (not tested)
"""


def my_auto_reuse_variable_scope(func):
    """
    A decorator which automatically reuses the current variable scope if the
    function has been called with the same variable scope before.
    Remarks:
        This requires tf.__version__ >= 1.4.
        Tensorpack only keeps track of the *entire* variable scope.  `tf.AUTO_REUSE`
        however, does it on each variable separately, which is in this case much nicer.
    Examples:
    .. code-block:: python
        @my_auto_reuse_variable_scope
        def myfunc(x, i):
            if i > 0:
                x = tf.layers.conv2d(x, 128, 3)
            return tf.layers.conv2d(x, 128, 3)
        myfunc(x, 0)  # will reuse
        with tf.variable_scope('newscope'):
            myfunc(x, 1)  # does only work with `my_auto_reuse_variable_scope`
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            return func(*args, **kwargs)

    return wrapper


# monkey-patch
Conv2D = MyConv2D


def combine_img(current, previous, alpha):
    return alpha * current + (1 - alpha) * previous


def visualize_images(name, *imgs):
    """Generate tensor for TensorBoard (casting, clipping)

    Args:
        name: name for visualiaztion operation
        *imgs: multiple images in *args style

    Example:

        visualize_images('viz1', img1)
        visualize_images('viz2', img1, img2, img3)
    """
    xy = (tf.concat(imgs, axis=2) + 1.) * 128.
    xy = tf.cast(tf.clip_by_value(xy, 0, 255), tf.uint8, name='viz')
    tf.summary.image(name, xy, max_outputs=30)


# Do not change these params below, unless you know what you are doing
# -------------------------------------------------
# will be set later (these NEED to be None! to avoid accidental usage):
TRANSITION = None
BLOCKS = None
BATCH_SIZE = None
SHAPE = None
EPOCHS = None
STEPS_PER_EPOCH = None

NOISE_DIM = 512
LR = 0.001
# not sure about maximal batch sizes
# paper uses P100 with 16g, but we only have 12gb/12gb/11gb in TitanX/TitanXp/Gtx 1080Ti)
BATCH_SIZES = [16, 16, 16, 16, 8, 8, 4, 2, 1]
channels = [512, 512, 512, 512, 256, 128, 64, 32, 16, 8, 4, 1]
shapes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
# they say they process 600k images
NUM_IMAGES = 600000
# the celebHQ has only 30k images
DB_ENTRIES = 30000
# drift loss on discriminator part
EPS_DRIFT = 0.001


@layer_register(log_shape=True)
def Upsample(x, factor=2):
    """An alias for nn-upsampling
    """
    _, h, w, _ = x.get_shape().as_list()
    x = tf.image.resize_nearest_neighbor(x, [int(factor * h), int(factor * w)], align_corners=True)
    return x


def Downsample(name, x, factor=2):
    """An alias for nn-downsampling
    """
    assert factor == 2
    return tf.layers.average_pooling2d(x, factor, factor)


@layer_register(use_scope=None)
def PixelwiseNorm(x, eps=1e-8):
    # "N is the number of feature maps" -> along axis=3
    scale = tf.sqrt(tf.reduce_mean(x**2, axis=3, keepdims=True) + eps)
    return x / scale


@layer_register(log_shape=True)
def MiniBatchStd(x, eps=1e-8):
    # mbstat_avg = 'all',  # Which dimensions to average the statistic over?
    _, h, w, _ = x.get_shape().as_list()
    _, var = tf.nn.moments(x, axes=[0], keep_dims=True)
    stddev = tf.sqrt(tf.reduce_mean(var, keepdims=True))  # new arg-keyword
    y = tf.tile(stddev, [BATCH_SIZE, h, w, 1])
    return tf.concat([x, y], axis=3)


class Model(GANModelDesc):

    def _get_inputs(self):
        if TRANSITION:
            # the input is a pair (face, down_sampled_face)
            return [InputDesc(tf.float32, (None, SHAPE, SHAPE, 3), 'face'),
                    InputDesc(tf.float32, (None, SHAPE // 2, SHAPE // 2, 3), 'face_smaller')]
        else:
            # no transition, just a single face at a specific resolution
            return [InputDesc(tf.float32, (None, SHAPE, SHAPE, 3), 'face')]

    @auto_reuse_variable_scope
    def generator(self, latent_vector, alpha=None):
        """Represents the generator sub-modul of the network

        Args:
            latent_vector: tensor of incoming latent_vector

        Remarks:

            block_id    description         act.     output size        parameters

               0        Latent vector       –        512 × 1 × 1        –
               0        Conv 4 × 4          LReLU    512 × 4 × 4        4.2M
               0        Conv 3 × 3          LReLU    512 × 4 × 4        2.4M

               1        Upsample            –        512 × 8 × 8        –
               1        Conv 3 × 3          LReLU    512 × 8 × 8        2.4M
               1        Conv 3 × 3          LReLU    512 × 8 × 8        2.4M

               2        Upsample            –        512 × 16 × 16      –
               2        Conv 3 × 3          LReLU    512 × 16 × 16      2.4M
               2        Conv 3 × 3          LReLU    512 × 16 × 16      2.4M

               3        Upsample            –        512 × 32 × 32      –
               3        Conv 3 × 3          LReLU    512 × 32 × 32      2.4M
               3        Conv 3 × 3          LReLU    512 × 32 × 32      2.4M

               4        Upsample            –        512 × 64 × 64      –
               4        Conv 3 × 3          LReLU    256 × 64 × 64      1.2M
               4        Conv 3 × 3          LReLU    256 × 64 × 64      590k

               5        Upsample            –        256 × 128 × 128    –
               5        Conv 3 × 3          LReLU    128 × 128 × 128    295k
               5        Conv 3 × 3          LReLU    128 × 128 × 128    148k

               6        Upsample            –        128 × 256 × 256    –
               6        Conv 3 × 3          LReLU    64 × 256 × 256     74k
               6        Conv 3 × 3          LReLU    64 × 256 × 256     37k

               7        Upsample            –        64 × 512 × 512     –
               7        Conv 3 × 3          LReLU    32 × 512 × 512     18k
               7        Conv 3 × 3          LReLU    32 × 512 × 512     9.2k

               8        Upsample            –        32 × 1024 × 1024   –
               8        Conv 3 × 3          LReLU    16 × 1024 × 1024   4.6k
               8        Conv 3 × 3          LReLU    16 × 1024 × 1024   2.3k

               0        Conv 1 × 1          linear 3 × 1024 × 1024      51

            Total trainable parameters 23.1M

        Returns:
            current last prediction
        """
        with argscope(Conv2D, activation=tf.nn.leaky_relu, stride=1):
            end_points = []

            logger.info("")
            logger.info("generator")
            logger.info("-------------------------")

            def up_block(x, block_id):
                with tf.name_scope("up_block_%i" % block_id):
                    assert block_id in range(9)

                    # the first block need to march on a different tune, very intutitiv ?!
                    usample_factor = 4 if block_id == 0 else 2

                    h_in, w_in, c_in = x.get_shape().as_list()[1:]
                    h_out, w_out, c_out = h_in * usample_factor, w_in * usample_factor, channels[block_id]

                    logger.info("genenerator-block %i (%i x %i x %i) --> (%i x %i x %i)" %
                                (block_id, c_in, h_in, w_in, c_out, h_out, w_out))

                    if block_id == 0:
                        # Latent vector - Conv 4 × 4       - Conv 3 × 3
                        # http://lasagne.readthedocs.io/en/latest/modules/layers/conv.html
                        # lasagne::pad==FULL "'full' pads with one less than the filter size on both sides."
                        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
                        # PN(WS(Conv2DLayer(...)
                        x = Conv2D('block_%03i_0' % block_id, x, channels[block_id], kernel_shape=4, padding="VALID")
                        x = PixelwiseNorm('PixelNorm_%03i_0' % block_id, x)
                        x = Conv2D('block_%03i_1' % block_id, x, channels[block_id], kernel_shape=3)
                        x = PixelwiseNorm('PixelNorm_%03i_1' % block_id, x)
                    else:
                        # Upsample      - Conv 3 × 3       - Conv 3 × 3
                        x = Upsample("upsample_block_%i" % block_id, x, factor=usample_factor)
                        x = Conv2D('block_%03i_0' % block_id, x, channels[block_id], kernel_shape=3)
                        x = PixelwiseNorm('PixelNorm_%03i_0' % block_id, x)
                        x = Conv2D('block_%03i_1' % block_id, x, channels[block_id], kernel_shape=3)
                        x = PixelwiseNorm('PixelNorm_%03i_1' % block_id, x)

                    return x

            # reshape latent random vector ...
            x = tf.reshape(latent_vector, [-1, 1, 1, 512])
            x = PixelwiseNorm('norm_latent', x)
            # consecutively upsample the images
            for block_id in range(BLOCKS):
                x = up_block(x, block_id)
                end_points.append(x)

            fake_img = self.toRGB(end_points[-1], BLOCKS)

            if TRANSITION:
                fake_prev = self.toRGB(end_points[-2], BLOCKS - 1)
                fake_prev = Upsample("upsample_prev", fake_prev, factor=2)
                fake_img = combine_img(fake_img, fake_prev, alpha)

            return fake_img

    @auto_reuse_variable_scope
    def discriminator(self, observed_image, alpha=None):
        """Represents the discrimnator sub-modul of the network

        Args:
            observed_image: incoming imgae that should be judge being (real/forged)
            prev_image (None, optional): Description
            alpha (None, optional): Description

        Remarks:

            block_id

               8        Input image         –         3 × 1024 × 1024   –
               8        Conv 1 × 1          LReLU    16 × 1024 × 1024   64
               8        Conv 3 × 3          LReLU    16 × 1024 × 1024   2.3k
               8        Conv 3 × 3          LReLU    32 × 1024 × 1024   4.6k
               8        Downsample          –        32 × 512 × 512     –

               7        Conv 3 × 3          LReLU    32 × 512 × 512     9.2k
               7        Conv 3 × 3          LReLU    64 × 512 × 512     18k
               7        Downsample          –        64 × 256 × 256     –

               6        Conv 3 × 3          LReLU    64 × 256 × 256     37k
               6        Conv 3 × 3          LReLU    128 × 256 × 256    74k
               6        Downsample          –        128 × 128 × 128    –

               5        Conv 3 × 3          LReLU    128 × 128 × 128    148k
               5        Conv 3 × 3          LReLU    256 × 128 × 128    295k
               5        Downsample          –        256 × 64 × 64      –

               4        Conv 3 × 3          LReLU    256 × 64 × 64      590k
               4        Conv 3 × 3          LReLU    512 × 64 × 64      1.2M
               4        Downsample          –        512 × 32 × 32      –

               3        Conv 3 × 3          LReLU    512 × 32 × 32      2.4M
               3        Conv 3 × 3          LReLU    512 × 32 × 32      2.4M
               3        Downsample          –        512 × 16 × 16      –

               2        Conv 3 × 3          LReLU    512 × 16 × 16      2.4M
               2        Conv 3 × 3          LReLU    512 × 16 × 16      2.4M
               2        Downsample          –        512 × 8 × 8        –

               1        Conv 3 × 3          LReLU    512 × 8 × 8        2.4M
               1        Conv 3 × 3          LReLU    512 × 8 × 8        2.4M
               1        Downsample          –        512 × 4 × 4        –

               0        Minibatch stddev    –        513 × 4 × 4        –
               0        Conv 3 × 3          LReLU    512 × 4 × 4        2.4M
               0        Conv 4 × 4          LReLU    512 × 1 × 1        4.2M
               0        Fully-connected     linear     1 × 1 × 1        513

        Returns:
            TYPE: Description
        """

        with argscope(FullyConnected, W_init=tf.truncated_normal_initializer(stddev=0.02)):
            with argscope(Conv2D, activation=tf.nn.leaky_relu, stride=1):

                x = self.fromRGB(observed_image, BLOCKS)

                logger.info("")
                logger.info("discriminator")
                logger.info("-------------------------")

                logger.info("discriminator input 1/1 {}".format(x.get_shape()))

                def down_block(x, block_id, last_activation=tf.nn.leaky_relu):
                    with tf.name_scope("down_block_%i" % block_id):
                        h_in, w_in, c_in = x.get_shape().as_list()[1:]
                        downsample_factor = 2 if block_id > 0 else 4
                        channel_multiplier = 2 if block_id > 3 else 1

                        h_out, w_out, c_out = h_in // downsample_factor, w_in // downsample_factor, channels[
                            block_id] * channel_multiplier
                        logger.info("discriminator-block %i (%i x %i x %i) --> (%i x %i x %i)" %
                                    (block_id, c_in, h_in, w_in, c_out, h_out, w_out))

                        with argscope(Conv2D, kernel_shape=3):

                            if block_id == 0:
                                # is last block in discriminator
                                x = MiniBatchStd('MiniBatchStd', x)
                                x = Conv2D('block_%03i_0' % block_id, x, 512, kernel_shape=3, padding='SAME')
                                x = Conv2D('block_%03i_1' % block_id, x, 512, kernel_shape=4, padding='VALID')
                            else:
                                x = Conv2D('block_%03i_0' % block_id, x, channels[block_id])
                                x = Conv2D('block_%03i_1' % block_id, x, channels[block_id] * channel_multiplier)
                                x = Downsample('downsample_%03i' % block_id, x)

                        return x

                # starting by last stage_id: 8, 7, 6, 5, 4, 3, 2, 1, 0
                #                        cc: 0, 1, 2, 3, 4, 5, 6, 7, 8
                for cc, block_id in enumerate(reversed(range(BLOCKS))):
                    x = down_block(x, block_id)
                    if cc == 0 and TRANSITION:
                        # need to fade out previous image
                        #   [fromRGB, 32x32, 0.5x] + left branch in Figure 2 (b)
                        prev_image = Downsample('downsample_prev', observed_image)
                        prev_image = self.fromRGB(prev_image, BLOCKS - 1)
                        x = alpha * x + (1 - alpha) * prev_image

                return FullyConnected('fc', x, 1, activation=tf.identity)

    @my_auto_reuse_variable_scope
    def toRGB(self, features, access_stage):
        """Mapping from features to 3-channel output

        Args:
            x: features from layer activations
            access_stage: number of stages when growing

        Returns:
            TYPE: Description
        """
        return Conv2D('toRGB_%03i' % (access_stage - 1), features, 3, kernel_shape=1, activation=tf.identity)

    @my_auto_reuse_variable_scope
    def fromRGB(self, x, access_stage):
        """Summary

        Args:
            x (TYPE): images from layer activations
            access_stage: number of stages when growing

        Returns:
            TYPE: Description
        """
        assert len(x.get_shape().as_list()) == 4
        assert x.get_shape().as_list()[-1] == 3

        x = Conv2D('fromRGB_%03i' % (access_stage - 1), x, channels[access_stage - 1], kernel_shape=1,
                   activation=tf.nn.leaky_relu)
        return x

    def _build_graph(self, inputs):

        # keep track of statistics to interpolate between different checkpoints
        # from 0 (only old checkpoint) to 1 (only new network)
        glbstep = tf.identity(get_global_step_var(), name="glob_step")
        seen_images = tf.cast(get_global_step_var() * BATCH_SIZE, tf.int64, name='seen_images')
        if TRANSITION:
            alpha = tf.divide(tf.cast(seen_images, tf.float32), tf.cast(NUM_IMAGES, tf.float32), name='alpha')
            transition_phase = tf.get_variable('transistion_phase', initializer=1., trainable=False, dtype=tf.float32)
        else:
            alpha = tf.identity(0, name='alpha')
            transition_phase = tf.get_variable('transistion_phase', initializer=0., trainable=False, dtype=tf.float32)
        add_moving_summary(alpha, seen_images, tf.identity(transition_phase, name="transistion"), glbstep)

        if TRANSITION:
            real_img, real_prev = inputs[0] / 128.0 - 1, Upsample("upsample_realprev", inputs[1] / 128.0 - 1, factor=2)
            real_img = combine_img(real_img, real_prev, alpha)
        else:
            real_prev = None
            real_img, real_prev = inputs[0] / 128.0 - 1, None

        # noise which the generator is starting from
        z = tf.random_uniform([BATCH_SIZE, NOISE_DIM], -1, 1, name='z_train')
        z = tf.placeholder_with_default(z, [None, NOISE_DIM], name='z')

        # GENERATOR
        # ---------------------------------------------------------------------
        with tf.variable_scope('gen'):
            fake_img = self.generator(z, alpha=alpha)
            visualize_images('real_fake', real_img, fake_img)

            fake_output = (fake_img + 1.) * 128.
            fake_output = tf.cast(tf.clip_by_value(fake_output, 0, 255), tf.uint8, name='viz')
            tf.identity(fake_output, name='fake_img')

        # DISCRIMINATOR
        # ---------------------------------------------------------------------
        with tf.variable_scope('discrim'):
            WGAN_alpha = tf.random_uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0., maxval=1., name='alpha')
            interp_img = real_img + WGAN_alpha * (fake_img - real_img)

            visualize_images('real_fake_interp', real_img, fake_img, interp_img)

            real_score = self.discriminator(real_img, alpha=alpha)
            fake_score = self.discriminator(fake_img, alpha=alpha)
            interp_score = self.discriminator(interp_img, alpha=alpha)

            mean_real_score = tf.reduce_mean(real_score, name='mean_real_score')
            mean_fake_score = tf.reduce_mean(fake_score, name='mean_fake_score')
            mean_interp_score = tf.reduce_mean(interp_score, name='mean_interp_score')
            add_moving_summary(mean_real_score, mean_fake_score, mean_interp_score)

        # the Wasserstein-GAN losses
        self.d_loss = tf.reduce_mean(fake_score - real_score, name='d_loss')
        self.g_loss = tf.negative(tf.reduce_mean(fake_score), name='g_loss')
        loss_diff = tf.subtract(self.g_loss, self.d_loss, name="loss-diff-g-d")
        add_moving_summary(self.d_loss, self.g_loss, loss_diff)

        # the gradient penalty loss
        def wasserstein_grad_penalty(score, input, name=None):
            with tf.name_scope(name):
                gradients = tf.gradients(score, [input])[0]
                gradients = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))
                gradients_rms = symbolic_functions.rms(gradients, 'gradient_rms')
                gradient_penalty = tf.reduce_mean(tf.square(gradients - 1), name='gradient_penalty')
                return gradients_rms, gradient_penalty

        gradients_rms, gradient_penalty = wasserstein_grad_penalty(interp_score, interp_img)
        add_moving_summary(gradient_penalty, gradients_rms)

        # drift-loss
        drift_loss = tf.reduce_mean(tf.square(real_score), name='drift_loss')
        self.d_loss = tf.add_n([self.d_loss, 10 * gradient_penalty, EPS_DRIFT * drift_loss], name='total_d_loss')
        add_moving_summary(self.d_loss, drift_loss)

        self.collect_variables()

        def count_params_in_scope(scope):
            vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            return np.sum([int(np.prod(v.shape)) for v in vs])

        logger.info(colored("Number of Parameters:", 'cyan'))
        logger.info("generator #params:     {:,}".format(count_params_in_scope('gen')))
        logger.info("discriminator #params: {:,}".format(count_params_in_scope('discrim')))

    def _get_optimizer(self):
        # "We train the networks using Adam (Kingma & Ba, 2015) with α = 0.001, β1 = 0, β2 = 0.99, and eps = 10−8."
        # return tf.train.AdamOptimizer(learning_rate=LR, beta1=0, beta2=0.99, epsilon=1e-8)
        return tf.train.RMSPropOptimizer(1e-4)


class ImageDecode(MapDataComponent):
    def __init__(self, ds, dtype=np.uint8, index=0):
        def func(im_data):
            img = cv2.imdecode(np.asarray(bytearray(im_data), dtype=dtype), cv2.IMREAD_COLOR)
            return img
        super(ImageDecode, self).__init__(ds, func, index=index)


def get_data(lmdb):
    from PIL import Image
    import os
    assert os.path.isfile(lmdb)

    def resize(img, shp=4):
        img = img.resize((shp, shp), Image.ANTIALIAS)
        return np.array(img)

    def bgr2rgb(x):
        return x[:, :, ::-1]

    ds = LMDBDataPoint(lmdb, shuffle=True)
    ds = ImageDecode(ds, index=0)

    ds = MapDataComponent(ds, bgr2rgb, index=0)
    if TRANSITION:
        ds = MapData(ds, lambda x: [resize(Image.fromarray(x[0]), SHAPE),
                                    resize(Image.fromarray(x[0]), SHAPE // 2)])
    else:
        ds = MapData(ds, lambda x: [resize(Image.fromarray(x[0]), SHAPE)])

    ds = PrefetchDataZMQ(ds, 12)
    ds = BatchData(ds, BATCH_SIZE)
    return ds


def sample(model_path):
    import cv2
    import tqdm
    pred = OfflinePredictor(PredictConfig(
        session_init=get_model_loader(model_path),
        model=Model(),
        input_names=['z'],
        output_names=['gen/fake_img']))

    np.random.seed(42)

    # CUDA_VISIBLE_DEVICES=3 python pgan.py --gpu 2 --blocks 6 --batch_size 4 \
    #     --load /graphics/scratch/wieschol/CHECKPOINTS/pgan/block6_b/checkpoint --sample

    for i in tqdm.tqdm(range(400)):
        fake_img = pred(np.random.uniform(low=-1.0, high=1.0, size=[4, NOISE_DIM]))[0][0]
        # fake_img = pred()[0][0]
        fake_img = np.clip(fake_img[:, :, ::-1], 0, 255).astype(np.uint8)
        cv2.imwrite('/tmp/fake%03i.jpg' % i, fake_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--action', help='load model', default="")
    parser.add_argument('--transition', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--blocks', default=1)
    parser.add_argument('--batch_size', default=-1, type=int)
    parser.add_argument('--lr', help='learning rate', default=0.001, type=np.float32)
    parser.add_argument('--lmdb', help='', default='/graphics/scratch/datasets/celebHQ/celeb_hq_256.lmdb')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    TRANSITION = True if args.transition else False
    LR = args.lr
    BLOCKS = int(args.blocks)
    assert BLOCKS < 10
    if BLOCKS == 1:
        assert TRANSITION is False

    if args.batch_size > 0:
        BATCH_SIZE = args.batch_size
    else:
        BATCH_SIZE = BATCH_SIZES[BLOCKS - 1]
    SHAPE = shapes[BLOCKS - 1]
    if not args.debug:
        STEPS_PER_EPOCH = DB_ENTRIES // BATCH_SIZE
        EPOCHS = NUM_IMAGES // (STEPS_PER_EPOCH * BATCH_SIZE)
    else:
        STEPS_PER_EPOCH = 3
        EPOCHS = 2

    if args.sample:
        sample(args.load)
    else:
        logger.auto_set_dir()

        nr_tower = max(get_nr_gpu(), 1)
        data = QueueInput(get_data(args.lmdb))
        model = Model()

        logger.info("run %i epochs", EPOCHS)
        logger.info("use %i blocks", BLOCKS)
        logger.info("use %i as batchsize", BATCH_SIZE)

        if nr_tower == 1:
            trainer = GANTrainer(data, model)
        else:
            trainer = MultiGPUGANTrainer(nr_tower, data, model)

        callbacks = [
            ModelSaver(),
            MovingAverageSummary(),
            ProgressBar(['d_loss', 'g_loss', 'alpha', 'loss-diff-g-d']),
            MergeAllSummaries(),
            RunUpdateOps()
        ]

        trainer.train_with_defaults(
            callbacks=callbacks,
            session_init=SaverRestore(args.load, ignore=['global_step']) if args.load else None,
            steps_per_epoch=STEPS_PER_EPOCH,
            max_epoch=EPOCHS,
        )
