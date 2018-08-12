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

import models

enable_argscope_for_module(tf.layers)

"""
This is a tensorpack script re-implementation of
FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks
https://arxiv.org/abs/1612.01925

This is not an attempt to reproduce the lengthly training protocol,
but to rely on tensorpack's "OfflinePredictor" for easier inference.

The ported pre-trained Caffe-model are here
http://files.patwie.com/recipes/models/flownet2-s.npz
http://files.patwie.com/recipes/models/flownet2-c.npz

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

To run it on actual data:

    python flownet2.py --gpu 0 \
        --left 00001_img1.ppm \
        --right 00001_img2.ppm \
        --load flownet2-s.npz
        --model "flownet2-s"

"""


MODEL_MAP = {'flownet2-s': models.FlowNet2S,
             'flownet2-c': models.FlowNet2C,
             'flownet2': models.FlowNet2}


def apply(model_name, model_path, left, right, ground_truth=None):
    model = MODEL_MAP[model_name]
    left = cv2.imread(left).astype(np.float32).transpose(2, 0, 1)[None, ...]
    right = cv2.imread(right).astype(np.float32).transpose(2, 0, 1)[None, ...]

    predict_func = OfflinePredictor(PredictConfig(
        model=model(),
        session_init=get_model_loader(model_path),
        input_names=['left', 'right'],
        output_names=['prediction']))

    output = predict_func(left, right)[0].transpose(0, 2, 3, 1)
    flow = Flow()

    img = flow.visualize(output[0])
    if ground_truth is not None:
        img = np.concatenate([img, flow.visualize(Flow.read(ground_truth))], axis=1)

    cv2.imshow('flow output', img)
    cv2.imwrite('flownet2_full.jpg', img * 255)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--left', help='input', type=str)
    parser.add_argument('--right', help='input', type=str)
    parser.add_argument('--model', help='model', type=str)
    parser.add_argument('--gt', help='ground_truth', type=str, default=None)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    apply(args.model, args.load, args.left, args.right, args.gt)
