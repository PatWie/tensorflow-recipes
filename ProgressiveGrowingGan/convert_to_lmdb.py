#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert CelebHQ into jpeg compressed LMDB files
"""

import argparse
import numpy as np
import cv2
import os
import h5py
import Image

from tensorpack import *


class ImageEncode(MapDataComponent):
    def __init__(self, ds, mode='.jpg', dtype=np.uint8, index=0):
        def func(img):
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return np.asarray(bytearray(cv2.imencode(mode, img)[1].tostring()), dtype=dtype)

        super(ImageEncode, self).__init__(ds, func, index=index)


class ImageDecode(MapDataComponent):
    def __init__(self, ds, dtype=np.uint8, index=0):
        def func(im_data):
           # return im_data
            img = cv2.imdecode(np.asarray(bytearray(im_data), dtype=dtype), cv2.IMREAD_COLOR)
            return img

        super(ImageDecode, self).__init__(ds, func, index=index)


class CelebAHQH5Reader(RNGDataFlow):
    """docstring for CelebAHQH5Reader"""

    def __init__(self, h5_file, shuffle=False):
        super(CelebAHQH5Reader, self).__init__()
        self.h5_file = h5_file
        self.shuffle = shuffle
        assert os.path.isfile(h5_file)
        logger.info('read %s' % h5_file)

    def get_data(self):

        h5 = h5py.File(self.h5_file, 'r')
        lods = sorted([value for key, value in h5.iteritems() if key.startswith('data')], key=lambda lod: -lod.shape[3])

        shape = lods[0].shape
        indices = range(shape[0])

        if self.shuffle:
            self.rng.shuffle(indices)

        # I tested the h5 --> contains only 3-channels images
        for idx in indices:
            img = lods[0][idx]
            img = img.transpose(1, 2, 0)  # CHW => HWC
            img = img[:, :, ::-1]  # RGB => BGR
            yield [img]


class resizeDataFlow(dataflow.DataFlow):
    def __init__(self, size):
        super(resizeDataFlow, self).__init__()
        self.image_size = int(size)
        self.remainingImages = -1

    def get_data(self):
        lmdb = "/datasets/celebHQ/celeb_hq.lmdb"
        ds = LMDBDataPoint(lmdb, shuffle=True)
        ds = ImageDecode(ds, index=0)
        ds.reset_state()
        resample = Image.BICUBIC

        self.remainingImages = ds.size()

        for dp in ds.get_data():
            #read image
            bgr = dp[0]

            #convert to Pil Image and resize

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(rgb)
            pil_im = pil_im.resize((self.image_size, self.image_size), resample=resample)

            #convert back to opencv fomat
            resized = np.array(pil_im)
            resized = resized[:, :, ::-1].copy()

            #beak for less images
            self.remainingImages -= 1

            print self.remainingImages
            # if (self.remainingImages < 29950):
            #     break
            yield [resized]


    def size(self):
        return self.remainingImages


def resize_lmdb_content(size):
    resizedLMDBPath = "/datasets/celebHQ/resized_" +  size + ".lmdb"
    dataflow = resizeDataFlow(size)
    dataflow = ImageEncode(dataflow, index=0)
    dftools.dump_dataflow_to_lmdb(dataflow, resizedLMDBPath)


def debug_h5(h5):
    ds = CelebAHQH5Reader(h5)
    ds.reset_state()

    for dp in ds.get_data():
        bgr = dp[0]
        cv2.imshow('winname', bgr)
        cv2.waitKey(0)


def debug_lmdb(lmdb):
    ##################################
    # THIS is the interesting part   #
    ##################################

    lmdb = "/datasets/celebHQ/resized.lmdb"
    ds = LMDBDataPoint(lmdb, shuffle=False)
    ds = ImageDecode(ds, index=0)
    ds.reset_state()

    gen = ds.get_data()
    for dp in gen:
        print("here i am")
        bgr = dp[0]
        cv2.imshow('winname', bgr)
        cv2.waitKey(0)


def convert(h5, lmdb):
    ds = CelebAHQH5Reader(args.h5)
    ds = ImageEncode(ds, index=0)
    dftools.dump_dataflow_to_lmdb(ds, lmdb)


def benchmark(h5, lmdb):
    ds = CelebAHQH5Reader(h5, shuffle=True)
    ds.reset_state()
    TestDataSpeed(ds, 500).start()

    ds = LMDBDataPoint(lmdb, shuffle=True)
    ds = ImageDecode(ds, index=0)
    ds.reset_state()
    TestDataSpeed(ds, 500).start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--str', help='dummy string', type=str)
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--benchmark', action='store_true', help='benchmark')
    parser.add_argument('--convert', action='store_true', help='debug')
    parser.add_argument('--h5', default="")
    parser.add_argument('--lmdb', default="")
    parser.add_argument('--resize', help='resize lmdb to other lmdb :)', default=400)

    args = parser.parse_args()

    if args.benchmark:
        benchmark(args.h5, args.lmdb)
    elif args.debug:
        if args.h5 is not '':
            debug_h5(args.h5)
        elif args.lmdb is not '':
            debug_lmdb(args.lmdb)
    elif args.convert:
        assert args.lmdb is not ""
        assert os.path.isfile(args.h5)
        convert(args.h5, args.lmdb)
    elif args.resize:
        resize_lmdb_content(args.resize)

