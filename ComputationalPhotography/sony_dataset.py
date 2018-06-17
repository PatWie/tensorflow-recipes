#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset reader for the Sony Dataset from the paper
Learning to see in the dark

see: https://github.com/cchen156/Learning-to-See-in-the-Dark/blob/master/download_dataset.py
"""

import numpy as np
import os
import glob
from tensorpack import *
import tensorpack.dataflow.imgaug as imgaug


class SonyDataset(RNGDataFlow):
    """ Produce raw input data (short and long exposure times). """

    def __init__(self, root_dir, shuffle=True, subset='train'):
        """Summary

        Args:
            root_dir (string): path to directory containing "short", "long"
            shuffle (bool, optional): Description
        """
        assert os.path.isdir(root_dir)
        self.shuffle = shuffle
        self.root_dir = root_dir
        self.subset = subset

        self.prefix = 0 if subset == 'train' else 1
        self.files = glob.glob(os.path.join(root_dir, 'long', '%i*.npy' % self.prefix))
        self.ids = [int(f.replace(os.path.join(root_dir, 'long', '%i' % self.prefix), '').split('_')[0])
                    for f in self.files]

    def size(self):
        return len(self.ids)

    def get_data(self):
        permutation = list(range(self.size()))
        if self.shuffle:
            self.rng.shuffle(permutation)

        for p in permutation:
            file_id = self.ids[p]
            filename_long = os.path.join(self.root_dir, 'long', '%i%04i_00_10s.npy' % (self.prefix, file_id))
            if not os.path.isfile(filename_long):
                filename_long = os.path.join(self.root_dir, 'long', '%i%04i_00_30s.npy' % (self.prefix, file_id))
            assert os.path.isfile(filename_long), filename_long

            # choose random short exposure-time
            filenames_short = glob.glob(os.path.join(self.root_dir, 'short', '%i%04i_00*.npy' % (self.prefix, file_id)))
            filename_short = filenames_short[self.rng.randint(len(filenames_short))]
            assert os.path.isfile(filename_short), filename_short

            exposuretime_long = float(filename_long.replace(os.path.join(self.root_dir, 'long'), '')[10:-5])
            exposuretime_short = float(filename_short.replace(os.path.join(self.root_dir, 'short'), '')[10:-5])

            factor = min(exposuretime_long / exposuretime_short, 300)

            gt_uint16 = np.load(filename_long)
            gt_float = (gt_uint16 / 65535.0).astype(np.float32)

            input_uint16 = np.load(filename_short)
            input_float = input_uint16.astype(np.float32)
            # black level subtraction
            input_float = np.maximum(input_float - 512, 0) / (16383 - 512)
            input_float = input_float * factor

            yield [gt_float, input_float]


class RandomCropRaw(ProxyDataFlow):
    def __init__(self, ds, patch_size=512):
        super(RandomCropRaw, self).__init__(ds)
        self.aug = imgaug.RandomCrop(patch_size)

    def reset_state(self):
        self.ds.reset_state()

    def size(self):
        return self.ds.size()

    def get_data(self):
        for dp in self.ds.get_data():
            gt_float, input_float = dp[:2]
            shp = self.aug._get_augment_params(input_float)
            input_float = input_float[shp.h0: shp.h0 + shp.h, shp.w0: shp.w0 + shp.w, :]
            gt_float = gt_float[shp.h0 * 2: shp.h0 * 2 + shp.h * 2, shp.w0 * 2: shp.w0 * 2 + shp.w * 2, :]

            yield [gt_float, input_float]


if __name__ == '__main__':
    ds = SonyDataset('/scratch/wieschol/seeindark/dataset/Sony')
    ds = RandomCropRaw(ds)
    aus = [imgaug.Flip(horiz=True), imgaug.Flip(vert=True), imgaug.Transpose()]
    ds = AugmentImageComponents(ds, aus, index=(0, 1), copy=False)
    ds.reset_state()
    next(ds.get_data())
    ds = PrefetchDataZMQ(ds, nr_proc=10)
    ds = BatchData(ds, 8)
    ds = PrintData(ds)
    ds.reset_state()
    next(ds.get_data())
    # TestDataSpeed(ds).start()
