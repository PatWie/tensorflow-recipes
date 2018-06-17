#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepares dataset for faster reading.

see: https://github.com/cchen156/Learning-to-See-in-the-Dark/blob/master/download_dataset.py
"""

import argparse
import numpy as np
import rawpy
import os
import glob
import tqdm


# source: https://github.com/cchen156/Learning-to-See-in-the-Dark/blob/0c30dec046a82e1f4d6167eb060a35b3f1625f67/train_Sony.py#L100-L114
def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible
    # we store uint16 on disc saving space
    # im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/scratch/wieschol/seeindark/dataset/Sony')
    parser.add_argument('--subset', type='str',
                        default='train')
    args = parser.parse_args()

    assert os.path.isdir(args.root_dir)
    assert args.subset in ['train', 'test']

    prefix = 0 if args.subset == 'train' else 1
    files = glob.glob(os.path.join(args.root_dir, 'long', '%i*.ARW' % prefix))
    ids = [int(f.replace(os.path.join(args.root_dir, 'long', '%i' % prefix), '').split('_')[0])
           for f in files]


    # pre-process long exposure time files
    for file_id in tqdm.tqdm(ids):
        filename_long = os.path.join(args.root_dir, 'long', '%i%04i_00_10s.ARW' % (prefix, file_id))
        if not os.path.isfile(filename_long):
            filename_long = os.path.join(args.root_dir, 'long', '%i%04i_00_30s.ARW' % (prefix, file_id))
        assert os.path.isfile(filename_long), filename_long
        outfile = filename_long.replace('.ARW', '.npy')

        if not os.path.isfile(outfile):
            gt_raw = rawpy.imread(filename_long)
            gt_uint16 = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            np.save(outfile, gt_uint16)


    # pre-process short exposure time files
    for file_id in tqdm.tqdm(ids):
        filename_long = os.path.join(args.root_dir, 'long', '%i%04i_00_10s.ARW' % (prefix, file_id))
        if not os.path.isfile(filename_long):
            filename_long = os.path.join(args.root_dir, 'long', '%i%04i_00_30s.ARW' % (prefix, file_id))

        filenames_short = glob.glob(os.path.join(args.root_dir, 'short', '%i%04i_00*.ARW' % (prefix, file_id)))

        for filename_short in filenames_short:
            assert os.path.isfile(filename_short), filename_short
            outfile = filename_short.replace('.ARW', '.npy')

            if not os.path.isfile(outfile):
                input_raw = rawpy.imread(filename_short)
                input_uint16 = pack_raw(input_raw)
                np.save(outfile, input_uint16)
