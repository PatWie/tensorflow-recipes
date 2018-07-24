from zipfile import ZipFile
import cv2
import os
import numpy as np
import scipy.io as sio
from tensorpack import *
import argparse

"""
python data_sampler.py --zip /tmp/WIDER_val.zip \
                       --mat /tmp/wider_face_val.mat \
                       --lmdb /tmp/WIDER_val.lmdb
img buffer is RGB
"""


def draw_rect(img, top, left, bottom, right, rgb, margin=1):
    m = margin
    r, g, b = rgb
    img[top:bottom, left - m:left + m, 0] = r
    img[top:bottom, left - m:left + m, 1] = g
    img[top:bottom, left - m:left + m, 2] = b

    img[top:bottom, right - m:right + m, 0] = r
    img[top:bottom, right - m:right + m, 1] = g
    img[top:bottom, right - m:right + m, 2] = b

    img[top - m:top + m, left:right, 0] = r
    img[top - m:top + m, left:right, 1] = g
    img[top - m:top + m, left:right, 2] = b

    img[bottom - m:bottom + m, left:right, 0] = r
    img[bottom - m:bottom + m, left:right, 1] = g
    img[bottom - m:bottom + m, left:right, 2] = b

    return img


class RawWiderFaceReader(RNGDataFlow):
    """Read images directly from tar file without unpacking
    boxes: left, top, width, height
    """

    def __init__(self, matfile, zipfile):
        super(RawWiderFaceReader, self).__init__()
        assert os.path.isfile(matfile)
        assert os.path.isfile(zipfile)
        self.matfile = matfile
        self.zipfile = zipfile
        self.subset = matfile.split('_')[-1].replace('.mat', '')
        f = sio.loadmat(matfile)
        events = [f['event_list'][i][0][0] for i in range(len(f['event_list']))]
        raw_files = [f['file_list'][i][0] for i in range(len(f['file_list']))]
        raw_bbx = [f['face_bbx_list'][i][0] for i in range(len(f['face_bbx_list']))]

        col_files = []
        for file, bbx in zip(raw_files, raw_bbx):
            for filee, bbxe in zip(file, bbx):
                col_files.append((filee[0][0], bbxe[0]))

        self.col_files2 = []
        for file, bbx in col_files:
            for ev in events:
                if file.startswith(ev.replace('--', '_')):
                    self.col_files2.append((str('WIDER_%s/images/' % self.subset + ev +
                                                '/' + file + '.jpg').encode('ascii', 'ignore'), bbx))
                    break

    def get_data(self):
        with ZipFile(self.zipfile, 'r') as zip_hnd:
            for fn, bbx in self.col_files2:
                print fn
                buf = zip_hnd.read('%s' % fn)
                yield [buf, bbx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb', help='path to database (to be written)')
    parser.add_argument('--zip', help='path to database (to be red)',
                        default='WIDER_train.zip')
    parser.add_argument('--mat', help='path to database (to be red)',
                        default='wider_face_split/wider_face_train.mat')
    parser.add_argument('--debug', action='store_true',
                        help='just show the images')
    args = parser.parse_args()

    if args.debug:
        ds = RawWiderFaceReader(matfile=args.mat, zipfile=args.zip)
        ds.reset_state()
        for jpeg, bbx in ds.get_data():
            rgb = cv2.imdecode(np.asarray(bytearray(jpeg), dtype=np.uint8), cv2.IMREAD_COLOR)
            for bb in bbx:
                left, top, width, height = bb
                right = left + width
                bottom = top + height

                rgb = draw_rect(rgb, top, left, bottom, right,
                                np.random.uniform(low=0, high=255, size=(3)), 10)
            rgb = cv2.resize(rgb, (300, 300))
            cv2.imshow("image with bb", rgb)
            cv2.waitKey(0)
    else:
        ds = RawWiderFaceReader(matfile=args.mat, zipfile=args.zip)
        ds.reset_state()
        LMDBSerializer.save(ds, args.lmdb)
