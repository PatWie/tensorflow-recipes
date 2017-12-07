## SuperResolution - RealTime

Re-implements
[Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)
by Shi et al.

Given an low-resolution image, the network is trained to
produce a 4x resolution image by shifting pixel from differen spatial locations into different channels.

### Usage

1. Download MS COCO dataset:

```bash
wget http://images.cocodataset.org/zips/train2017.zip
python data_sampler.py --lmdb train2017.lmdb --input train2017.zip --create
wget http://images.cocodataset.org/zips/train2017.zip
python data_sampler.py --lmdb train2017.lmdb --input val2017.zip --create
```

2. Train a small real-time super-resolution network using:

```bash
python realtime_superresolution --lmdb_path train2017.lmdb
```

Training is highly unstable and does not often give results as good as the pretrained model.
You can download and play with the pretrained model [here](http://models.tensorpack.com/SuperResolution/).

3. Inference on an image and output in current directory:

```bash
python realtime_superresolution.py --apply --load /path/to/checkpoint --highres set14/monarch.bmp --output monarch
```

