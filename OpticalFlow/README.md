# OpticalFlow

## FlowNet2

Reproduces
[FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/abs/1612.01925)
by Ilg et al.

Given two images, the network is trained to predict the optical flow between these images.

<p align="center"> <img src="./preview.jpg" width="100%"> </p>

* Top: both input images from Flying Chairs, ground-truth, Caffe output
* Bottom: FlowNet2-C, FlowNet2-S, FlowNet2 results (when converted to TensorFlow)

The authors report the AEE of *2.03* (Caffe Model) on Sintel-clean and our implementation gives an AEE of *2.10*, which is better than other TensorFlow implementations.

| Model|  AEE(sintel clean) |
| ------ | ------ |
| FlowNet-S | 3.82|
| FlowNet-C | 3.08|
| FlowNet2 | 2.10 (authors report 2.03) |

### Usage

1. Download the pre-trained model:

```bash
wget http://files.patwie.com/recipes/models/flownet2-s.npz
wget http://files.patwie.com/recipes/models/flownet2-c.npz
wget http://files.patwie.com/recipes/models/flownet2.npz

```

*Note:* Using these weights, requires to accept the author's license:

```
Pre-trained weights are provided for research purposes only and without any warranty.
Any commercial use of the pre-trained weights requires FlowNet2 authors consent.
```

2. Run inference

```bash
python python flownet2.py --gpu 0 \
        --left left_img.ppm \
        --right right_img.ppm \
        --load flownet2-s.npz --model "flownet2-s"
python python flownet2.py --gpu 0 \
        --left left_img.ppm \
        --right right_img.ppm \
        --load flownet2-c.npz --model "flownet2-c"
python python flownet2.py --gpu 0 \
        --left left_img.ppm \
        --right right_img.ppm \
        --load flownet2.npz --model "flownet2"
```

For `FlowNet2-C` you need to compile the operation in user_ops until this [PR](https://github.com/tensorflow/tensorflow/pull/21392) gets merged:

```bash
cd user_ops
cmake .
make
```

## PWC

Reproduces
[PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume](https://arxiv.org/abs/1709.02371)
by Sun et al.

Given two images, the network is trained to predict the optical flow between these images.

<p align="center"> <img src="./preview-pwc.jpg" width="100%"> </p>

Left to right: Caffe-Version, this re-implementation, ground-truth

1. Download the pre-trained model:

```bash
wget http://files.patwie.com/recipes/models/pwc.npz
```

*Note:* Using these weights, (probably) requires to accept the author's license.

2. Run inference

```bash
python python pwc.py --gpu 0 \
        --left left_img.ppm \
        --right right_img.ppm \
        --load pwc.npz
```