## OpticalFlow - FlowNet2

Reproduces
[FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/abs/1612.01925)
by Ilg et al.

Given two images, the network is trained to predict the optical flow between these images.



### Usage

1. Download the pre-trained model:

```bash
wget http://files.patwie.com/recipes/models/flownet2-s.npz

```

*Note:* Using these weights, requires to accept the author's license:

```
Pre-trained weights are provided for research purposes only and without any warranty.
Any commercial use of the pre-trained weights requires FlowNet2 authors consent.
```

2. Run inference

```bash
python python flownet2s.py --gpu 0 \
        --left left_img.ppm \
        --right right_img.ppm \
        --apply --load flownet2-s.npz
```

