# TensorFlow-Recipes (Tensorpack-Recipes)

Several TensorFlow implementations of recent papers based on the [tensorpack](https://github.com/ppwwyyxx/tensorpack) framework.

Unfortunately, there is a difference between *re-implementing* deep-learning papers, and *re-obtaining* the published performance. The latter usually requires tedious hyper-parameter optimization amongst other things like very long training times. Hence, the following implementations have no guarantees to get the published performance. However you can judge this yourself using our pretrained models.

- **[Learning To See in the Dark](./LearningToSeeInTheDark)** (Chen et al., CVPR 2018) [[pdf]](https://arxiv.org/abs/1805.01934) [[pretrained model]](http://files.patwie.com/recipes/models/seeinthedark.npz)
*Learning to See in the Dark*
    + the toughest part seems the data pre-processing
    + there are some over-exposed pixels in the prediction
- **[ProgressiveGrowingGan](./ProgressiveGrowingGan)** (Karras et al., ICLR 2018) [[pdf]](https://arxiv.org/abs/1710.10196)
*Progressive Growing of GANs for Improved Quality, Stability, and Variation*
    + seems to produce visual good performance on smaller resolutions due to hardware constraints
    + uses no gradient clipping (forgot to activate) and RMSprop
- **[EnhanceNet](./EnhanceNet)** (Sajjadi et al., ICCV 2017) [[pdf]](https://arxiv.org/abs/1612.07919) [[pretrained model]](http://files.patwie.com/recipes/models/enet-pat.npy)
*EnhanceNet: Single Image Super-Resolution Through Automated Texture Synthesis*
    + visually similar performance; seems to produce less artifacts than the author's implementation
- **[FlowNet2-S/C](./OpticalFlow)** (Ilg et al., CVPR 2017) [[pdf]](https://arxiv.org/abs/1612.01925) [[ported caffe model]](http://files.patwie.com/recipes/models/weights.npz)
*FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks*
    + just the inference part of FlowNet2-S and FlowNet2-C
    + please respect the license of the pre-trained weights
- **[LetThereBeColor](./LetThereBeColor)** (Iizuka et al., SIGGRAPH 2016) [[pdf]](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/) [[pretrained model]](http://files.patwie.com/recipes/models/let-there-be-color.npy)
*Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification*
    + slightly worse performance probably due to shorter training time (authors reported 3 weeks; we just trained a few days)
- **[DeepVideoDeblurring](./DeepVideoDeblurring)** (Su et al., CVPR 2017) [[pdf]](https://arxiv.org/abs/1611.08387)
*Deep Video Deblurring*
    + similar performance, when trained on our [dataset](https://github.com/cgtuebingen/learning-blind-motion-deblurring)
- **[SplitBrainAutoEncoder](./SplitBrainAutoEncoder)** (Zhang et al., CVPR 2017) [[pdf]](https://arxiv.org/abs/1611.09842)
*Split-Brain Autoencoders: Unsupervised Learning by Cross-Channel Prediction*
    + not finished yet
- **[PointNet](./PointNet)** (Qi et al., CVPR 2017) [[pdf]](https://arxiv.org/abs/1612.00593) [[pretrained model]](http://files.patwie.com/recipes/models/pointnet.npy)
*PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation*
    + reproduces the accuracy from the paper
    + use dataset provided by the authors
- **[SubPixelSuperResolution](./SubPixelSuperResolution)** (Shi et al., CVPR 216) [[pdf]](https://arxiv.org/abs/1609.05158)
*Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network*
    + not reproduced yet, might be cause by the resizing method (PIL vs OpenCV, vs TensorFlow)
- **[ImageRestorationSymmetricSkip](./ImageRestorationSymmetricSkip)** (Mao et al., NIPS 2016 [[pdf]](https://arxiv.org/abs/1606.08921)
*Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections*
    + slightly worse performance
- **[AlphaGo](./AlphaGo)** (Silver et al, Nature 2016) [[pdf]](https://gogameguru.com/i/2016/03/deepmind-mastering-go.pdf)
    + just the Policy-Network (SL) from AlphaGO
    + validation accuracy is ~51% (paper reports 54%)
- **[DynamicFilterNetwork](./DynamicFilterNetwork)** (Brabandere et al., NIPS 2016) [[pdf]](https://arxiv.org/abs/1605.09673)
*Dynamic Filter Network*
    + reproduces the steering filter example


 I do not judge the papers and methods. Reproducing deep-learning papers with *meaningful* performance is difficult. So there can be some tricks, I missed.
 There is no motivation/time to make them all work perfectly -- *when* possible.
