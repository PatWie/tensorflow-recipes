# Tensorpack-Recipes

Several implementations of recent papers based on [tensorpack](https://github.com/ppwwyyxx/tensorpack). We provide pretrained models using our implementation. Unfortunately, sometimes the success of *re-implementing* deep-learning papers with published performance requires tedious hyper-parameter optimization amongst other things like very long training times. Hence, these implementations have not guarantees to get the published performance, however you can judge this yourself using our pretrained models

| Paper  | PDF | Status | model | notes |
| ------ | ------ | ----- | ----- | ----- |
| [DynamicFilterNetwork](DynamicFilterNetwork) | [pdf](https://arxiv.org/abs/1605.09673) | reproduce steering filter example  |  | 
| [PointNet](PointNet) | [pdf](https://arxiv.org/abs/1612.00593) | similar performance  | [pretrained model](http://files.patwie.com/recipes/models/pointnet.npy) | see author's implementation for creating the datasets | 
| [EnhanceNet](EnhanceNet) | [pdf](https://arxiv.org/abs/1612.07919) | similar and better performance  | [pretrained model](http://files.patwie.com/recipes/models/enet-pat.npy) | 
| [Let There Be Color](LetThereBeColor) | [pdf](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/) | slightly worse performance | [pretrained model](http://files.patwie.com/recipes/models/let-there-be-color.npy) | we just trained for a few days |
| [RealTime Superresolution](SubPixelSuperResolution) | [pdf](https://arxiv.org/abs/1609.05158) | not reproduceable yet | | might be caused by the used resizing method
| [Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections](ImageRestorationSymmetricSkip) | [pdf](https://arxiv.org/abs/1606.08921) | slightly worse performance | |
| [Deep Video Deblurring](DeepVideoDeblurring) | [pdf](https://arxiv.org/abs/1611.08387) | similar performance | | when trained on our [dataset](https://github.com/cgtuebingen/learning-blind-motion-deblurring) |
| [Split-Brain Autoencoders: Unsupervised Learning by Cross-Channel Prediction](SplitBrainAutoEncoder) | [pdf](https://arxiv.org/abs/1611.09842) | not finished | |
| [Policy-Network (SL) from AlphaGO](AlphaGO) | [pdf](https://gogameguru.com/i/2016/03/deepmind-mastering-go.pdf) | approx. 51% accuracy on validation set | |