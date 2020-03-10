
# [Transfer Learning from Synthetic to Real-Noise Denoising with Adaptive Instance Normalization](https://arxiv.org/abs/2002.11244) (Accepted for CVPR 2020)

Yoonsik Kim, Jae Woong Soh, Gu Yong Park, and Nam Ik Cho

[[Arxiv](https://arxiv.org/abs/2002.11244)]

<img src = "/figs/st_jpeg.png" width="400"> <img src = "/figs/st_sr_jpeg.png" width="400"> 

## Environments
- Ubuntu 18.04
- [Tensorflow 1.8](http://www.tensorflow.org/)
- CUDA 9.0 & cuDNN 7.1
- Python 3.6

## Abstract
Real-noise denoising is a challenging task because the statistics of real-noise do not follow the normal distribution, and they are also spatially and temporally changing.
In order to cope with various and complex real-noise, we propose a well-generalized denoising architecture and a transfer learning scheme.
Specifically, we adopt an adaptive instance normalization to build a denoiser, which can regularize the feature map and prevent the network from overfitting to the training set.
We also introduce a transfer learning scheme that transfers knowledge learned from synthetic-noise data to the real-noise denoiser.
From the proposed transfer learning, the synthetic-noise denoiser can learn general features from various synthetic-noise data, and the real-noise denoiser can learn the real-noise characteristics from real data.
From the experiments, we find that the proposed denoising method has great generalization ability, such that our network trained with synthetic-noise achieves the best performance for Darmstadt Noise Dataset (DND) among the methods from published papers.
We can also see that the proposed transfer learning scheme robustly works for real-noise images through the learning with a very small number of labeled data. 


