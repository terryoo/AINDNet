
# [Transfer Learning from Synthetic to Real-Noise Denoising with Adaptive Instance Normalization](https://arxiv.org/abs/2002.11244) (Accepted for CVPR 2020)

Yoonsik Kim, Jae Woong Soh, Gu Yong Park, and Nam Ik Cho

[[Arxiv](https://arxiv.org/abs/2002.11244)]

<img src = "/figs/patch_NOISY.png" width="256"> <img src = "/figs/patch_AINDNETTF.png" width="256"> 

## Environments
- Ubuntu 16.04
- [Tensorflow 1.8](http://www.tensorflow.org/)
- CUDA 9.0 & cuDNN 7.1
- Python 3.6

## Test Code

[**Code**](https://github.com/terryoo/AINDNet/xxx)

[**Trained Model**](https://drive.google.com/drive/folders/1kpZNJmgzlZPyBbPiP0Py4kwPGNaQCKWo?usp=sharing)

## Abstract
Real-noise denoising is a challenging task because the statistics of real-noise do not follow the normal distribution, and they are also spatially and temporally changing.
In order to cope with various and complex real-noise, we propose a well-generalized denoising architecture and a transfer learning scheme.
Specifically, we adopt an adaptive instance normalization to build a denoiser, which can regularize the feature map and prevent the network from overfitting to the training set.
We also introduce a transfer learning scheme that transfers knowledge learned from synthetic-noise data to the real-noise denoiser.
From the proposed transfer learning, the synthetic-noise denoiser can learn general features from various synthetic-noise data, and the real-noise denoiser can learn the real-noise characteristics from real data.
From the experiments, we find that the proposed denoising method has great generalization ability, such that our network trained with synthetic-noise achieves the best performance for Darmstadt Noise Dataset (DND) among the methods from published papers.
We can also see that the proposed transfer learning scheme robustly works for real-noise images through the learning with a very small number of labeled data. 

## Brief Description of Proposed Method
### Adaptive Instance Normalization Denoising Network

<p align="center"><img src = "/figs/figure_overview.png" width="800">
  
<br><br>

<img src = "/figs/AIN_Resblock.png" width="320">

We propose a novel well-generalized denoiser based on the AIN, which enables the CNN to work for various noise from many camera devices.	

### Transfer Learning
<img src = "/figs/transfer_learning.png" width="480">
We introduce a transfer learning for the denoising scheme, which learns the domain-invariant information from synthetic noise (SN) data and updates affine transform parameters of AIN for the different-domain data.

## Experimental Results
### Quantitative Results on DND and SIDD benchmarks
<p align="center"><img src="/figs/dnd_table.png" width="400">&nbsp;&nbsp;<img src="/figs/SIDD_table.png" width="400"></p> 

Average PSNR of the denoised images on the DND (left) and SIDD (right) benchmarks, we denote the environment of training, i.e., training with SN data only, RN data only, and both. 

### Visualized Results
<p align="center"><img src = "/figs/visual_comparison.png" width="700">




  


## Citation
```
Will be updated soon.
```

## Acknowledgement
Our work and implementations is inspired by and based on
SPADE [[site](https://github.com/NVlabs/SPADE)].




