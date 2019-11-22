# [Dual Attention Network for Scene Segmentation(CVPR2019)](https://arxiv.org/pdf/1809.02983.pdf)
[Jun Fu](https://github.com/junfu1115/), [Jing Liu](http://www.nlpr.ia.ac.cn/iva/liujing/index.html), [Haijie Tian](https://github.com/tianhaijie), [Yong Li](http://www.foreverlee.net), Yongjun Bao, Zhiwei Fang,and Hanqing Lu 
## Introduction

We propose a Dual Attention Network (DANet) to adaptively integrate local features with their global dependencies based on the self-attention mechanism. And we achieve new state-of-the-art segmentation performance on three challenging scene segmentation datasets, i.e., Cityscapes, PASCAL Context and COCO Stuff-10k dataset.

![image](img/overview.png)

## Cityscapes testing set result

We train our DANet-101 with only fine annotated data and submit our test results to the official evaluation server.

![image](img/tab3.png)

## Usage

0. Ensure CUDA 10.0 is used.
   ```shell
	$ nvcc -V
   ```
	should output cuda 10.0

1. Create a virtual env with conda using python 3.6.7
   ```shell
	$ conda create -n danet-env python==3.6.7
	$ which pip
   ```
   Ensure the pip is the pip within the virutal environment

2. Install pytorch 1.0.0 and torch vision 
   ```shell
	$ conda install pytorch==1.0.0 torchvision==0.2.1 cuda100 -c pytorch
   ```
  
3. Clone the repository and build:

   ```shell
   git clone https://github.com/junfu1115/DANet.git 
   cd DANet 
   pip install .
   python setup.py install --record files.txt
   ```
   In case the ninja problem still persists. 
   ```shell
   cd DANet
   xargs rm -rf < files.txt
   python setup.py install --record files.txt
   ```
   
4. Dataset

  - Download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset and convert the dataset to [19 categories](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py). 
  - Please put dataset in folder `./datasets`

5. Evaluation

  - Download trained model [DANet101](https://drive.google.com/open?id=1XmpFEF-tbPH0Rmv4eKRxYJngr3pTbj6p) and put it in folder `./danet/cityscapes/model`
  - Evaluation code is in folder `./danet/cityscapes`
  - `cd danet`

  - For single scale testing, please run:
  
   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset cityscapes --model danet --resume-dir cityscapes/model --base-size 2048 --crop-size 768 --workers 1 --backbone resnet101 --multi-grid --multi-dilation 4 8 16 --eval
   ```
   
  - For multi-scale testing, please run:
  
   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset cityscapes --model danet --resume-dir cityscapes/model --base-size 2048 --crop-size 1024 --workers 1 --backbone resnet101 --multi-grid --multi-dilation 4 8 16 --eval --multi-scales
   ```  
   
  - If you want to visualize the result of DAN-101, you can run:
 
   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset cityscapes --model danet --resume-dir cityscapes/model --base-size 2048 --crop-size 768 --workers 1 --backbone resnet101 --multi-grid --multi-dilation 4 8 16
   ```
   
5. Evaluation Result:

   The expected scores will show as follows:
   
   (single scale testing denotes as 'ss' and multiple scale testing denotes as 'ms')
   
   DANet101 on cityscapes val set (mIoU/pAcc): **79.93/95.97** (ss) and **81.49/96.41** (ms)


6. Training:

  - Training code is in folder `./danet/cityscapes`
  - `cd danet`
  
   You can reproduce our result by run:

  ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset cityscapes --model  danet --backbone resnet101 --checkname danet101  --base-size 1024 --crop-size 768 --epochs 240 --batch-size 8 --lr 0.003 --workers 2 --multi-grid --multi-dilation 4 8 16
   ```
 
   Note that: We adopt multiple losses in end of the network for better training. 
   

## Citation
If DANet is useful for your research, please consider citing:
```
@article{fu2018dual,
  title={Dual Attention Network for Scene Segmentation},
  author={Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang,and Hanqing Lu},
  booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```
## Acknowledgement
Thanks [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding), especially the Synchronized BN!
