# S2-FPN: Scale-ware Strip Attention Guided Feature Pyramid Network for Real-time Semantic Segmentation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/s-textsuperscript-2-fpn-scale-ware-strip/real-time-semantic-segmentation-on-camvid)](https://paperswithcode.com/sota/real-time-semantic-segmentation-on-camvid?p=s-textsuperscript-2-fpn-scale-ware-strip) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/s-textsuperscript-2-fpn-scale-ware-strip/real-time-semantic-segmentation-on-cityscapes-3)](https://paperswithcode.com/sota/real-time-semantic-segmentation-on-cityscapes-3?p=s-textsuperscript-2-fpn-scale-ware-strip) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/s-textsuperscript-2-fpn-scale-ware-strip/semantic-segmentation-on-cityscapes-2)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-2?p=s-textsuperscript-2-fpn-scale-ware-strip)

This project contains the Pytorch implementation for the proposed S<sup>2</sup>-FPN: [[Arxiv]](http://arxiv.org/abs/2206.07298).

### Introduction

Modern high-performance semantic segmentation methods employ a heavy backbone and dilated convolution to extract the relevant feature. Although extracting features with both contextual and semantic information is critical for the segmentation tasks, it brings a memory footprint and high computation cost for real-time applications. This paper presents a new model to achieve a trade-off between accuracy/speed for real-time road scene semantic segmentation. Specifially, we proposed a lightweight model named Scale-aware Strip Attention Guided Feature Pyramid Network (S<sup>2</sup>-FPN). Our network consists of three main modules: Attention Pyramid Fusion (APF) module, Scale-aware Strip Attention Module (SSAM), and Global Feature Upsample (GFU) module. APF adopts an attention mechanisms to learn discriminative multi-scale features and help close the semantic gap between different levels. APF uses the scale-aware attention to encode global context with vertical stripping operation and models the long-range dependencies, which helps relate pixels with similar semantic label. In addition, APF employs channel-wise reweighting block (CRB) to emphasize the channel features. Finally, the decoder of S<sup>2</sup>-FPN then adopts GFU, which is used to fuse features from APF and the encoder. Extensive experiments have been conducted on two challenging semantic segmentation benchmarks, which demonstrate that our approach achieves better accuracy/speed trade-off with different model settings. The proposed models have achieved a results of 76.2%mIoU/87.3FPS, 77.4%mIoU/67FPS, and 77.8%mIoU/30.5FPS on Cityscapes dataset, and 69.6%mIoU,71.0% mIoU,and 74.2% mIoU on Camvid dataset.
                The detailed architecture of S<sup>2</sup>-FPN

<p align="center"><img width="90%" src="./demo_images/scale_aware_network.jpg" /></p>

                Attention Pyramid Fusion Module

<p align="center"><img width="75%" src="./demo_images/spstrip_attention_fusion.jpg" /></p>

### Updates

- S2FPN works with [ResNet18,ResNet34,ResNet50,ResNet101,ResNet152]. We tested it with ResNet18 and 34.
- upload pretrained weights

- 18/10/2023 we updated the scale-aware attention block and updated with new model. Now you can use the new models as S2FPN and the old one is changed to S2FPNv1
- Our paper was marked as state of the art in [Papers with Code](https://paperswithcode.com/task/real-time-semantic-segmentation).

### Installation

1. Pyroch Environment

- Env: Python 3.6; PyTorch 1.0; CUDA 10.1; cuDNN V8
- Install some packages

```
pip install opencv-python pillow numpy matplotlib
```

1. Clone this repository

```
git clone https://github.com/mohamedac29/S2-FPN
cd S2-FPN
```

3. Dataset

You need to download the [Cityscapes](https://www.cityscapes-dataset.com/), and put the files in the `dataset` folder with following structure.

```
├── cityscapes_test_list.txt
├── cityscapes_train_list.txt
├── cityscapes_trainval_list.txt
├── cityscapes_val_list.txt
├── cityscapes_val.txt
├── gtCoarse
│   ├── train
│   ├── train_extra
│   └── val
├── gtFine
│   ├── test
│   ├── train
│   └── val
├── leftImg8bit
│   ├── test
│   ├── train
│   └── val
```

- Convert the dataset to [19 categories](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py).

- Download the [Camvid](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid) dataset and put the files in the following structure.

```
├── camvid_test_list.txt
├── camvid_train_list.txt
├── camvid_trainval_list.txt
├── camvid_val_list.txt
├── test
├── testannot
├── train
├── trainannot
├── val
└── valannot


```
### Pretrained Weights for the old version S2FPN

You can download the pretrained weights. There are some differences in the 
accuracy listed here
- Camvid and Cityscapes Datasets. FPS computed based on GTX1080Ti

|Model |     Dataset      |  Pretrained  | Train type |  test (mIoU)    |  FPS  |                                                                    model                                                                     |
| :--------------:| :--------------: | :----------: | :--------: | :--------: | :---: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
|S2FPN18| Cityscapes | ResNet18 |  train  | **76.7%** | 84.2 | [ckpt](https://drive.google.com/file/d/1WDiguApZiHeelUzZwoJhaL52tZMHlq8F/view?usp=share_link)|
|S2FPN34| Cityscapes | ResNet34 |  train  | **77.4%**  | 64.4 | [ckpt](https://drive.google.com/file/d/1j4seqf67HW7_OKIPTGLuC-oTehwppIHB/view?usp=share_link) |
|S2FPN34M| Cityscapes | ResNet34M|  train  | **78.2%**  | 28.5 |  |
|S2FPN18| CamVid     | ResNet18 |  trainval  | **70.4%** | 122.2 | [ckpt](https://drive.google.com/file/d/1-rQCfzlwENo_KRDwyaWPLMqoox-e-5LL/view?usp=share_link)         |
|S2FPN34| CamVid     | ResNet34 |  trainval  | **71.2%** | 104.2 |  [ckpt](https://drive.google.com/file/d/19Sp1PQLU4AuLOi64Fx4hMEr7k_1Ekefe/view?usp=share_link)   |
|S2FPN34M| CamVid     | ResNet34M |  trainval  | **74.5%**  |53.4  | [ckpt](https://drive.google.com/file/d/13H-foMl4utJqOXXHd8dZxWonk8a3hiTl/view?usp=share_link)       |



### Pretrained Weights for the old version S2FPNv1

You can download the pretrained weights. There are some differences in the 
accuracy listed here
- Camvid and Cityscapes Datasets. FPS computed based on GTX1080Ti

|     Dataset      |  Pretrained  | Train type |    mIoU    |  FPS  |                                                                    model                                                                     |
| :--------------: | :----------: | :--------: | :--------: | :---: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| Cityscapes | ResNet18v1 |  train  | **76.3%** | 87.3 | [ckpt](https://drive.google.com/file/d/1Q_wFh9W9SmSOR1f4PVY7LD_Rztl0IIFB/view?usp=share_link)|
| Cityscapes | ResNet34 |    |  |  |  |
| Cityscapes | ResNet34M|    |  |  |  |
| CamVid     | ResNet18 |  trainval  | **70.1%** | 124.2 | [ckpt](https://drive.google.com/file/d/1H1iTzYaP8CbuDeeW0phnvCTBigHe8CD8/view?usp=share_link)         |
| CamVid     | ResNet34 |  trainval  | **71.0%** | 107.2 |     |
| CamVid     | ResNet34M |  trainval  | **74.76%**  |55.5  | [ckpt](https://drive.google.com/file/d/1XI8jNAm1C5anW9ExJvneddhYAaIPdOHa/view?usp=share_link)       |




### Training 

- Training on Camvid datsaset
```
python train.py --dataset camvid --model SSFPN --max_epochs 150 --train_type trainval --lr 3e-4 --batch_size 8
```
- Training on Camvid datsaset - train_type [trainval,trainval]
```
python train.py --dataset cityscapes --model SSFPN --max_epochs 500 --train_type trainval --lr 3e-4 --batch_size 8
```
### Testing 
- Testing on Camvid datsaset
```
python test.py --dataset camvid --model SSFPN --checkpoint ./checkpoint/camvid/SSFPNbs8gpu1_trainval/model_150.pth --gpus 0
```
- Testing on Cityscapes datsaset
```
python test.py --dataset cityscapes --model SSFPN --checkpoint ./checkpoint/cityscapes/SSFPNbs8gpu1_trainval/model_500.pth --gpus 0
```
### Inference Speed
- Inference speed with input resolution 512x1024
```
python eval_fps.py 512,1024
```

### Citation

If you find this work useful in your research, please consider citing.

```
@article{elhassan2022s,
  title={S2-FPN: Scale-ware Strip Attention Guided Feature Pyramid Network for Real-time Semantic Segmentation},
  author={Elhassan, Mohammed AM and Yang, Chenhui and Huang, Chenxi and Legesse Munea, Tewodros and Hong, Xin},
  journal={arXiv e-prints},
  pages={arXiv--2206},
  year={2022}
}
```
