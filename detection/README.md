# Detection (COCO)

This folder contains the COCO detection code for **ViT-Hook**.

Our detection code is developed on top of [MMDetection v3.3](https://github.com/open-mmlab/mmdetection).

## Overview

We provide the detection codebase of **ViT-Hook**, including training and evaluation pipelines for COCO object detection and instance segmentation.

## Data Preparation

Please prepare the **COCO 2017** dataset according to the official MMDetection data preparation guidelines.

Expected datasets:
- COCO 2017 train
- COCO 2017 val

Reference:
- https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html

A typical directory structure is:

```text
data/
└── coco/
    ├── annotations/
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    ├── train2017/
    └── val2017/
```

## Pretraining Sources

We support multiple publicly available pretrained backbones.

| Name | Year | Type | Pretraining Data | Repo | Paper |
|------|------|------|------------------|------|-------|
| DeiT | 2021 | Supervised | ImageNet-1K | [repo](https://github.com/facebookresearch/deit) | [paper](https://arxiv.org/abs/2012.12877) |
| AugReg | 2021 | Supervised | ImageNet-21K / ImageNet-22K style pretraining and augmentation recipes | [repo](https://github.com/huggingface/pytorch-image-models) | [paper](https://arxiv.org/abs/2106.10270) |
| BEiTv2 | 2022 | Masked Image Modeling | ImageNet-22K | [repo](https://github.com/microsoft/unilm/tree/master/beit2) | [paper](https://arxiv.org/abs/2208.06366) |
| DINOv2 | 2023 | Self-Supervised | Large-scale curated image dataset | [repo](https://github.com/facebookresearch/dinov2) | [paper](https://arxiv.org/abs/2304.07193) |
| AM-RADIO | 2024 | Foundation Model / Distillation | Multi-source teacher supervision | [repo](https://github.com/NVlabs/RADIO) | [paper](https://arxiv.org/abs/2312.06709) |

## Configurations

Example COCO configs are provided in:

```text
detection/configs/coco/vithook/
```

These configs include different backbone initializations and detector settings, such as:
- ViT-Hook with DeiT initialization
- ViT-Hook with AugReg initialization
- BEiT-Hook
- DINO-Hook
- RADIO-Hook

## Training

We recommend running the code from the repository root.

### Example: training with Python

```bash
python -m detection.train --config detection/configs/coco/vithook/vithook_s_deit-mask_rcnn_fpn-1x-coco.py
```

### Example: specify work directory

```bash
python -m detection.train --config detection/configs/coco/vithook/vithook_s_deit-mask_rcnn_fpn-1x-coco.py --work-dir work_dirs/vithook_coco
```

### Example: resume training automatically

```bash
python -m detection.train --config detection/configs/coco/vithook/vithook_s_deit-mask_rcnn_fpn-1x-coco.py --work-dir work_dirs/vithook_coco --resume
```

