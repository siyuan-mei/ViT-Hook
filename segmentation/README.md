# Segmentation

This folder contains the semantic segmentation code for **ViT-Hook**.

Our implementation is built on top of the **local `mmseg` package** included in this repository. Please make sure the local package is used instead of another externally installed `mmseg` package.

## Overview

We provide the segmentation codebase of **ViT-Hook**, including training and evaluation pipelines based on a local MMSegmentation-style framework.

Currently supported:
- Semantic segmentation training
- Semantic segmentation evaluation
- Multiple ViT-based backbone initializations

## Data Preparation

Please prepare the datasets according to the official MMSegmentation guidelines.

Supported datasets:
- ADE20K
- Cityscapes
- COCO-Stuff
- Pascal Context

Reference:
- https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets

## Pretraining Sources

We support multiple publicly available pretrained backbones.

| Name | Year | Type | Pretraining Data | Repo | Paper |
|------|------|------|------------------|------|-------|
| DeiT | 2021 | Supervised | ImageNet-1K | [repo](https://github.com/facebookresearch/deit) | [paper](https://arxiv.org/abs/2012.12877) |
| AugReg | 2021 | Supervised | ImageNet-21K / ImageNet-22K style pretraining and augmentation recipes | [repo](https://github.com/huggingface/pytorch-image-models) | [paper](https://arxiv.org/abs/2106.10270) |
| BEiTv2 | 2022 | Masked Image Modeling | ImageNet-22K | [repo](https://github.com/microsoft/unilm/tree/master/beit2) | [paper](https://arxiv.org/abs/2208.06366) |
| DINOv2 | 2023 | Self-Supervised | Large-scale curated image dataset | [repo](https://github.com/facebookresearch/dinov2) | [paper](https://arxiv.org/abs/2304.07193) |
| AM-RADIO | 2024 | Foundation Model / Distillation | Multi-source teacher supervision | [repo](https://github.com/NVlabs/RADIO) | [paper](https://arxiv.org/abs/2312.06709) |

## Training

We recommend running the code from the repository root so that the local `mmseg` package is correctly imported.

### Example: training with Slurm

```bash
sbatch train.sh --config configs/ade20k/vithook/vithook_l_augreg-upernet-b4x4-160k-ade20k-512x512.py
```

### Example: training with Python

```bash
python -m segmentation.train configs/ade20k/vithook/vithook_l_augreg-upernet-b4x4-160k-ade20k-512x512.py
```

### Example: specify work directory

```bash
python -m segmentation.train configs/ade20k/vithook/vithook_l_augreg-upernet-b4x4-160k-ade20k-512x512.py --work-dir work_dirs/vithook_ade20k
```

## Notes

- This codebase uses the **local `mmseg` package** in this repository.
- We recommend launching training and evaluation from the repository root.
- If you also have another `mmseg` package installed in your environment, make sure the local one is imported first.
- The deformable attention operators need to be compiled before training.
