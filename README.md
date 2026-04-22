# ViT-Hook

<<<<<<< HEAD
=======
Official implementation of the paper **Vision Transformer Hook for Dense Predictions**.

## Overview

ViT-Hook is a Vision Transformer-based framework for dense prediction tasks.
This repository currently provides the codebase for:
- semantic segmentation
- object detection

## Progress

- [x] Public repository released
- [x] Segmentation code released
- [x] Detection code released
- [x] Apache-2.0 license added
- [x] Pretrained checkpoints released
- [ ] Trained model weights released

## Repository Structure

```text
ViT-Hook/
├── mmseg/                # local deployed MMSegmentation package
├── segmentation/         # segmentation training / evaluation code
├── detection/            # detection training / evaluation code
├── LICENSE
└── README.md
```

## Installation

We recommend creating a clean conda environment before installation.

```bash
conda create -n vit-hook python=3.9 -y
conda activate vit-hook

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
pip install mmcv==2.2.0
pip install mmengine==0.10.5
pip install mmdet==3.3.0
pip install timm==0.9.12
pip install numpy==1.24.3
pip install yapf==0.40.2
```

## Compile Deformable Attention

```bash
cd mmseg/models/backbones/vit_hooks/ops
sh make.sh
cd ../../../../..
```

## Released Code

### Segmentation

The semantic segmentation code is available in [`segmentation/`](segmentation).

Please refer to:
- [`segmentation/README.md`](segmentation/README.md)

### Detection

The object detection code is available in [`detection/`](detection).

Please refer to:
- [`detection/README.md`](detection/README.md)

## Notes

- This repository uses the **local `mmseg` package** included in the project.
- We recommend running training and evaluation commands from the repository root.
- If another `mmseg` package is installed in your environment, please make sure the local one is imported first.
- Pretrained checkpoints and trained model weights will be released separately.

>>>>>>> 43f5cae (Update README files for public release)
## Acknowledgements

Many thanks to the following repositories that helped us build this codebase:
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [ViT-Adapter](https://github.com/czczup/ViT-Adapter)

## License

This repository is released under the Apache 2.0 License. Please refer to the [LICENSE](LICENSE) file for details.
