# PSST-Net

Official implementation of **PSST-Net: Progressive Spatial--Spectral
Refinement for Hyperspectral Image Reconstruction**.

PSST-Net is an end-to-end CASSI reconstruction network organized as three
progressive spatial--spectral refinement stages. Each stage reuses the
measurement-derived initialization and calibrated mask, performs multi-scale
spatial restoration, and applies stage-terminal local spectral refinement.

## Repository structure

The release follows the experiment organization of the
[MST toolbox](https://github.com/caiyuanhao1998/MST), while containing only
the components required by PSST-Net.

```text
PSST-Net/
├── datasets/
│   └── README.md
├── real/
│   ├── train_code/
│   └── test_code/
├── simulation/
│   ├── train_code/
│   └── test_code/
├── README.md
└── requirements.txt
```

## Environment

- Python 3.9 or later
- PyTorch with CUDA support
- packages listed in `requirements.txt`

## Dataset preparation

Download the datasets separately and follow the directory tree documented in
[`datasets/README.md`](datasets/README.md). Large data files, checkpoints,
and reconstruction results are intentionally excluded from Git.

## Experiments

- Simulation training entry: `simulation/train_code/train.py`
- Simulation testing entry: `simulation/test_code/test.py`
- Real-domain training entry: `real/train_code/train.py`
- Real-measurement testing entry: `real/test_code/test.py`

Detailed launch scripts and pretrained-model links will be added separately.

## Acknowledgement

The repository layout and CASSI dataset organization follow the public MST
toolbox. We thank its authors and the broader spectral compressive imaging
community for their open-source contributions.
