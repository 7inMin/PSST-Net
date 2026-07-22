# Real experiment

This directory follows the `train_code`/`test_code` organization of the MST
toolbox. Both folders contain the same real-domain PSST-Net architecture.

```text
real/
├── train_code/
│   ├── architecture/PSST_Net.py
│   └── train.py
└── test_code/
    ├── architecture/PSST_Net.py
    └── test.py
```

The real training pipeline consumes CAVE and KAIST cubes together with the
calibrated real mask. It synthesizes noisy coded measurements and optimizes
all three PSST-Net stages end to end. Real testing consumes the measured
snapshot directly and saves each reconstructed cube as a MATLAB file.

Use `datasets/TSA_real_data/mask.mat` for this experiment. A simulation mask
or simulation checkpoint is not interchangeable with its real-domain
counterpart.

Before training, `train_code/check_datasets.py` can validate the CAVE and
KAIST directory layout, MATLAB variable names, cube dimensions, and optional
full-file loading without importing the model or initializing CUDA.
