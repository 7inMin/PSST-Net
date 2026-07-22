# Simulation experiment

This directory follows the `train_code`/`test_code` organization of the MST
toolbox.

```text
simulation/
├── train_code/
│   ├── architecture/PSST_Net.py
│   ├── option.py
│   ├── train.py
│   ├── utils.py
│   └── ssim_torch.py
└── test_code/
    ├── architecture/PSST_Net.py
    ├── test.py
    ├── utils.py
    └── ssim_torch.py
```

The released architecture is the final three-stage PSST-Net implementation.
Training retains the original `0.1/0.3/1.0` deep-supervision weights, RMSE
loss, measurement formation, mask input, optimizer, and scheduler.
