# Datasets

Please download the CASSI datasets separately and arrange them as follows:

```text
datasets/
├── cave_1024_28/
│   ├── scene1.mat
│   └── ...
├── CAVE_512_28/
│   ├── scene1.mat
│   └── ...
├── KAIST_CVPR2021/
│   ├── 1.mat
│   └── ...
├── TSA_simu_data/
│   ├── mask.mat
│   └── Truth/
│       ├── scene01.mat
│       └── ...
└── TSA_real_data/
    ├── mask.mat
    └── Measurements/
        ├── scene1.mat
        └── ...
```

The datasets are not redistributed in this repository. The directory names
follow the public [MST toolbox](https://github.com/caiyuanhao1998/MST).

Expected MATLAB variables:

- simulation training cubes: `img_expand` or `img`;
- simulation ground truth: `img`;
- real-training cubes: `data_slice`, `HSI`, `hsi`, or `cube`;
- real coded measurements: `meas_real` by default;
- calibrated masks: `mask`.
