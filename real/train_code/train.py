"""Train SSLT_93 from scratch for the real CASSI noise and mask domain."""

from __future__ import annotations

import argparse
import os
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset


REAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REAL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from architecture.PSST_Net import RealSSLT93  # noqa: E402
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SSLT_93 from scratch for real CASSI")
    parser.add_argument("--cave-path", required=True, help="Directory containing CAVE .mat cubes")
    parser.add_argument("--kaist-path", required=True, help="Directory containing KAIST .mat cubes")
    parser.add_argument("--mask-path", required=True, help="Real-system mask.mat")
    parser.add_argument("--output-dir", default=str(REAL_DIR / "checkpoints"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=499,
                        help="DSMT runs range(1, 500), i.e. 499 optimization epochs")
    parser.add_argument("--samples-per-epoch", type=int, default=1250,
                        help="DSMT default for a 384 patch: 20000 / (384 / 96)^2")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0,
                        help="Keep 0 on Windows to avoid duplicating large HSI arrays")
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=384, choices=(384,))
    parser.add_argument("--exposure-scale", type=float, default=1.2,
                        help="DSMT real-training exposure multiplier")
    parser.add_argument("--qe", type=float, default=0.4)
    parser.add_argument("--bit-depth", type=int, default=2048)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--cache-cubes", type=int, default=1,
                        help="Number of full HSI cubes cached per data-loader process")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Load and synthesize one CPU sample, then exit before CUDA initialization")
    return parser.parse_args()


def _read_cube(path_string: str) -> np.ndarray:
    path = Path(path_string)
    data = sio.loadmat(path)
    for key in ("data_slice", "HSI", "hsi", "cube"):
        if key in data:
            cube = np.asarray(data[key], dtype=np.float32).squeeze()
            break
    else:
        visible = sorted(key for key in data if not key.startswith("__"))
        raise KeyError(f"{path}: no hyperspectral cube found; variables: {visible}")
    if cube.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D cube, got {cube.shape}")
    if cube.shape[0] == 28 and cube.shape[-1] != 28:
        cube = np.transpose(cube, (1, 2, 0))
    if cube.shape[-1] != 28:
        raise ValueError(f"{path}: expected 28 bands on one axis, got {cube.shape}")
    if float(cube.max()) > 1.5:
        cube = cube / 65535.0
    return np.clip(cube, 0.0, 1.0).astype(np.float32, copy=False)


def _shift_cube(cube: np.ndarray, step: int = 2) -> np.ndarray:
    bands, height, width = cube.shape
    shifted = np.zeros((bands, height, width + (bands - 1) * step), dtype=np.float32)
    for band in range(bands):
        shifted[band, :, step * band:step * band + width] = cube[band]
    return shifted


class RealDomainDataset(Dataset):
    def __init__(self, cave_path: str, kaist_path: str, mask_path: str, samples: int,
                 patch_size: int, exposure_scale: float, qe: float, bit_depth: int,
                 cache_cubes: int):
        self.files = sorted(Path(cave_path).glob("*.mat")) + sorted(Path(kaist_path).glob("*.mat"))
        if not self.files:
            raise FileNotFoundError("No CAVE/KAIST .mat cubes were found")
        mask_data = sio.loadmat(mask_path)
        if "mask" not in mask_data:
            raise KeyError(f"{mask_path} does not contain `mask`")
        self.mask = np.asarray(mask_data["mask"], dtype=np.float32).squeeze()
        self.samples = samples
        self.patch = patch_size
        self.exposure_scale = exposure_scale
        self.qe = qe
        self.bit_depth = bit_depth
        self._load_cube = lru_cache(maxsize=max(0, cache_cubes))(_read_cube)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cube = self._load_cube(str(random.choice(self.files)))
        height, width, _ = cube.shape
        if height < self.patch or width < self.patch:
            raise ValueError(f"Cube {cube.shape} is smaller than patch size {self.patch}")
        y = random.randint(0, height - self.patch)
        x = random.randint(0, width - self.patch)
        target = cube[y:y + self.patch, x:x + self.patch].copy()

        if random.random() < 0.5:
            target = target[:, ::-1].copy()
        if random.random() < 0.5:
            target = target[::-1].copy()
        target = np.rot90(target, random.randint(0, 3)).copy()

        mask_h, mask_w = self.mask.shape
        if mask_h < self.patch or mask_w < self.patch:
            raise ValueError(f"Mask {self.mask.shape} is smaller than patch size {self.patch}")
        my = random.randint(0, mask_h - self.patch)
        mx = random.randint(0, mask_w - self.patch)
        mask_2d = self.mask[my:my + self.patch, mx:mx + self.patch]
        mask_3d = np.repeat(mask_2d[None], 28, axis=0).astype(np.float32)

        target_chw = np.transpose(target, (2, 0, 1)).astype(np.float32)
        coded = _shift_cube(target_chw * mask_3d).sum(axis=0)
        coded = coded / 28.0 * 2.0 * self.exposure_scale
        photon_count = np.maximum(coded * self.bit_depth / self.qe, 0.0).astype(np.int64)
        noisy_coded = np.random.binomial(photon_count, self.qe).astype(np.float32) / self.bit_depth
        phi = _shift_cube(mask_3d)
        return torch.from_numpy(noisy_coded), torch.from_numpy(target_chw), torch.from_numpy(phi)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.batch_size != 1:
        raise ValueError("SSLT_93 real training is restricted to --batch-size 1 for stability")
    if args.workers > 0 and os.name == "nt":
        print("Warning: Windows data workers duplicate process memory; --workers 0 is recommended.")

    dataset = RealDomainDataset(args.cave_path, args.kaist_path, args.mask_path,
                                args.samples_per_epoch, args.patch_size,
                                args.exposure_scale, args.qe, args.bit_depth, args.cache_cubes)
    if args.preflight_only:
        measurement, target, phi = dataset[0]
        print(f"CPU preflight passed: Y={tuple(measurement.shape)}, "
              f"target={tuple(target.shape)}, Phi={tuple(phi.shape)}")
        print("CUDA was not initialized.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Real-domain training requires CUDA")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    print(f"GPU: {torch.cuda.get_device_name(device)}, VRAM: {total_vram_gb:.1f} GB")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    model = RealSSLT93(input_resolution=args.patch_size)
    print("Training Y-interface SSLT_93 from scratch for the real CASSI domain.")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    loss_fn = torch.nn.L1Loss()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for iteration, (measurement, target, phi) in enumerate(loader, start=1):
            measurement = measurement.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            phi = phi.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out1, out2, out3 = model(measurement, phi, return_stages=True)
            loss = 0.1 * loss_fn(out1, target) + 0.3 * loss_fn(out2, target) + loss_fn(out3, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for group in optimizer.param_groups for p in group["params"]), 1.0)
            optimizer.step()
            running += loss.item()
            if iteration % 50 == 0:
                print(f"epoch {epoch:03d} iter {iteration:04d}/{len(loader):04d} loss {running / iteration:.6f}")
        scheduler.step()
        print(f"epoch {epoch:03d} mean loss {running / len(loader):.6f}")
        print(f"peak CUDA memory: {torch.cuda.max_memory_allocated(device) / (1024 ** 3):.2f} GB")
        torch.cuda.reset_peak_memory_stats(device)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({"epoch": epoch, "net": model.state_dict(), "optimizer": optimizer.state_dict()},
                       output_dir / f"sslt93_real_epoch_{epoch:03d}.pth")


if __name__ == "__main__":
    main()
