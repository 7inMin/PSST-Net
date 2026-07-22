"""Test Y-interface SSLT_93 on real CASSI measurements."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


REAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REAL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from architecture.PSST_Net import RealSSLT93  # noqa: E402


MEASUREMENT_KEYS = ("meas_real", "meas", "measurement", "Y", "y")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Y-interface SSLT_93 on real CASSI measurements")
    parser.add_argument("--data-path", required=True, help="A .mat measurement or directory of .mat files")
    parser.add_argument("--mask-path", required=True, help="Real-system mask .mat containing `mask`")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint produced by real/train.py")
    parser.add_argument("--output-dir", default=str(REAL_DIR / "results"))
    parser.add_argument("--gpu", default="0", help="CUDA device index; use `cpu` for diagnostics")
    parser.add_argument("--patch-size", type=int, default=384, choices=(384,))
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--measurement-key", default=None)
    parser.add_argument("--mask-key", default="mask")
    parser.add_argument("--normalization", choices=("max", "none"), default="max")
    parser.add_argument("--measurement-peak", type=float, default=0.8,
                        help="Match DSMT real/test: meas / meas.max() * 0.8")
    parser.add_argument("--clip", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _read_2d(path: Path, preferred_key: str | None, fallback_keys: tuple[str, ...]) -> np.ndarray:
    data = sio.loadmat(path)
    keys = (preferred_key,) if preferred_key else fallback_keys
    for key in keys:
        if key and key in data:
            value = np.asarray(data[key], dtype=np.float32).squeeze()
            if value.ndim != 2:
                raise ValueError(f"{path}: `{key}` must be 2-D, got {value.shape}")
            return value
    visible = sorted(key for key in data if not key.startswith("__"))
    raise KeyError(f"{path}: none of {keys} found; available variables: {visible}")


def _normalize_measurement(meas: np.ndarray, mode: str, peak: float) -> np.ndarray:
    meas = np.maximum(meas.astype(np.float32, copy=False), 0.0)
    if mode == "none":
        return meas
    maximum = float(meas.max())
    if maximum <= 0:
        raise ValueError("Measurement maximum is zero; cannot normalize it")
    return meas / maximum * peak


def _shift_mask(mask: np.ndarray, bands: int = 28, step: int = 2) -> np.ndarray:
    height, width = mask.shape
    phi = np.zeros((bands, height, width + (bands - 1) * step), dtype=np.float32)
    for band in range(bands):
        phi[band, :, step * band:step * band + width] = mask
    return phi


def _tile_starts(length: int, patch: int, overlap: int) -> list[int]:
    if length < patch:
        raise ValueError(f"Reconstruction side {length} is smaller than patch size {patch}")
    stride = patch - overlap
    if stride <= 0:
        raise ValueError("Overlap must be smaller than patch size")
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def _blend_window(size: int) -> np.ndarray:
    one_d = np.hanning(size + 2)[1:-1].astype(np.float32)
    return np.maximum(np.outer(one_d, one_d), 1e-3)[None]


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, torch.nn.Module):
        state = checkpoint.state_dict()
    elif isinstance(checkpoint, dict) and "net" in checkpoint:
        state = checkpoint["net"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        state = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format")
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_model(path: Path, device: torch.device, input_resolution: int) -> RealSSLT93:
    model = RealSSLT93(input_resolution=input_resolution)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    return model.to(device).eval()


def reconstruct(model: RealSSLT93, y_full: np.ndarray, phi_full: np.ndarray,
                device: torch.device, patch: int, overlap: int) -> np.ndarray:
    height, coded_width = y_full.shape
    width = coded_width - 54
    if phi_full.shape != (28, height, coded_width):
        raise ValueError(f"Phi shape {phi_full.shape} does not match Y shape {y_full.shape}")
    ys = _tile_starts(height, patch, overlap)
    xs = _tile_starts(width, patch, overlap)
    blend = _blend_window(patch)
    result = np.zeros((28, height, width), dtype=np.float32)
    weights = np.zeros((1, height, width), dtype=np.float32)
    total = len(ys) * len(xs)
    number = 0

    with torch.inference_mode():
        for top in ys:
            for left in xs:
                number += 1
                print(f"\rReconstructing tile {number}/{total}", end="", flush=True)
                coded_right = left + patch + 54
                y_tile = torch.from_numpy(y_full[top:top + patch, left:coded_right]).unsqueeze(0).to(device)
                phi_tile = torch.from_numpy(phi_full[:, top:top + patch, left:coded_right]).unsqueeze(0).to(device)
                pred = model(y_tile, phi_tile).squeeze(0).float().cpu().numpy()
                result[:, top:top + patch, left:left + patch] += pred * blend
                weights[:, top:top + patch, left:left + patch] += blend
    print()
    return result / np.maximum(weights, 1e-8)


def _measurement_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files found in {path}")
    return files


def main() -> None:
    args = parse_args()
    if args.gpu.lower() == "cpu":
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device = torch.device(f"cuda:{args.gpu}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(Path(args.checkpoint), device, args.patch_size)

    for path in _measurement_files(Path(args.data_path)):
        print(f"Processing {path}")
        y_full = _read_2d(path, args.measurement_key, MEASUREMENT_KEYS)
        y_full = _normalize_measurement(y_full, args.normalization, args.measurement_peak)
        height, coded_width = y_full.shape
        width = coded_width - 54
        mask = _read_2d(Path(args.mask_path), args.mask_key, ("mask", "Mask"))
        if mask.shape[0] < height or mask.shape[1] < width:
            raise ValueError(f"Mask {mask.shape} is smaller than reconstruction field {(height, width)}")
        phi_full = _shift_mask(mask[:height, :width])
        reconstruction = reconstruct(model, y_full, phi_full, device, args.patch_size, args.overlap)
        if args.clip:
            reconstruction = np.clip(reconstruction, 0.0, 1.0)
        res = np.transpose(reconstruction, (1, 2, 0)).astype(np.float32)
        save_path = output_dir / f"{path.stem}_SSLT93_real.mat"
        sio.savemat(save_path, {"res": res}, do_compression=True)
        print(f"Saved {save_path} with shape {res.shape}")


if __name__ == "__main__":
    main()
