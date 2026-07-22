"""Check whether CAVE and KAIST MAT/NPY cubes work with real training.

This script is intentionally independent of PyTorch and the PSST-Net model. By
default it only reads MAT/NPY metadata, so a large hyperspectral cube is not
loaded into RAM. Use ``--load-sample`` only when a full read test is required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio


SUPPORTED_KEYS = ("data_slice", "HSI", "hsi", "cube")
SUPPORTED_SUFFIXES = {".mat", ".npy"}
DEFAULT_CAVE_PATH = Path(r"G:\Adata2\CAVE_512_28")
DEFAULT_KAIST_PATH = Path(r"G:\Adata2\KAIST_CVPR2021")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate CAVE/KAIST data for PSST-Net real-domain training"
    )
    parser.add_argument("--cave-path", type=Path, default=DEFAULT_CAVE_PATH)
    parser.add_argument("--kaist-path", type=Path, default=DEFAULT_KAIST_PATH)
    parser.add_argument(
        "--patch-size",
        type=int,
        default=384,
        help="Minimum spatial size required by real/train_code/train.py",
    )
    parser.add_argument(
        "--load-sample",
        action="store_true",
        help="Fully load the first valid cube from each dataset and verify values",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path for a JSON compatibility report",
    )
    return parser.parse_args()


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def select_cube_variable(path: Path) -> tuple[str, tuple[int, ...], str]:
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return "<array>", tuple(int(value) for value in array.shape), str(array.dtype)
    variables = sio.whosmat(path)
    for key in SUPPORTED_KEYS:
        for name, shape, matlab_type in variables:
            if name == key:
                return name, tuple(int(value) for value in shape), matlab_type
    visible = [name for name, _, _ in variables]
    raise KeyError(
        f"none of {SUPPORTED_KEYS} found; available variables: {visible}"
    )


def validate_shape(shape: tuple[int, ...], patch_size: int) -> tuple[bool, str]:
    if len(shape) != 3:
        return False, f"expected a 3-D cube, got {shape}"
    band_axes = [axis for axis, length in enumerate(shape) if length == 28]
    if not band_axes:
        return False, f"no 28-band axis found in {shape}"
    band_axis = band_axes[-1]
    spatial = [length for axis, length in enumerate(shape) if axis != band_axis]
    if min(spatial) < patch_size:
        return False, f"spatial size {tuple(spatial)} is smaller than {patch_size}"
    return True, f"28 bands on axis {band_axis}; spatial size {tuple(spatial)}"


def load_sample(path: Path, key: str) -> dict[str, object]:
    if path.suffix.lower() == ".npy":
        cube = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False)).squeeze()
    else:
        cube = np.asarray(sio.loadmat(path)[key]).squeeze()
    finite = np.isfinite(cube)
    return {
        "loaded_shape": list(cube.shape),
        "dtype": str(cube.dtype),
        "finite": bool(finite.all()),
        "minimum": float(np.nanmin(cube)),
        "maximum": float(np.nanmax(cube)),
    }


def inspect_dataset(
    name: str, path: Path, patch_size: int, should_load_sample: bool
) -> dict[str, object]:
    result: dict[str, object] = {
        "name": name,
        "path": str(path.resolve()),
        "compatible": True,
        "errors": [],
        "warnings": [],
        "files": [],
    }
    errors: list[str] = result["errors"]  # type: ignore[assignment]
    warnings: list[str] = result["warnings"]  # type: ignore[assignment]
    records: list[dict[str, object]] = result["files"]  # type: ignore[assignment]

    if not path.is_dir():
        errors.append("directory does not exist")
        result["compatible"] = False
        return result

    recursive_files = sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in SUPPORTED_SUFFIXES
    )
    direct_files = [file for file in recursive_files if file.parent == path]
    nested_files = [file for file in recursive_files if file.parent != path]

    result["direct_cube_files"] = len(direct_files)
    result["recursive_cube_files"] = len(recursive_files)
    result["total_size_bytes"] = sum(file.stat().st_size for file in recursive_files)

    if not direct_files:
        errors.append("no root-level .mat or .npy hyperspectral cubes")
        if nested_files:
            warnings.append(
                f"found {len(nested_files)} nested .mat/.npy file(s); pass their parent directory instead"
            )
        result["compatible"] = False
        return result

    for file in direct_files:
        record: dict[str, object] = {
            "file": file.name,
            "size_bytes": file.stat().st_size,
        }
        try:
            key, shape, matlab_type = select_cube_variable(file)
            valid, detail = validate_shape(shape, patch_size)
            record.update(
                {
                    "key": key,
                    "shape": list(shape),
                    "matlab_type": matlab_type,
                    "valid": valid,
                    "detail": detail,
                }
            )
            if not valid:
                errors.append(f"{file.name}: {detail}")
            elif should_load_sample and not any("sample_load" in item for item in records):
                record["sample_load"] = load_sample(file, key)
        except Exception as error:  # report every incompatible file in one run
            record.update({"valid": False, "error": str(error)})
            errors.append(f"{file.name}: {error}")
        records.append(record)

    result["compatible"] = not errors
    return result


def print_result(result: dict[str, object]) -> None:
    status = "PASS" if result["compatible"] else "FAIL"
    print(f"\n[{status}] {result['name']}: {result['path']}")
    if "direct_cube_files" in result:
        print(
            f"  root-level MAT/NPY files: {result['direct_cube_files']}; "
            f"recursive MAT/NPY files: {result['recursive_cube_files']}; "
            f"total size: {human_size(int(result['total_size_bytes']))}"
        )
    for record in result["files"]:  # type: ignore[union-attr]
        if record.get("valid"):
            print(
                f"  OK {record['file']}: key={record['key']}, "
                f"shape={tuple(record['shape'])}, type={record['matlab_type']}"
            )
        else:
            print(f"  BAD {record['file']}: {record.get('error', record.get('detail'))}")
    for warning in result["warnings"]:  # type: ignore[union-attr]
        print(f"  WARNING: {warning}")
    for error in result["errors"]:  # type: ignore[union-attr]
        print(f"  ERROR: {error}")


def main() -> int:
    args = parse_args()
    results = [
        inspect_dataset("CAVE", args.cave_path, args.patch_size, args.load_sample),
        inspect_dataset("KAIST", args.kaist_path, args.patch_size, args.load_sample),
    ]
    for result in results:
        print_result(result)

    compatible = all(bool(result["compatible"]) for result in results)
    report = {"compatible": compatible, "datasets": results}
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report saved to {args.report}")

    print("\nOVERALL:", "PASS" if compatible else "FAIL")
    return 0 if compatible else 1


if __name__ == "__main__":
    sys.exit(main())
