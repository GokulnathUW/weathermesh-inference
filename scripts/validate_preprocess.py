#!/usr/bin/env python3
"""
Validate preprocessed GFS encoder data against the WindBorne sample, then run inference.

Usage (from repo root):
    python3 scripts/validate_preprocess.py \\
        --sample-timestamp 1741305600 --processed-timestamp 1781179200
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inference.export import denormalize_forecast, reorder_field_lon180
from inference.inference import load_sample_input, run_inference
from inference.preprocess import WM3_CORE_SFC_VARS, WM3_EXTRA_SFC_VARS, WM3_PRESSURE_VARS
from inference.wm3_env import (
    PREPROCESSED_DIR,
    REPO_ROOT as INFERENCE_ROOT,
    SAMPLE_DATA_DIR,
    setup_weathermesh,
    weathermesh_cwd,
)

N_LAT, N_LON = 721, 1440
N_GFS_LEVELS = 25


def _month(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m")


def load_gfs_bundle(root, timestamp):
    """Load neogfs base NPZ + extra surface files for one init time."""
    root = Path(root)
    month = _month(timestamp)
    base_path = root / "neogfs" / "f000" / month / f"{timestamp}.npz"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing GFS base file: {base_path}")

    with np.load(base_path) as z:
        bundle = {"pr": z["pr"], "sfc": z["sfc"]}

    for var in WM3_EXTRA_SFC_VARS:
        path = root / "neogfs" / "extra" / var / month / f"{timestamp}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing GFS extra file for {var}: {path}")
        with np.load(path) as z:
            bundle[var] = z["x"]

    return bundle


def _stats(arr):
    x = np.asarray(arr, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }


def _print_stats_row(label, sample_stats, ours_stats):
    print(f"  {label:<28}  sample: min={sample_stats['min']:9.4f} max={sample_stats['max']:9.4f} "
          f"mean={sample_stats['mean']:9.4f} std={sample_stats['std']:9.4f}")
    print(f"  {'':28}  ours  : min={ours_stats['min']:9.4f} max={ours_stats['max']:9.4f} "
          f"mean={ours_stats['mean']:9.4f} std={ours_stats['std']:9.4f}")


def check_shapes(sample, ours):
    errors = []
    expected = {
        "pr": (N_LAT, N_LON, len(WM3_PRESSURE_VARS), N_GFS_LEVELS),
        "sfc": (N_LAT, N_LON, len(WM3_CORE_SFC_VARS)),
    }
    for key, shape in expected.items():
        if sample[key].shape != shape:
            errors.append(f"sample {key} shape {sample[key].shape}, expected {shape}")
        if ours[key].shape != shape:
            errors.append(f"processed {key} shape {ours[key].shape}, expected {shape}")
        if sample[key].shape != ours[key].shape:
            errors.append(f"shape mismatch on {key}: sample {sample[key].shape} vs processed {ours[key].shape}")

    for var in WM3_EXTRA_SFC_VARS:
        exp = (N_LAT, N_LON)
        if sample[var].shape != exp:
            errors.append(f"sample extra {var} shape {sample[var].shape}, expected {exp}")
        if ours[var].shape != exp:
            errors.append(f"processed extra {var} shape {ours[var].shape}, expected {exp}")
        if sample[var].shape != ours[var].shape:
            errors.append(f"shape mismatch on extra {var}: sample vs processed")

    return errors


def check_finite(bundle, label):
    errors = []
    for key in ("pr", "sfc"):
        arr = bundle[key].astype(np.float32)
        if not np.isfinite(arr).all():
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            errors.append(f"{label} {key}: {n_nan} NaNs, {n_inf} Infs")
    for var in WM3_EXTRA_SFC_VARS:
        arr = bundle[var].astype(np.float32)
        if not np.isfinite(arr).all():
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            errors.append(f"{label} extra {var}: {n_nan} NaNs, {n_inf} Infs")
    return errors


def print_side_by_side_stats(sample, ours):
    _print_stats_row("pr (all)", _stats(sample["pr"]), _stats(ours["pr"]))
    for i, var in enumerate(WM3_PRESSURE_VARS):
        s = sample["pr"][:, :, i, :]
        o = ours["pr"][:, :, i, :]
        _print_stats_row(f"pr {var}", _stats(s), _stats(o))
    for i, var in enumerate(WM3_CORE_SFC_VARS):
        _print_stats_row(f"sfc {var}", _stats(sample["sfc"][:, :, i]), _stats(ours["sfc"][:, :, i]))
    for var in WM3_EXTRA_SFC_VARS:
        _print_stats_row(f"extra {var}", _stats(sample[var]), _stats(ours[var]))


def output_temp_channel(output_mesh):
    return output_mesh.n_pr + output_mesh.core_sfc_vars.index("167_2t")


def denorm_2t(normalized, mean, std):
    return normalized * std + mean


def run_inference_check(processed_dir, timestamp, weights, forecast_hours, device):
    setup_weathermesh()
    with weathermesh_cwd():
        from model import get_WeatherMesh3

        model = get_WeatherMesh3(str(weights)).to(device).eval()
        output_mesh = model.config.outputs[0]
        expected_shape = (len(output_mesh.lats), len(output_mesh.lons), output_mesh.n_vars)

        data = load_sample_input(processed_dir, timestamp, model.config.inputs, device=device)
        result = run_inference(model, data, forecast_hours=forecast_hours, device=device)
        forecast = result["forecast"]
        if forecast.ndim == 4 and forecast.shape[0] == 1:
            forecast = forecast.squeeze(0)

    errors = []
    if tuple(forecast.shape) != expected_shape:
        errors.append(f"forecast shape {tuple(forecast.shape)}, expected {expected_shape}")
    if torch.isnan(forecast).any():
        errors.append(f"forecast contains {int(torch.isnan(forecast).sum().item())} NaNs")
    if torch.isinf(forecast).any():
        errors.append(f"forecast contains {int(torch.isinf(forecast).sum().item())} Infs")

    return {
        "forecast": forecast,
        "output_mesh": output_mesh,
        "expected_shape": expected_shape,
        "inference_seconds": result["inference_seconds"],
        "errors": errors,
    }


def save_temp_map(forecast, output_mesh, out_path):
    temp_k = denormalize_forecast(forecast, output_mesh)[..., output_temp_channel(output_mesh)]
    temp_c = temp_k - 273.15

    lats = output_mesh.lats
    lons = output_mesh.lons
    temp_c, lats, lons = reorder_field_lon180(temp_c, lons, lats)
    lon2d, lat2d = np.meshgrid(lons, lats)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    pcm = ax.pcolormesh(lon2d, lat2d, temp_c, shading="auto", cmap="RdYlBu_r", vmin=-40, vmax=40)
    ax.set_title("2m temperature (denormalized forecast, °C)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(lats.min(), lats.max())
    fig.colorbar(pcm, ax=ax, label="°C")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path, temp_c


def main():
    parser = argparse.ArgumentParser(description="Validate preprocessed GFS data and run inference")
    parser.add_argument("--sample-dir", type=Path, default=SAMPLE_DATA_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PREPROCESSED_DIR)
    parser.add_argument("--sample-timestamp", type=int, default=1741305600)
    parser.add_argument("--processed-timestamp", type=int, required=True)
    parser.add_argument("--weights", type=Path, default=INFERENCE_ROOT / "weights" / "WeatherMesh3.pt")
    parser.add_argument("--forecast-hours", type=int, default=6)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "validation_temp.png")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ok = True
    print("=" * 70)
    print("PREPROCESS + INFERENCE VALIDATION")
    print("=" * 70)
    print(f"Sample dir     : {args.sample_dir.resolve()}  t={args.sample_timestamp}")
    print(f"Processed dir  : {args.processed_dir.resolve()}  t={args.processed_timestamp}")
    print(f"Device         : {device}")
    print()

    try:
        sample = load_gfs_bundle(args.sample_dir, args.sample_timestamp)
        ours = load_gfs_bundle(args.processed_dir, args.processed_timestamp)
    except FileNotFoundError as e:
        print(f"LOAD: FAIL\n  {e}")
        sys.exit(1)

    print("SHAPES")
    shape_errors = check_shapes(sample, ours)
    if shape_errors:
        ok = False
        print("  FAIL")
        for err in shape_errors:
            print(f"  {err}")
    else:
        print("  OK (sample and processed neogfs layouts match)")

    print("\nFINITE VALUES")
    finite_errors = check_finite(sample, "sample") + check_finite(ours, "processed")
    if finite_errors:
        ok = False
        print("  FAIL")
        for err in finite_errors:
            print(f"  {err}")
    else:
        print("  OK (no NaNs or Infs in sample or processed GFS)")

    print("\nSTATS (sample vs processed — different init times are expected to differ)")
    print_side_by_side_stats(sample, ours)

    if not args.weights.exists():
        print(f"\nINFERENCE: SKIP (missing weights: {args.weights})")
        sys.exit(1 if not ok else 0)

    print(f"\nINFERENCE on processed t={args.processed_timestamp} ({args.forecast_hours}h)")
    try:
        inf = run_inference_check(
            args.processed_dir,
            args.processed_timestamp,
            args.weights,
            args.forecast_hours,
            device,
        )
    except Exception as e:
        ok = False
        print(f"  FAIL: {e}")
        sys.exit(1)

    if inf["errors"]:
        ok = False
        print("  FAIL")
        for err in inf["errors"]:
            print(f"  {err}")
    else:
        print(f"  OK shape={inf['expected_shape']} in {inf['inference_seconds']:.2f}s")

    out_path, temp_c = save_temp_map(inf["forecast"], inf["output_mesh"], args.output)
    print(f"\nPLOT saved: {out_path.resolve()}")
    print(f"  denorm 2m temp (°C): min={temp_c.min():.2f} max={temp_c.max():.2f} mean={temp_c.mean():.2f}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
