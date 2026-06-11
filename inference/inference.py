#!/usr/bin/env python3
"""
Single-pass WeatherMesh-3 inference.

Usage (from repo root):
    python3 -m inference.inference
    python3 -m inference.inference --timestamp 1741305600
"""

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from inference.export import (
    default_forecast_path,
    default_windborne_dir,
    save_forecast_netcdf,
    save_forecast_windborne_api,
)
from inference.wm3_env import OUTPUTS_DIR, PREPROCESSED_DIR, REPO_ROOT, setup_weathermesh, weathermesh_cwd

logger = logging.getLogger(__name__)


def load_native_stack(data_dir, source, timestamp, mesh):
    """
    Load one encoder's sample NPZ files into a single (n_lat, n_lon, C) array.

    Args:
        data_dir (Path): Root with neogfs/ and neohres/ (e.g. data/sample or data/preprocessed)
        source (str): "neogfs" or "neohres"
        timestamp (int): Unix UTC time used in sample filenames
        mesh (LatLonGrid): Matching encoder mesh from model.config.inputs

    Returns:
        np.ndarray: Stacked pressure + surface channels, float32
    """
    data_dir = Path(data_dir).resolve()
    month = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y%m")

    base_path = data_dir / source / "f000" / month / f"{timestamp}.npz"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing sample base file: {base_path}")

    with np.load(base_path) as z:
        pr = z["pr"]
        sfc = z["sfc"]

    extras = []
    for var in mesh.extra_sfc_vars:
        path = data_dir / source / "extra" / var / month / f"{timestamp}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing sample extra file for {var}: {path}")
        with np.load(path) as z:
            extras.append(z["x"][..., np.newaxis])

    n_lon = len(mesh.lons)
    pr_flat = pr.reshape(pr.shape[0], n_lon, -1)
    sfc_all = np.concatenate([sfc] + extras, axis=-1)
    stacked = np.concatenate([pr_flat, sfc_all], axis=-1).astype(np.float32)

    # Sample files include the -90 south pole row; mesh.lats stops at -89.75
    return stacked[: len(mesh.lats), :, :]


def to_model_tensor(native_stack, mesh, device):
    """Interpolate pressure levels to 28 and pad output-only surface channels."""
    from utils import interp_levels

    x = torch.from_numpy(native_stack).unsqueeze(0).to(device)
    x = interp_levels(x, mesh, mesh.input_levels, mesh.levels)

    # zeropad channels are forecast outputs only, not in analysis input
    pad = torch.zeros(
        (1, len(mesh.lats), len(mesh.lons), mesh.extra_sfc_pad),
        dtype=x.dtype,
        device=device,
    )
    return torch.cat([x, pad], dim=-1)


def load_sample_input(data_dir, timestamp, meshes, device=None):
    """
    Load WindBorne sample data and assemble both encoder inputs.

    Args:
        data_dir (Path): Root with neogfs/ and neohres/ (e.g. data/sample or data/preprocessed)
        timestamp (int): Unix UTC initialization time
        meshes (list): model.config.inputs — [gfs_mesh, hres_mesh]
        device (torch.device, optional): Defaults to CPU

    Returns:
        dict: {"gfs", "hres", "t0"} ready for run_inference()
    """
    if device is None:
        device = torch.device("cpu")

    gfs_mesh, hres_mesh = meshes[0], meshes[1]

    try:
        gfs_native = load_native_stack(data_dir, "neogfs", timestamp, gfs_mesh)
        hres_native = load_native_stack(data_dir, "neohres", timestamp, hres_mesh)

        gfs = to_model_tensor(gfs_native, gfs_mesh, device)
        hres = to_model_tensor(hres_native, hres_mesh, device)
    except Exception as e:
        raise RuntimeError(f"Failed to load sample input for t={timestamp}: {e}") from e

    assert gfs.shape[-1] == gfs_mesh.n_vars and hres.shape[-1] == hres_mesh.n_vars
    t0 = torch.tensor([timestamp], dtype=torch.long, device=device)
    return {"gfs": gfs, "hres": hres, "t0": t0}


def run_inference(model, data, forecast_hours=6, device=None):
    """
    Run one WeatherMesh-3 forward pass.

    Args:
        model: Loaded ForecastModel from get_WeatherMesh3()
        data (dict): {"gfs", "hres", "t0"} from load_sample_input()
        forecast_hours (int): Forecast lead time (default 6)
        device (torch.device, optional): Inferred from data if omitted

    Returns:
        dict: {"forecast", "forecast_hours", "inference_seconds", "latent_l2"}
    """
    if device is None:
        device = data["gfs"].device

    gfs = data["gfs"].to(device)
    hres = data["hres"].to(device)
    t0 = data["t0"].to(device)
    model = model.to(device).eval()

    try:
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model([gfs, hres, t0], todo=[forecast_hours], send_to_cpu=False)
        elapsed = time.perf_counter() - start
    except Exception as e:
        raise RuntimeError(f"Inference failed for {forecast_hours}h forecast: {e}") from e

    forecast = outputs[forecast_hours][0]
    if forecast.ndim == 4 and forecast.shape[0] == 1:
        forecast = forecast.squeeze(0)
    if torch.isnan(forecast).any():
        raise RuntimeError(f"Forecast contains NaN values at {forecast_hours}h")

    logger.info(
        "Inference complete: %dh shape=%s in %.2fs",
        forecast_hours,
        tuple(forecast.shape),
        elapsed,
    )
    return {
        "forecast": forecast,
        "forecast_hours": forecast_hours,
        "inference_seconds": elapsed,
        "latent_l2": float(outputs["latent_l2"]),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run WeatherMesh-3 inference on sample data")
    parser.add_argument("--data-dir", type=Path, default=PREPROCESSED_DIR)
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "weights" / "WeatherMesh3.pt")
    parser.add_argument("--timestamp", type=int, default=1781179200)
    parser.add_argument("--forecast-hours", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument(
        "--format",
        choices=("cf", "windborne", "both"),
        default="both",
        help="NetCDF layout: cf (single multi-var file), windborne (API one-var-per-file), both",
    )
    parser.add_argument("--output", type=Path, default=None, help="CF NetCDF path (default: output-dir/forecast_<ts>_f<lead>.nc)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_weathermesh()

    try:
        with weathermesh_cwd():
            from model import get_WeatherMesh3

            model = get_WeatherMesh3(str(args.weights)).to(device).eval()
            output_mesh = model.config.outputs[0]
            data = load_sample_input(args.data_dir, args.timestamp, model.config.inputs, device=device)
    except Exception as e:
        raise RuntimeError(f"Failed to prepare model or input: {e}") from e

    result = run_inference(model, data, forecast_hours=args.forecast_hours, device=device)

    y = result["forecast"]
    finite = y[torch.isfinite(y)]
    print(
        f"Forecast {result['forecast_hours']}h: shape={tuple(y.shape)}, "
        f"min={finite.min().item():.4f}, max={finite.max().item():.4f}, "
        f"mean={finite.mean().item():.4f}, time={result['inference_seconds']:.2f}s"
    )

    out_path = args.output or default_forecast_path(
        args.output_dir, args.timestamp, args.forecast_hours
    )
    if args.format in ("cf", "both"):
        saved = save_forecast_netcdf(
            y, output_mesh, args.timestamp, args.forecast_hours, out_path
        )
        print(f"Saved CF NetCDF: {saved.resolve()}")
    if args.format in ("windborne", "both"):
        api_paths = save_forecast_windborne_api(
            y, output_mesh, args.timestamp, args.forecast_hours, args.output_dir
        )
        api_dir = default_windborne_dir(args.output_dir, args.timestamp, args.forecast_hours)
        print(f"Saved {len(api_paths)} WindBorne API NetCDF files under {api_dir.resolve()}")


if __name__ == "__main__":
    main()
