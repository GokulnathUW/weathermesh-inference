#!/usr/bin/env python3
"""
End-to-end WeatherMesh-3 pipeline: fetch GFS analysis, preprocess, infer, upload.

Usage (from repo root):
    python3 pipeline.py

Requires AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_S3_BUCKET in the environment
(or in .env). Optional: AWS_DEFAULT_REGION (defaults to us-east-2).
"""

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import boto3
import torch
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from inference.export import (
    default_forecast_path,
    default_windborne_dir,
    save_forecast_netcdf,
    save_forecast_windborne_api,
)
from inference.inference import load_sample_input, run_inference
from inference.monitor import send_failure_alert, validate_forecast, write_status
from inference.preprocess import (
    fetch_gfs_analysis,
    preprocess_gfs_encoder,
    preprocess_hres_encoder,
)
from inference.wm3_env import (
    OUTPUTS_DIR,
    PREPROCESSED_DIR,
    RAW_GFS_DIR,
    setup_weathermesh,
    weathermesh_cwd,
)

logger = logging.getLogger("weathermesh.pipeline")

WEIGHTS = REPO_ROOT / "weights" / "WeatherMesh3.pt"
DEVICE = torch.device("cuda")
FORECAST_HOURS = 6
LOG_FILE = OUTPUTS_DIR / "pipeline.log"


def setup_logging():
    """Log structured JSON to pipeline.log; mirror to stdout only when interactive."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(message)s")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    file_handler = logging.FileHandler(LOG_FILE, mode="a")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if sys.stdout.isatty():
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        root.addHandler(stream)

    for name in ("botocore", "boto3", "urllib3", "natten"):
        logging.getLogger(name).setLevel(logging.WARNING)


def log_event(event, **fields):
    payload = {
        "event": event,
        "ts": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    logger.info(json.dumps(payload, default=str))


def download_gfs():
    """Download latest GFS 0.25° f000 analysis from NOAA AWS."""
    info = fetch_gfs_analysis(RAW_GFS_DIR)
    return info


def preprocess_gfs(grib_path):
    """Extract, normalize, and save neogfs + neohres encoder NPZ files."""
    grib_path = Path(grib_path)
    gfs = preprocess_gfs_encoder(grib_path, output_dir=PREPROCESSED_DIR)
    hres = preprocess_hres_encoder(grib_path, output_dir=PREPROCESSED_DIR)
    timestamp = gfs["timestamp"]
    if hres["timestamp"] != timestamp:
        raise RuntimeError(
            f"Encoder timestamp mismatch: neogfs={timestamp} neohres={hres['timestamp']}"
        )
    return timestamp


def infer_forecast(timestamp):
    """Run WeatherMesh-3 on preprocessed data and write NetCDF output."""
    setup_weathermesh()
    with weathermesh_cwd():
        from model import get_WeatherMesh3

        model = get_WeatherMesh3(str(WEIGHTS)).to(DEVICE).eval()
        output_mesh = model.config.outputs[0]
        data = load_sample_input(PREPROCESSED_DIR, timestamp, model.config.inputs, device=DEVICE)

    result = run_inference(model, data, forecast_hours=FORECAST_HOURS, device=DEVICE)
    forecast = result["forecast"]

    cf_path = save_forecast_netcdf(
        forecast,
        output_mesh,
        timestamp,
        FORECAST_HOURS,
        default_forecast_path(OUTPUTS_DIR, timestamp, FORECAST_HOURS),
    )
    api_paths = save_forecast_windborne_api(
        forecast, output_mesh, timestamp, FORECAST_HOURS, OUTPUTS_DIR
    )
    return {
        "forecast_hours": FORECAST_HOURS,
        "inference_seconds": result["inference_seconds"],
        "forecast": forecast,
        "output_mesh": output_mesh,
        "cf_netcdf": cf_path,
        "windborne_dir": default_windborne_dir(OUTPUTS_DIR, timestamp, FORECAST_HOURS),
        "windborne_count": len(api_paths),
    }


def upload_netcdf_to_s3(local_path, timestamp, forecast_hours):
    """Upload the CF forecast NetCDF to S3 using credentials from the environment."""
    bucket = os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        raise RuntimeError("AWS_S3_BUCKET environment variable is not set")

    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
    key = f"forecasts/forecast_{timestamp}_f{int(forecast_hours):03d}.nc"
    client = boto3.client("s3", region_name=region)
    client.upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"


def _ts_iso(timestamp):
    return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat()


def run_cycle():
    """Run one fetch → preprocess → infer → upload cycle."""
    cycle_start = datetime.now(timezone.utc)
    log_event("cycle_start", cycle_start=cycle_start.isoformat())

    try:
        log_event("gfs_fetch_start")
        fetch_info = download_gfs()
        log_event(
            "gfs_fetch_ok",
            status="ok",
            path=str(fetch_info["path"]),
            date=fetch_info["date"],
            hour=fetch_info["hour"],
            s3_key=fetch_info["s3_key"],
        )

        log_event("preprocess_start", grib_path=str(fetch_info["path"]))
        timestamp = preprocess_gfs(fetch_info["path"])
        log_event(
            "preprocess_ok",
            status="ok",
            timestamp=timestamp,
            init_time=_ts_iso(timestamp),
        )

        log_event("inference_start", timestamp=timestamp, forecast_hours=FORECAST_HOURS)
        out = infer_forecast(timestamp)
        validation = validate_forecast(out["forecast"], out["output_mesh"], out["cf_netcdf"])
        log_event(
            "validation_ok",
            status="ok",
            t2m_k_min=round(validation["t2m_k_min"], 2),
            t2m_k_max=round(validation["t2m_k_max"], 2),
        )
        log_event(
            "inference_ok",
            status="ok",
            duration_seconds=round(out["inference_seconds"], 3),
            forecast_hours=out["forecast_hours"],
            cf_netcdf=str(out["cf_netcdf"]),
            windborne_dir=str(out["windborne_dir"]),
            windborne_count=out["windborne_count"],
        )

        log_event("s3_upload_start", local_path=str(out["cf_netcdf"]))
        s3_uri = upload_netcdf_to_s3(out["cf_netcdf"], timestamp, out["forecast_hours"])
        log_event(
            "s3_upload_ok",
            status="ok",
            uri=s3_uri,
            local_path=str(out["cf_netcdf"]),
        )

        elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        log_event(
            "cycle_complete",
            status="ok",
            timestamp=timestamp,
            init_time=_ts_iso(timestamp),
            inference_seconds=round(out["inference_seconds"], 3),
            s3_uri=s3_uri,
            elapsed_seconds=round(elapsed, 3),
        )
        write_status(
            "ok",
            timestamp=timestamp,
            init_time=_ts_iso(timestamp),
            inference_seconds=round(out["inference_seconds"], 3),
            elapsed_seconds=round(elapsed, 3),
            s3_uri=s3_uri,
            cf_netcdf=str(out["cf_netcdf"]),
            t2m_k_min=round(validation["t2m_k_min"], 2),
            t2m_k_max=round(validation["t2m_k_max"], 2),
        )
        return True

    except Exception as exc:
        elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        log_event(
            "cycle_failed",
            status="error",
            error_type=type(exc).__name__,
            error=str(exc),
            elapsed_seconds=round(elapsed, 3),
            traceback=traceback.format_exc(),
        )
        write_status(
            "error",
            error_type=type(exc).__name__,
            error=str(exc),
            elapsed_seconds=round(elapsed, 3),
        )
        alerted = send_failure_alert(
            "WeatherMesh pipeline cycle failed",
            str(exc),
            error_type=type(exc).__name__,
            elapsed_seconds=round(elapsed, 3),
        )
        log_event("alert_sent" if alerted else "alert_skipped", webhook=alerted)
        return False


def main():
    load_dotenv(REPO_ROOT / ".env")
    setup_logging()
    run_cycle()


if __name__ == "__main__":
    main()
