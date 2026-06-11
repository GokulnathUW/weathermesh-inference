"""Pipeline output validation, status file, and failure alerts."""

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from inference.export import denormalize_forecast
from inference.wm3_env import OUTPUTS_DIR

STATUS_FILE = OUTPUTS_DIR / "pipeline_status.json"
TEMP_2M_K_MIN = 180.0
TEMP_2M_K_MAX = 330.0


def validate_forecast(forecast, output_mesh, cf_path):
    """
    Sanity-check model output before upload.

    Raises RuntimeError if shape, finiteness, 2m temperature, or NetCDF checks fail.
    """
    errors = []
    expected_shape = (len(output_mesh.lats), len(output_mesh.lons), output_mesh.n_vars)

    if isinstance(forecast, torch.Tensor):
        y = forecast.detach().cpu().numpy()
    else:
        y = np.asarray(forecast)

    if tuple(y.shape) != expected_shape:
        errors.append(f"forecast shape {tuple(y.shape)}, expected {expected_shape}")
    if not np.isfinite(y).all():
        n_bad = int((~np.isfinite(y)).sum())
        errors.append(f"forecast has {n_bad} non-finite values")

    physical = denormalize_forecast(y, output_mesh)
    t2m_ch = output_mesh.n_pr + output_mesh.core_sfc_vars.index("167_2t")
    t2m_k = physical[..., t2m_ch]
    t_min, t_max = float(np.nanmin(t2m_k)), float(np.nanmax(t2m_k))
    if t_min < TEMP_2M_K_MIN or t_max > TEMP_2M_K_MAX:
        errors.append(
            f"167_2t out of range: min={t_min:.2f}K max={t_max:.2f}K "
            f"(expected {TEMP_2M_K_MIN}-{TEMP_2M_K_MAX}K)"
        )

    cf_path = Path(cf_path)
    if not cf_path.is_file() or cf_path.stat().st_size == 0:
        errors.append(f"missing or empty NetCDF: {cf_path}")
    else:
        import xarray as xr

        with xr.open_dataset(cf_path) as ds:
            if "init_time" not in ds.attrs:
                errors.append("NetCDF missing init_time attribute")
            if len(ds.data_vars) < 20:
                errors.append(f"NetCDF has {len(ds.data_vars)} variables, expected ~22")

    if errors:
        raise RuntimeError("; ".join(errors))

    return {"t2m_k_min": t_min, "t2m_k_max": t_max}


def write_status(status, **fields):
    """Write latest pipeline status for quick external checks."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    STATUS_FILE.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return STATUS_FILE


def send_failure_alert(title, error, **fields):
    """Log failure; optionally POST to ALERT_WEBHOOK_URL (Slack-compatible JSON)."""
    message = f"{title}: {error}"
    if fields:
        message += "\n" + json.dumps(fields, indent=2, default=str)

    url = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
    if not url:
        return False

    body = json.dumps({"text": message}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
        return True
    except urllib.error.URLError:
        return False
