"""Denormalize WeatherMesh-3 forecasts and write NetCDF output."""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from inference.wm3_env import setup_weathermesh, weathermesh_cwd

logger = logging.getLogger(__name__)

# ECMWF-style physical units for WM-3 output variables
WM3_UNITS = {
    "129_z": "m2 s-2",
    "130_t": "K",
    "131_u": "m s-1",
    "132_v": "m s-1",
    "133_q": "kg kg-1",
    "165_10u": "m s-1",
    "166_10v": "m s-1",
    "167_2t": "K",
    "151_msl": "Pa",
    "45_tcc": "1",
    "168_2d": "K",
    "246_100u": "m s-1",
    "247_100v": "m s-1",
    "15_msnswrf": "W m-2",
    "142_lsp": "kg m-2",
    "143_cp": "kg m-2",
    "201_mx2t": "K",
    "202_mn2t": "K",
    "142_lsp-6h": "kg m-2",
    "143_cp-6h": "kg m-2",
    "201_mx2t-6h": "K",
    "202_mn2t-6h": "K",
}

# WindBorne gridded-forecast API variable names (see api.windbornesystems.com)
WM3_TO_API = {
    "129_z": "geopotential",
    "130_t": "temperature",
    "131_u": "wind_u",
    "132_v": "wind_v",
    "133_q": "specific_humidity",
    "165_10u": "wind_u_10m",
    "166_10v": "wind_v_10m",
    "167_2t": "temperature_2m",
    "151_msl": "pressure_msl",
    "45_tcc": "total_cloud_cover",
    "168_2d": "dewpoint_2m",
    "246_100u": "wind_u_100m",
    "247_100v": "wind_v_100m",
    "15_msnswrf": "short_wave_radiation",
    "142_lsp": "total_precipitation_1h",
    "201_mx2t": "maximum_temperature_2m_3h",
    "202_mn2t": "minimum_temperature_2m_3h",
    "142_lsp-6h": "total_precipitation_6h",
}

API_UNITS = {
    "geopotential": "m2/s2",
    "temperature": "K",
    "wind_u": "m/s",
    "wind_v": "m/s",
    "specific_humidity": "kg/kg",
    "wind_u_10m": "m/s",
    "wind_v_10m": "m/s",
    "temperature_2m": "K",
    "pressure_msl": "hPa",
    "total_cloud_cover": "%",
    "dewpoint_2m": "K",
    "wind_u_100m": "m/s",
    "wind_v_100m": "m/s",
    "short_wave_radiation": "W/m2",
    "total_precipitation_1h": "mm",
    "total_precipitation_6h": "mm",
    "maximum_temperature_2m_3h": "K",
    "minimum_temperature_2m_3h": "K",
}


def _as_numpy(forecast):
    if isinstance(forecast, torch.Tensor):
        forecast = forecast.detach().cpu().numpy()
    if forecast.ndim == 4 and forecast.shape[0] == 1:
        forecast = forecast.squeeze(0)
    return forecast.astype(np.float32)


def lon180_sort_indices(lons):
    """
    Column indices to sort WM-3 longitude order into monotonic [-180, 180).

    WM-3 stores lons as [0, …, 179.75, -180, …, -0.25], not west-to-east.
    """
    lon = np.asarray(lons, dtype=np.float64)
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    return np.argsort(lon180), lon180


def reorder_field_lon180(field, lons, lats=None):
    """Reorder a (lat, lon) or (lat, lon, ...) field to monotonic -180..180 longitude."""
    xi, lon180 = lon180_sort_indices(lons)
    field = np.asarray(field)
    reordered = np.take(field, xi, axis=1)
    lon_out = lon180[xi].astype(np.float32)
    if lats is None:
        return reordered, lon_out
    return reordered, np.asarray(lats, dtype=np.float32), lon_out


def denormalize_forecast(normalized, output_mesh):
    """Apply z-score inverse: physical = normalized * std + mean."""
    setup_weathermesh()
    with weathermesh_cwd():
        from utils import load_normalization

        _, std, mean = load_normalization(output_mesh, with_means=True)

    x = _as_numpy(normalized)
    mean = mean.reshape(1, 1, -1)
    std = std.reshape(1, 1, -1)
    return x * std + mean


def _to_api_grid(field, lats, lons):
    """Reorder to WindBorne convention: lon in [-180, 180], lat increasing."""
    lat = np.asarray(lats, dtype=np.float64)
    z, lon180 = reorder_field_lon180(field, lons)

    if lat[0] > lat[-1]:
        lat = lat[::-1].astype(np.float32)
        z = z[::-1, :]
    else:
        lat = lat.astype(np.float32)

    return z, lat, lon180


def _to_api_units(wm3_var, values):
    if wm3_var == "151_msl":
        return values / 100.0
    if wm3_var == "45_tcc":
        return values * 100.0
    return values


def _api_init_iso(init_timestamp):
    return (
        datetime.fromtimestamp(int(init_timestamp), tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _write_api_field_netcdf(path, api_name, field, lats, lons, wm3_var, init_timestamp, forecast_hours):
    import xarray as xr

    data, lat, lon = _to_api_grid(_to_api_units(wm3_var, field), lats, lons)
    units = API_UNITS[api_name]

    ds = xr.Dataset(
        {
            "latitude": (["lat"], lat),
            "longitude": (["lon"], lon),
            api_name: (["lat", "lon"], data),
        },
        attrs={
            "initialization_time": _api_init_iso(init_timestamp),
            "forecast_hour": int(forecast_hours),
            "description": f"Gridded forecast data for {api_name}",
            "source": "WeatherMesh-3 inference",
        },
    )
    ds[api_name].attrs["units"] = units
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def default_windborne_dir(output_dir, init_timestamp, forecast_hours):
    return Path(output_dir) / f"windborne_{init_timestamp}_f{int(forecast_hours):03d}"


def save_forecast_windborne_api(forecast, output_mesh, init_timestamp, forecast_hours, output_dir):
    """
    Write one NetCDF per API variable (2D surface fields, 2D upper-level slices).

    Matches WindBorne gridded-forecast layout: lat/lon dims, latitude/longitude
    data vars, initialization_time + forecast_hour global attrs.
    """
    physical = denormalize_forecast(forecast, output_mesh)
    lats = output_mesh.lats
    lons = output_mesh.lons
    levels = output_mesh.levels
    n_pr_vars = len(output_mesh.pressure_vars)
    n_levels = len(levels)
    n_pr = n_pr_vars * n_levels

    pr_block = physical[..., :n_pr].reshape(len(lats), len(lons), n_pr_vars, n_levels)

    out_dir = default_windborne_dir(output_dir, init_timestamp, forecast_hours)
    paths = []

    for vi, wm3_var in enumerate(output_mesh.pressure_vars):
        api_name = WM3_TO_API.get(wm3_var)
        if api_name is None:
            continue
        for li, level in enumerate(levels):
            fname = f"{api_name}_{int(level)}hPa.nc"
            path = _write_api_field_netcdf(
                out_dir / fname,
                api_name,
                pr_block[:, :, vi, li],
                lats,
                lons,
                wm3_var,
                init_timestamp,
                forecast_hours,
            )
            paths.append(path)

    for si, wm3_var in enumerate(output_mesh.sfc_vars):
        if wm3_var == "zeropad":
            continue
        api_name = WM3_TO_API.get(wm3_var)
        if api_name is None:
            logger.debug("Skipping %s (no WindBorne API name)", wm3_var)
            continue
        path = _write_api_field_netcdf(
            out_dir / f"{api_name}.nc",
            api_name,
            physical[..., n_pr + si],
            lats,
            lons,
            wm3_var,
            init_timestamp,
            forecast_hours,
        )
        paths.append(path)

    logger.info("Wrote %d WindBorne-format NetCDF files under %s", len(paths), out_dir.resolve())
    return paths


def forecast_to_dataset(physical, output_mesh, init_timestamp, forecast_hours):
    """Build an xarray Dataset with lat/lon/level coords and CF metadata."""
    import xarray as xr

    physical = _as_numpy(physical)
    lats = output_mesh.lats
    lons = output_mesh.lons
    levels = output_mesh.levels
    n_pr_vars = len(output_mesh.pressure_vars)
    n_levels = len(levels)
    n_pr = n_pr_vars * n_levels

    pr_block = physical[..., :n_pr].reshape(len(lats), len(lons), n_pr_vars, n_levels)
    pr_block = np.transpose(pr_block, (2, 3, 0, 1))

    _, lats_out, lons_out = reorder_field_lon180(physical[..., 0], lons, lats)
    xi, _ = lon180_sort_indices(lons)

    init_time = datetime.fromtimestamp(int(init_timestamp), tz=timezone.utc)
    valid_time = init_time + timedelta(hours=int(forecast_hours))

    data_vars = {}
    for vi, var in enumerate(output_mesh.pressure_vars):
        pr_sorted = pr_block[vi][:, :, xi]
        data_vars[var] = xr.DataArray(
            pr_sorted,
            dims=("level", "latitude", "longitude"),
            coords={"level": levels, "latitude": lats_out, "longitude": lons_out},
            attrs={"units": WM3_UNITS.get(var, "unknown"), "long_name": var},
        )

    for si, var in enumerate(output_mesh.sfc_vars):
        if var == "zeropad":
            continue
        sfc_sorted, _, _ = reorder_field_lon180(physical[..., n_pr + si], lons, lats)
        data_vars[var] = xr.DataArray(
            sfc_sorted,
            dims=("latitude", "longitude"),
            coords={"latitude": lats_out, "longitude": lons_out},
            attrs={"units": WM3_UNITS.get(var, "unknown"), "long_name": var},
        )

    return xr.Dataset(
        data_vars,
        attrs={
            "title": "WeatherMesh-3 forecast",
            "init_time": init_time.isoformat().replace("+00:00", "Z"),
            "valid_time": valid_time.isoformat().replace("+00:00", "Z"),
            "forecast_lead_time_hours": int(forecast_hours),
            "source": "WeatherMesh-3",
            "Conventions": "CF-1.8",
        },
    )


def default_forecast_path(output_dir, init_timestamp, forecast_hours):
    return Path(output_dir) / f"forecast_{init_timestamp}_f{int(forecast_hours):03d}.nc"


def save_forecast_netcdf(forecast, output_mesh, init_timestamp, forecast_hours, path):
    """Denormalize a model forecast tensor and write NetCDF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    physical = denormalize_forecast(forecast, output_mesh)
    ds = forecast_to_dataset(physical, output_mesh, init_timestamp, forecast_hours)
    ds.to_netcdf(path)
    logger.info("Wrote forecast NetCDF: %s", path.resolve())
    return path
