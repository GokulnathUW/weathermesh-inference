import logging
from pathlib import Path

from inference.wm3_env import PREPROCESSED_DIR, RAW_GFS_DIR, setup_weathermesh, weathermesh_cwd

logger = logging.getLogger(__name__)
GFS_BUCKET        = "noaa-gfs-bdp-pds"
GFS_ANALYSIS_FILE = "pgrb2.0p25.f000"

WM3_PRESSURE_VARS = ["129_z", "130_t", "131_u", "132_v", "133_q"]
WM3_CORE_SFC_VARS = ["165_10u", "166_10v", "167_2t", "151_msl"]
WM3_EXTRA_SFC_VARS = ["45_tcc", "168_2d", "246_100u", "247_100v"]
WM3_SFC_VARS = WM3_CORE_SFC_VARS + WM3_EXTRA_SFC_VARS
WM3_INPUT_VARS = WM3_PRESSURE_VARS + WM3_SFC_VARS


def wm3_longitude_axis(n_lon=1440, resolution=0.25):
    """WM-3 column order: 0..179.75°E, then -180..-0.25°W (not monotonic west-to-east)."""
    import numpy as np

    lons = np.arange(0, 359.99, resolution)
    lons[lons >= 180] -= 360
    return lons[:n_lon]


def grib_lon_to_wm3_indices(grib_lons):
    """
    Map GRIB longitude coordinates to WM-3 column indices.

    NOAA GFS cfgrib uses 0..360° columns that are already geographically aligned
    with WM-3 indices (col 720 = 180°/-180°). If GRIB arrives sorted -180..180,
    reorder columns so they match WM-3 / WindBorne sample layout.
    """
    import numpy as np

    grib_lons = np.asarray(grib_lons, dtype=np.float64)
    target = wm3_longitude_axis(len(grib_lons))
    if np.allclose(grib_lons, target, atol=0.01):
        return np.arange(len(grib_lons), dtype=int)
    if np.allclose(grib_lons, (target + 360.0) % 360.0, atol=0.01):
        return np.arange(len(grib_lons), dtype=int)

    idx = np.zeros(len(target), dtype=int)
    for i, lon_wm in enumerate(target):
        delta = np.abs(((grib_lons - lon_wm + 180.0) % 360.0) - 180.0)
        idx[i] = int(np.argmin(delta))
    return idx


def apply_lon_axis(arr, lon_idx, lon_axis):
    """Reorder one array along its longitude axis."""
    return np.take(arr, lon_idx, axis=lon_axis)


def find_latest_gfs_cycle(s3=None, max_lookback_hours=72):
    """Find the newest GFS f000 cycle on NOAA AWS Open Data."""
    from datetime import datetime, timedelta, timezone

    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    if s3 is None:
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    try:
        start = datetime.now(timezone.utc) - timedelta(hours=5)
        for hours_back in range(0, max_lookback_hours + 1, 6):
            t = start - timedelta(hours=hours_back)
            t = t.replace(hour=(t.hour // 6) * 6, minute=0, second=0, microsecond=0)
            date = t.strftime("%Y%m%d")
            hour = f"{t.hour:02d}"
            key = f"gfs.{date}/{hour}/atmos/gfs.t{hour}z.{GFS_ANALYSIS_FILE}"
            try:
                s3.head_object(Bucket=GFS_BUCKET, Key=key)
                return {"date": date, "hour": hour, "s3_key": key}
            except s3.exceptions.ClientError:
                continue
        raise RuntimeError(f"No GFS analysis found in the last {max_lookback_hours} hours")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to locate latest GFS analysis on S3: {e}") from e


def fetch_gfs_analysis(output_dir=RAW_GFS_DIR):
    """Download the latest GFS 0.25° f000 file from NOAA AWS Open Data."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    output_dir = Path(output_dir)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    cycle = find_latest_gfs_cycle(s3)
    date, hour, key = cycle["date"], cycle["hour"], cycle["s3_key"]

    dest = output_dir / date / hour / f"gfs.t{hour}z.{GFS_ANALYSIS_FILE}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not (dest.exists() and dest.stat().st_size > 0):
        try:
            s3.download_file(GFS_BUCKET, key, str(dest))
        except Exception as e:
            raise RuntimeError(f"Failed to download s3://{GFS_BUCKET}/{key}: {e}") from e

    return {"path": dest, "date": date, "hour": hour, "s3_key": key, "bucket": GFS_BUCKET}


def read_raw_gfs(date, hour, raw_dir=RAW_GFS_DIR):
    """Open a downloaded GFS f000 file with cfgrib (one dataset per GRIB type)."""
    hour = f"{int(hour):02d}"
    path = Path(raw_dir) / date / hour / f"gfs.t{hour}z.{GFS_ANALYSIS_FILE}"
    if not path.exists():
        raise FileNotFoundError(f"No raw GFS file at {path}. Run fetch_gfs_analysis() first.")

    # cfgrib may leave stale *.idx sidecars after interrupted reads; remove them
    for idx in path.parent.glob(f"{path.name}.*.idx"):
        idx.unlink(missing_ok=True)

    try:
        import cfgrib
        # indexpath="" builds a temp index instead of reusing sidecar files
        return cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    except Exception as e:
        raise RuntimeError(f"Failed to read GRIB file {path}: {e}") from e


def list_grib_fields(grib_path):
    """Scan a GRIB file; return unique messages as dicts (shortName, name, typeOfLevel, level, paramId)."""
    import eccodes

    grib_path = Path(grib_path)
    seen = {}
    with open(grib_path, "rb") as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break
            row = {
                "shortName": eccodes.codes_get(gid, "shortName"),
                "name": eccodes.codes_get(gid, "name"),
                "typeOfLevel": eccodes.codes_get(gid, "typeOfLevel"),
                "level": eccodes.codes_get(gid, "level"),
                "paramId": eccodes.codes_get(gid, "paramId"),
            }
            eccodes.codes_release(gid)
            seen[(row["shortName"], row["typeOfLevel"], row["level"])] = row
    return [seen[k] for k in sorted(seen)]


def match_wm3_field(inventory, wm_var):
    """Map a WM-3 sample variable name to a GRIB inventory row, or None if absent."""
    param_id = int(wm_var.split("_", 1)[0])
    suffix = wm_var.split("_", 1)[1]

    def first(pred):
        for row in inventory:
            if pred(row):
                return row
        return None

    if suffix in ("z", "t", "u", "v", "q"):
        row = first(lambda r: r["paramId"] == param_id and r["typeOfLevel"] == "isobaricInhPa")
        if row is None and suffix == "z":
            row = first(
                lambda r: r["typeOfLevel"] == "isobaricInhPa"
                and "geopotential height" in r["name"].lower()
            )
        return row

    row = first(lambda r: r["paramId"] == param_id)
    if row is not None:
        return row

    row = first(lambda r: r["shortName"] == suffix)
    if row is not None:
        return row

    if suffix == "msl":
        return first(lambda r: r["typeOfLevel"] == "meanSea" and "pressure" in r["name"].lower())
    if suffix == "tcc":
        return first(lambda r: r["shortName"] == "tcc" or "cloud cover" in r["name"].lower())
    if suffix in ("100u", "100v"):
        short = "u" if suffix == "100u" else "v"
        return first(
            lambda r: r["shortName"] == short
            and r["typeOfLevel"] == "heightAboveGround"
            and r["level"] == 100
        )
    return None


def build_field_map(grib_path):
    """Build cfgrib filter_by_keys for each WM-3 input variable from a GRIB scan."""
    inventory = list_grib_fields(grib_path)
    field_map = {}
    missing = []
    for wm_var in WM3_INPUT_VARS:
        row = match_wm3_field(inventory, wm_var)
        if row is None:
            missing.append(wm_var)
            continue
        filt = {"shortName": row["shortName"], "typeOfLevel": row["typeOfLevel"]}
        if row["typeOfLevel"] == "heightAboveGround":
            filt["level"] = row["level"]
        field_map[wm_var] = filt

    if missing:
        raise RuntimeError(
            f"GRIB file missing {len(missing)} WM-3 input field(s): {', '.join(missing)}"
        )
    return field_map


def extract_gfs_fields(grib_path):
    """
    Extract WM-3 fields from a GFS f000 file on the 721×1440 GFS grid.

    Field filters come from build_field_map() (GRIB scan, not hardcoded names).
    Returns physical units at native GFS pressure levels (25).
    """
    import cfgrib

    setup_weathermesh()
    from utils import levels_gfs

    grib_path = Path(grib_path)
    field_map = build_field_map(grib_path)

    for idx in grib_path.parent.glob(f"{grib_path.name}.*.idx"):
        idx.unlink(missing_ok=True)

    def open_field(filt):
        ds = cfgrib.open_dataset(
            grib_path, backend_kwargs={"filter_by_keys": filt, "indexpath": ""}
        )
        return ds[list(ds.data_vars)[0]]

    try:
        pr = {}
        lon_idx = None
        for wm_var in WM3_PRESSURE_VARS:
            da = open_field(field_map[wm_var])
            if lon_idx is None:
                lon_idx = grib_lon_to_wm3_indices(da.longitude.values)
            stacked = da.sel(isobaricInhPa=levels_gfs).values.transpose(1, 2, 0)
            stacked = apply_lon_axis(stacked, lon_idx, lon_axis=1)
            pr[wm_var] = stacked.copy()
        sfc = {}
        for wm_var in WM3_SFC_VARS:
            values = open_field(field_map[wm_var]).values.copy()
            sfc[wm_var] = apply_lon_axis(values, lon_idx, lon_axis=1)
        pr["129_z"] *= 9.80665
        sfc["45_tcc"] /= 100.0
    except Exception as e:
        raise RuntimeError(f"Failed to extract fields from {grib_path}: {e}") from e

    return {"levels": levels_gfs, "pr": pr, "sfc": sfc}


def normalize_encoder_fields(fields, mesh, pressure_levels, pr=None):
    """
    Z-score fields at native encoder levels using utils.load_normalization.

    Uses the norms dict from load_normalization indexed at pressure_levels
    (not mesh.normalization_matrix_* which is built for mesh.levels / 28 levels).
    Surface fields are normalized but not vertically interpolated.
    """
    import numpy as np
    from utils import levels_full, load_normalization

    with weathermesh_cwd():
        norms, _, _ = load_normalization(mesh, with_means=True)

    level_idx = [levels_full.index(level) for level in pressure_levels]

    if pr is None:
        pr = np.stack([fields["pr"][v] for v in WM3_PRESSURE_VARS], axis=2).astype(np.float32)
    for i, var in enumerate(mesh.pressure_vars):
        mean = np.array(norms[var]["mean"])[level_idx]
        std = np.array(norms[var]["std"])[level_idx]
        pr[..., i, :] = (pr[..., i, :] - mean) / std

    sfc = np.stack([fields["sfc"][v] for v in mesh.core_sfc_vars], axis=-1).astype(np.float32)
    for i, var in enumerate(mesh.core_sfc_vars):
        sfc[..., i] = (sfc[..., i] - norms[var]["mean"]) / norms[var]["std"]

    extras = {
        var: (fields["sfc"][var] - norms[var]["mean"]) / norms[var]["std"]
        for var in mesh.extra_sfc_vars
    }
    return {"pr": pr, "sfc": sfc, "extras": extras}


def save_encoder_npz(output_dir, encoder_name, timestamp, pr, sfc, extras):
    """Write WindBorne-format NPZ under data/preprocessed/{encoder_name}/."""
    import numpy as np
    from datetime import datetime, timezone

    month = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y%m")
    root = Path(output_dir) / encoder_name
    base_path = root / "f000" / month / f"{timestamp}.npz"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(base_path, pr=pr.astype(np.float16), sfc=sfc.astype(np.float16))

    extra_paths = {}
    for var, arr in extras.items():
        path = root / "extra" / var / month / f"{timestamp}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x=arr.astype(np.float16))
        extra_paths[var] = path

    logger.info("Wrote %s encoder data under %s for t=%s", encoder_name, root, timestamp)
    return {"base": base_path, "extras": extra_paths, "timestamp": timestamp}


def _cycle_timestamp(grib_path):
    from datetime import datetime, timezone

    hour = Path(grib_path).parent.name
    date = Path(grib_path).parent.parent.name
    return int(datetime.strptime(f"{date}{hour}", "%Y%m%d%H").replace(tzinfo=timezone.utc).timestamp())


def preprocess_gfs_encoder(grib_path, output_dir=PREPROCESSED_DIR, timestamp=None):
    """
    GFS encoder: extract -> normalize at 25 levels -> save.

    Normalization is before the 28-level interp (which runs at inference).
    """
    setup_weathermesh()
    from meshes import LatLonGrid
    from utils import levels_gfs, levels_medium

    grib_path = Path(grib_path)
    fields = extract_gfs_fields(grib_path)
    timestamp = timestamp or _cycle_timestamp(grib_path)

    with weathermesh_cwd():
        mesh = LatLonGrid(
            source="neogfs-25",
            extra_sfc_vars=WM3_EXTRA_SFC_VARS,
            extra_sfc_pad=9,
            input_levels=levels_gfs,
            levels=levels_medium,
        )

    normed = normalize_encoder_fields(fields, mesh, levels_gfs)
    return save_encoder_npz(output_dir, "neogfs", timestamp, **normed)


def interp_pressure_levels(pr, levels_in, levels_out):
    """
    Linear vertical interpolation on pressure data (H, W, n_vars, n_levels_in).

    Separate from utils.interp_levels because WM-3's version rejects downsampling
    when output levels aren't an exact subset of input levels (e.g. GFS 25 -> HRES 20).
    """
    import numpy as np

    out = np.empty((*pr.shape[:2], pr.shape[2], len(levels_out)), dtype=pr.dtype)
    for j, level in enumerate(levels_out):
        idx = np.searchsorted(levels_in, level)
        prev_i = max(idx - 1, 0)
        lo, hi = levels_in[prev_i], levels_in[idx]
        if idx == 0 or lo == hi:
            out[..., j] = pr[..., idx if idx < len(levels_in) else prev_i]
        else:
            t = (level - lo) / (hi - lo)
            out[..., j] = pr[..., prev_i] * (1 - t) + pr[..., idx] * t
    return out


def preprocess_hres_encoder(grib_path, output_dir=PREPROCESSED_DIR, timestamp=None):
    """
    HRES encoder: extract -> interp pressure 25->20 -> normalize at 20 levels -> save.

    Vertical interp to HRES levels happens before normalization; 28-level interp is at inference.
    """
    import numpy as np

    setup_weathermesh()
    from meshes import LatLonGrid
    from utils import levels_gfs, levels_hres, levels_medium

    grib_path = Path(grib_path)
    fields = extract_gfs_fields(grib_path)
    timestamp = timestamp or _cycle_timestamp(grib_path)

    with weathermesh_cwd():
        mesh = LatLonGrid(
            source="neohres-20",
            extra_sfc_vars=WM3_EXTRA_SFC_VARS,
            extra_sfc_pad=9,
            input_levels=levels_hres,
            levels=levels_medium,
        )

    pr = np.stack([fields["pr"][v] for v in WM3_PRESSURE_VARS], axis=2).astype(np.float32)
    pr = interp_pressure_levels(pr, levels_gfs, levels_hres)
    normed = normalize_encoder_fields(fields, mesh, levels_hres, pr=pr)
    return save_encoder_npz(output_dir, "neohres", timestamp, **normed)
