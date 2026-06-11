# WeatherMesh-3 Inference Pipeline

Automated pipeline that fetches the latest GFS analysis, preprocesses it for WeatherMesh-3, runs a 6-hour forecast on GPU, validates output, and uploads a CF-compliant NetCDF file to S3.

## Prerequisites

- Linux host with **NVIDIA GPU** and CUDA drivers (tested on Lambda Labs A10)
- **Python 3.10+**
- **Git**
- An **AWS S3 bucket** for forecast uploads (public read optional but recommended for reviewers)
- Google Drive links from the assignment for model weights and sample data

## Setup

### 1. Clone repositories

```bash
git clone git@github.com:GokulnathUW/weathermesh-inference.git
cd weathermesh-inference
git clone git@github.com:windborne/WeatherMesh-3.git
```

`WeatherMesh-3/` must live at the repo root (sibling to `pipeline.py`). Do not modify files under `WeatherMesh-3/` or `weights/`.

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
bash install_all.sh
```

`install_all.sh` installs WeatherMesh-3 dependencies (PyTorch, natten, matepoint) and this repo’s `requirements.txt`.

### 3. Environment variables

```bash
cp .env.example .env
```

Edit `.env`:

| Variable | Purpose |
|----------|---------|
| `GDRIVE_WEIGHTS_URL` | Google Drive link for `WeatherMesh3.pt` (from assignment) |
| `GDRIVE_DATA_URL` | Google Drive link for sample `data.zip` (from assignment) |
| `AWS_ACCESS_KEY_ID` | IAM user with `s3:PutObject` on your output bucket |
| `AWS_SECRET_ACCESS_KEY` | Matching secret key |
| `AWS_DEFAULT_REGION` | **Region where your bucket was created** (e.g. `us-east-2`) |
| `AWS_S3_BUCKET` | Output bucket name (e.g. `weathermesh-outputs`) |
| `ALERT_WEBHOOK_URL` | Optional Slack-compatible webhook on pipeline failure |

### 4. Download weights and sample data

```bash
python3 scripts/download_assets.py
```

This creates:

- `weights/WeatherMesh3.pt` (~1.2 GB)
- `data/sample/` — reference NPZ files for validation

### 5. Verify preprocessing (optional)

After a manual preprocess or first pipeline run, compare your encoder output to the WindBorne sample:

```bash
python3 scripts/validate_preprocess.py \
  --sample-timestamp 1741305600 \
  --processed-timestamp <your_timestamp>
```

Replace `<your_timestamp>` with the unix init time printed by the pipeline or found under `data/preprocessed/`.

## Running the pipeline

One full cycle from the repo root:

```bash
source .venv/bin/activate
python3 pipeline.py
```

Steps executed:

1. Download latest GFS 0.25° **f000** analysis from NOAA’s public S3 bucket  
2. Preprocess → `data/preprocessed/neogfs/` and `neohres/`  
3. Run WeatherMesh-3 inference (6 h lead) on CUDA  
4. Validate forecast (shapes, finite values, 2 m temperature range)  
5. Write outputs locally and upload NetCDF to S3  

Logs and status:

- `outputs/pipeline.log` — JSON event log  
- `outputs/pipeline_status.json` — last cycle summary  
- `outputs/forecast_<timestamp>_f006.nc` — CF NetCDF  
- `outputs/windborne_<timestamp>_f006/` — WindBorne API-format files  

Typical wall time: ~80–90 s (preprocess ~60 s, forward pass ~8 s on A10).

## Automated runs (cron)

Every 6 hours (aligned with GFS cycles):

```bash
crontab -e
```

Add:

```cron
0 */6 * * * /home/ubuntu/weathermesh-inference/scripts/run_pipeline_cron.sh
```

Use the **absolute path** to `run_pipeline_cron.sh`. The wrapper sources `.env`, activates `.venv` if present, and appends to `outputs/pipeline.log` (do not add `>> pipeline.log` in the cron line).

## Model outputs on S3

**Bucket:** `s3://weathermesh-outputs/`  
**Region:** `us-east-2`  
**Access:** Public read — no credentials required  

**Object layout:**

```
forecasts/forecast_{unix_timestamp}_f006.nc
```

**Example:**

```
https://weathermesh-outputs.s3.us-east-2.amazonaws.com/forecasts/forecast_1781179200_f006.nc
```

**List forecasts:**

```bash
aws s3 ls s3://weathermesh-outputs/forecasts/ --region us-east-2 --no-sign-request
```

**Download:**

```bash
aws s3 cp s3://weathermesh-outputs/forecasts/forecast_1781179200_f006.nc . \
  --region us-east-2 --no-sign-request
```

Use the bucket region in URLs and CLI flags. A wrong region (e.g. `us-east-1`) returns a redirect error in the browser.

## Project layout

```
weathermesh-inference/
├── pipeline.py              # Main entry point
├── install_all.sh           # Install WM-3 + inference deps
├── inference/
│   ├── preprocess.py        # GFS fetch + encoder preprocessing
│   ├── inference.py         # Model forward pass
│   ├── export.py            # NetCDF + WindBorne API export
│   └── monitor.py           # Pre-upload validation + alerts
├── scripts/
│   ├── download_assets.py   # Pull weights/sample from Google Drive
│   ├── validate_preprocess.py
│   └── run_pipeline_cron.sh
├── WeatherMesh-3/           # Upstream model (cloned separately)
├── weights/                 # WeatherMesh3.pt (not in git)
├── data/
│   ├── sample/              # Reference NPZ (not in git)
│   ├── raw/gfs/             # Downloaded GRIB
│   └── preprocessed/        # Encoder NPZ for inference
└── outputs/                 # Forecasts, logs (not in git)
```

## Repository

**Code + setup:** https://github.com/GokulnathUW/weathermesh-inference

**S3 outputs (public):** https://weathermesh-outputs.s3.us-east-2.amazonaws.com/forecasts/forecast_1781179200_f006.nc  
