import os
import sys
import zipfile
from pathlib import Path
from dotenv import load_dotenv

# Useful defaults for colorful printing (same as WeatherMesh-3/utils.py)
def MAGENTA(text): return f"\033[95m{text}\033[0m"
def ORANGE(text): return f"\033[38;5;214m{text}\033[0m"
def GREEN(text): return f"\033[92m{text}\033[0m"
def RED(text): return f"\033[91m{text}\033[0m"

# Default local paths
REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = REPO_ROOT / "weights" / "WeatherMesh3.pt"
DATA_DIR = REPO_ROOT / "data"
DATA_ZIP = REPO_ROOT / "data.zip"

# Google Drive links for weights and sample data
load_dotenv(REPO_ROOT / ".env")
GDRIVE_WEIGHTS_URL = os.environ.get("GDRIVE_WEIGHTS_URL")
GDRIVE_DATA_URL = os.environ.get("GDRIVE_DATA_URL")

def gdown_file(url, dest):
    """
    Download a file from Google Drive using gdown.

    Args:
        url (str): Google Drive share link from the assignment
        dest (Path): local file path
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Skip if we already have the file
    if dest.exists() and dest.stat().st_size > 0:
        print(ORANGE(f"Already exists, skipping: {dest}"))
        return

    print(MAGENTA(f"Downloading {url} -> {dest}"))
    try:
        import gdown
        gdown.download(url=url, output=str(dest), quiet=False)
    except Exception as e:
        print(RED(f"Failed to download {url}: {e}"))
        raise

    print(GREEN(f"Downloaded {dest} ({dest.stat().st_size / 1e6:.1f} MB)"))

def unzip_data(archive, dest_dir):
    """
    Extract sample data zip. Strips nested prefix up to data/ if present.
    """
    archive = Path(archive)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(MAGENTA(f"Extracting {archive} -> {dest_dir}/"))
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            names = zf.namelist()

            # WindBorne zip may be nested: huge/proc/.../data/neogfs/...
            strip = ""
            if names and "/" in names[0]:
                parts = names[0].split("/")
                for i, part in enumerate(parts):
                    if part == "data":
                        strip = "/".join(parts[:i+1]) + "/"
                        break

            for member in names:
                if member.endswith("/"):
                    continue
                target = member[len(strip):] if strip and member.startswith(strip) else member
                if not target:
                    continue

                out = dest_dir / target
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(zf.read(member))
    except Exception as e:
        print(RED(f"Failed to extract {archive}: {e}"))
        raise

    print(GREEN(f"Extracted sample data to {dest_dir}/"))

def verify_assets(weights_path, data_dir):
    """Check that weights and sample npz files are present."""
    weights_path = Path(weights_path)
    data_dir = Path(data_dir)

    assert weights_path.exists(), f"Missing weights: {weights_path}"
    assert weights_path.stat().st_size > 1e9, f"Weights file too small: {weights_path}"

    neogfs = list(data_dir.glob("neogfs/f000/*/*.npz"))
    neohres = list(data_dir.glob("neohres/f000/*/*.npz"))
    assert len(neogfs) > 0, f"No neogfs f000 npz under {data_dir}/neogfs/"
    assert len(neohres) > 0, f"No neohres f000 npz under {data_dir}/neohres/"

    print(GREEN("Verification OK: weights + sample data present"))

def download_assets(weights_url=None, data_url=None):
    """
    Pull model weights and sample data from Google Drive (WindBorne assignment links).
    """
    weights_url = weights_url or GDRIVE_WEIGHTS_URL
    data_url = data_url or GDRIVE_DATA_URL

    assert weights_url, "Set GDRIVE_WEIGHTS_URL in .env (link from Assignment.pdf)"
    assert data_url, "Set GDRIVE_DATA_URL in .env (link from Assignment.pdf)"

    # Download weights (~1.2 GB)
    gdown_file(weights_url, WEIGHTS_PATH)

    # Download sample data zip (~400 MB), then extract
    gdown_file(data_url, DATA_ZIP)
    unzip_data(DATA_ZIP, DATA_DIR)

    verify_assets(WEIGHTS_PATH, DATA_DIR)

if __name__ == "__main__":
    print(MAGENTA("Downloading WeatherMesh-3 assets from Google Drive..."))
    try:
        download_assets()
    except Exception as e:
        print(RED(f"Download failed: {e}"))
        sys.exit(1)

    print(MAGENTA("Done."))
