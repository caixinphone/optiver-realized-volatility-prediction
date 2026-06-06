"""Download Optiver Realized Volatility Prediction data via the Kaggle API.

Prereqs: ~/.kaggle/kaggle.json (or KAGGLE_KEY env), and accept rules on the website:
https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/rules
"""
import subprocess
import sys
import zipfile
from pathlib import Path

COMP = "optiver-realized-volatility-prediction"
DATA_DIR = Path(__file__).parent / "data"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Downloading '{COMP}' (several GB) ...")
    try:
        subprocess.run(["kaggle", "competitions", "download", "-c", COMP, "-p", str(DATA_DIR)],
                       check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        sys.exit(f"Download failed: {exc}\nCheck kaggle.json / KAGGLE_KEY and that you accepted the rules.")
    for z in DATA_DIR.glob("*.zip"):
        print(f"Unzipping {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(DATA_DIR)
    print("Done. Top-level files:")
    for p in sorted(DATA_DIR.iterdir())[:20]:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
