"""
Step 0. Download the public data file and check its fingerprint.

UCI Heart Disease, Cleveland subset: 303 patients, 13 clinical features. The notebooks read it
from data/raw/heart_disease.csv, which is not in the repository. This puts it there.

Usage:  python pipeline/00_fetch_data.py
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA_SHA256, DATA_URL, RAW, sha256

if __name__ == "__main__":
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        print(f"downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, RAW)
    found = sha256(RAW)
    if found != DATA_SHA256:
        raise SystemExit(f"The file's fingerprint is {found}, not the one this project was built on ({DATA_SHA256}). Delete it and run again.")
    print(f"{RAW.relative_to(RAW.parent.parent.parent)}: {len(RAW.read_text().splitlines())} patients, fingerprint as expected")
