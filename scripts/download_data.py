"""Download PhysioNet/CinC Challenge 2012 (open access) into data/physionet2012.

Source: https://physionet.org/content/challenge-2012/1.0.0/
Usage:  python scripts/download_data.py
"""
import io
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import RAW_DIR  # noqa: E402

BASE = "https://physionet.org/files/challenge-2012/1.0.0"


def fetch(url: str) -> bytes:
    print(f"GET {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if (RAW_DIR / "set-a").exists() and any((RAW_DIR / "set-a").glob("*.txt")):
        print("Set-A already present, skipping.")
    else:
        data = fetch(f"{BASE}/set-a.zip")
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(RAW_DIR)
        print("Extracted set-a.")
    out = RAW_DIR / "Outcomes-a.txt"
    if not out.exists():
        out.write_bytes(fetch(f"{BASE}/Outcomes-a.txt"))
    print("Done:", RAW_DIR)


if __name__ == "__main__":
    main()
