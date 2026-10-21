"""Download the UCI "Online Shoppers Purchasing Intention" dataset into ``data/``.

Usage:
    python download_data.py            # skips the download if the file exists
    python download_data.py --force    # download again
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)
CSV_NAME = "online_shoppers_intention.csv"
DEFAULT_DEST = Path(__file__).resolve().parent / "data" / CSV_NAME
EXPECTED_COLUMNS = {
    "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues",
    "SpecialDay", "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend", "Revenue",
}


def download(dest: Path | str = DEFAULT_DEST, url: str = URL, force: bool = False) -> Path:
    """Fetch the zip archive, extract the CSV to ``dest`` and return its path."""
    dest = Path(dest)
    if dest.exists() and not force:
        print(f"Dataset already present: {dest.name}")
        return dest

    print(f"Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (mlzoomcamp-midterm)"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except Exception as exc:  # network errors, HTTP errors, ...
        raise SystemExit(
            f"Could not download the dataset ({exc}).\n"
            "Download it manually from https://archive.ics.uci.edu/dataset/468 and save the CSV as\n"
            f"  {dest}"
        ) from exc

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [n for n in archive.namelist() if n.endswith(CSV_NAME)]
        if not members:
            raise SystemExit(f"{CSV_NAME} not found in the downloaded archive: {archive.namelist()}")
        csv_bytes = archive.read(members[0])

    header = csv_bytes.split(b"\n", 1)[0].decode("utf-8-sig").strip().split(",")
    missing = EXPECTED_COLUMNS - {h.strip() for h in header}
    if missing:
        raise SystemExit(f"Downloaded file has an unexpected schema, missing: {sorted(missing)}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(csv_bytes)
    print(f"Saved {len(csv_bytes) / 1e6:.2f} MB to {dest.name}")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="where to save the CSV")
    args = parser.parse_args()
    download(args.dest, force=args.force)
    sys.exit(0)
