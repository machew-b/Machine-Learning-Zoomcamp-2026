#!/usr/bin/env python
"""Download the Seoul Bike Sharing Demand dataset and check that its schema is as expected.

    python download_data.py                    # download to data/SeoulBikeData.csv
    python download_data.py --force            # download again
    python download_data.py --from-zip FILE    # use the zip you downloaded manually
    python download_data.py --from-csv FILE    # use the csv you downloaded manually

Source: UCI Machine Learning Repository, dataset 560 (CC BY 4.0), doi:10.24432/C5F62R.
Only file names are printed, never full local paths.
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import common

HERE = Path(__file__).resolve().parent


def fetch(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (capstone-1 download script)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def csv_from_zip(zip_bytes: bytes) -> bytes:
    """Return the bytes of the dataset's CSV file inside the zip archive."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        preferred = [name for name in members if Path(name).name.lower() == common.CSV_NAME.lower()]
        chosen = preferred or members
        if not chosen:
            raise ValueError("The zip archive does not contain a .csv file.")
        if len(chosen) > 1:
            raise ValueError("The zip archive contains several .csv files: " + ", ".join(chosen))
        return archive.read(chosen[0])


def validate(csv_path: Path):
    """Load the file with the project's loader (schema check) and print a short summary."""
    frame = common.load_raw(csv_path)
    print(
        f"{csv_path.name}: {len(frame)} rows, {frame.shape[1]} columns, "
        f"{frame['date'].min().date()} to {frame['date'].max().date()}"
    )
    if len(frame) != common.EXPECTED_ROWS:
        print(f"WARNING: expected {common.EXPECTED_ROWS} rows (365 days x 24 hours), found {len(frame)}.")
    return frame


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Download and validate the Seoul bike-sharing dataset.")
    parser.add_argument("--dest-dir", default=str(HERE / "data"), help="folder for the csv (default: data/)")
    parser.add_argument("--url", default=common.DATASET_URL, help="zip url (default: the UCI static download)")
    parser.add_argument("--from-zip", help="use a zip file you downloaded manually instead of downloading")
    parser.add_argument("--from-csv", help="use a csv file you downloaded manually instead of downloading")
    parser.add_argument("--force", action="store_true", help="download again even if the csv already exists")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / common.CSV_NAME

    try:
        if args.from_csv:
            source = Path(args.from_csv)
            if not source.exists():
                print(f"ERROR: {source.name} does not exist.")
                return 1
            if source.resolve() != dest.resolve():
                shutil.copyfile(source, dest)
            print(f"Using the csv you provided ({source.name}).")
        elif args.from_zip:
            source = Path(args.from_zip)
            if not source.exists():
                print(f"ERROR: {source.name} does not exist.")
                return 1
            dest.write_bytes(csv_from_zip(source.read_bytes()))
            print(f"Extracted {dest.name} from {source.name}.")
        elif dest.exists() and not args.force:
            print(f"{dest.name} already exists (use --force to download it again).")
        else:
            print("Downloading the dataset from the UCI Machine Learning Repository ...")
            dest.write_bytes(csv_from_zip(fetch(args.url)))
            print(f"Saved {dest.name}.")
        validate(dest)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"ERROR: download failed ({exc}).")
        print("Download the zip manually from https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand")
        print("and run:  python download_data.py --from-zip <the zip file you downloaded>")
        return 1
    except (zipfile.BadZipFile, ValueError) as exc:
        print(f"ERROR: {exc}")
        print("If the file in data/ is damaged or incomplete, run again with --force.")
        return 1

    print("Data licence: CC BY 4.0 (UCI Machine Learning Repository, doi:10.24432/C5F62R).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
