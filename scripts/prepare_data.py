"""Prepare verified Olist V2 and UCI Bike Sharing snapshots locally."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile, is_zipfile

import duckdb

from scripts.build_duckdb import build_database


ROOT = Path("data")
OLIST_URL = "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce?datasetVersionNumber=2"
BIKE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
BIKE_SHA = "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f"


def verify(path: Path, digest: str) -> None:
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != digest:
        raise ValueError(f"Unexpected SHA-256 for {path}")


def fetch(url: str, name: str, files: dict[Path, str]) -> None:
    missing = [path for path in files if not path.exists()]
    for path in files.keys() - set(missing):
        verify(path, files[path])
    if not missing:
        return
    destination = ROOT / "downloads" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or not is_zipfile(destination):
        subprocess.run(["curl", "-fLsS", "--max-time", "360", "-C", "-", "-A",
                        "Kaggle API Client", url, "-o", str(destination)], check=True)
    with ZipFile(destination) as archive:
        for path in missing:
            temporary = path.with_suffix(".part")
            with archive.open(path.name) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            verify(temporary, files[path])
            temporary.chmod(0o444)
            temporary.replace(path)


def prepare_olist() -> None:
    files = {Path(filename): digest for digest, filename in
             (line.split() for line in Path("docs/olist_raw.sha256").read_text().splitlines())}
    if any(not path.exists() for path in files):
        cache = Path.home() / ".cache/kagglehub/datasets/olistbr/brazilian-ecommerce/versions/2"
        for path, digest in files.items():
            candidate = cache / path.name
            if candidate.is_file() and not path.exists():
                verify(candidate, digest)
                shutil.copyfile(candidate, path)
                path.chmod(0o444)
    fetch(OLIST_URL, "olist-v2.zip", files)
    output = ROOT / "processed/olist.duckdb"
    if not output.exists():
        print(build_database(output_path=output))


def prepare_bike() -> None:
    path = ROOT / "raw/hour.csv"
    fetch(BIKE_URL, "bike.zip", {path: BIKE_SHA})
    output = ROOT / "processed/bike.duckdb"
    if output.exists():
        return
    with TemporaryDirectory(dir=output.parent) as staging:
        temporary = Path(staging) / output.name
        with duckdb.connect(str(temporary)) as connection:
            connection.execute("BEGIN")
            connection.execute("""
                CREATE TABLE bike_hours (
                    dteday DATE, hr INTEGER CHECK (hr BETWEEN 0 AND 23),
                    cnt INTEGER NOT NULL CHECK (cnt >= 0), workingday INTEGER NOT NULL,
                    weathersit INTEGER NOT NULL, season INTEGER NOT NULL,
                    PRIMARY KEY (dteday, hr)
                )
            """)
            connection.execute("""
                INSERT INTO bike_hours
                SELECT dteday, hr, cnt, workingday, weathersit, season
                FROM read_csv_auto(?) WHERE cnt = casual + registered
            """, [str(path)])
            if connection.execute("SELECT COUNT(*) FROM bike_hours").fetchone()[0] != 17379:
                raise ValueError("Expected 17,379 unique, consistent hourly records")
            connection.execute("COMMIT")
        temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=("olist", "bike", "all"), default="all", nargs="?")
    args = parser.parse_args()
    (ROOT / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "processed").mkdir(parents=True, exist_ok=True)
    for name in ("olist", "bike") if args.domain == "all" else (args.domain,):
        {"olist": prepare_olist, "bike": prepare_bike}[name]()


if __name__ == "__main__":
    main()
