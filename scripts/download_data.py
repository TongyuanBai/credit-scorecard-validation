"""Download the Lending Club extract used in this repository.

The file is the `lending_club` data set of the R package `modeldata`, as
published by the Rdatasets project. It is downloaded to data/lending_club.csv
and checked against the SHA-256 checksum of the copy used for the reported
results.

Usage (from the repository root):
    python scripts/download_data.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/modeldata/lending_club.csv"
SHA256 = "0e27aa7be8f496fde17083a7ca28487de84ee0e66da6fe09c0d69f19d5adcb7f"
TARGET = Path(__file__).resolve().parents[1] / "data" / "lending_club.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if not TARGET.exists():
        print(f"Downloading {URL}")
        partial = TARGET.with_suffix(".part")
        urllib.request.urlretrieve(URL, partial)
        partial.replace(TARGET)  # only a complete download gets the final name
    digest = sha256(TARGET)
    if digest != SHA256:
        print(f"Warning: checksum {digest} differs from the copy used for the reported results ({SHA256}).")
        print("The source may have been updated; results can differ from those in reports/.")
        return 1
    print(f"{TARGET} is in place and matches the expected checksum.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
