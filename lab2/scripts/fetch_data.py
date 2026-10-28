"""Fetch the 86 consecutive VIC# frames 2732..2817 over HTTP and crop them.

About 1 GB is downloaded (byte ranges of the 61.5 GiB archive, CRC32-checked
by the LR1 downloader); present files are re-verified, not re-downloaded.
The cropped block is cached as data/raw/turbine_1L/measurement_crop.npz.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab02 import config, realdata  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete-raw",
        action="store_true",
        help="remove the 3.7 GB of text frames after a verified crop",
    )
    args = parser.parse_args()
    start = time.perf_counter()
    records = realdata.fetch(config.RAW)
    print(f"verified {len(records)} frames in {time.perf_counter() - start:.0f} s", flush=True)
    data = config.real_data()
    print(
        f"cropped {data.velocity.shape}, identical to LR1 frames: {data.meta['identical_to_lr1']}"
    )
    realdata.write_manifest(records, config.RAW, config.ROOT / "data" / "turbine_1L_manifest.json")
    if args.delete_raw:
        for path in config.RAW.glob("B[0-9][0-9][0-9][0-9][0-9].txt"):
            path.unlink()


if __name__ == "__main__":
    main()
