from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_response.local_data_import import import_downloaded_targets, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify exact Dryad CSVs from a local download folder and copy them into data/raw."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path.home() / "Downloads",
        help="Folder containing the four browser-downloaded CSVs (default: ~/Downloads).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/r0_download_manifest.json"),
    )
    args = parser.parse_args()

    records = import_downloaded_targets(
        args.source_dir,
        overwrite=args.overwrite,
    )
    write_manifest(records, args.manifest)

    print("=== R0 VERIFIED LOCAL IMPORT ===")
    for record in records:
        print(
            f"verified {record['filename']} | {record['size']} bytes | "
            f"sha256={record['sha256'][:12]}... -> {record['destination']}"
        )
    print(f"Wrote provenance manifest: {args.manifest}")


if __name__ == "__main__":
    main()
