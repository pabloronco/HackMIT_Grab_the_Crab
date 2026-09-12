from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_response.data_fetch import R0_TARGETS, fetch_targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the exact R0 Dryad files and record local checksums."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace already-present local copies.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/r0_download_manifest.json"),
        help="Where to write provenance/checksum metadata.",
    )
    args = parser.parse_args()

    print("=== R0 VERIFIED DATA FETCH ===")
    print("Targets:")
    for target in R0_TARGETS:
        print(f"  {target.doi} -> {target.filename} -> {target.destination}")
    print()

    manifest = fetch_targets(overwrite=args.overwrite)

    for item in manifest:
        print(
            f"{item['status']:>15}  {item['filename']}  "
            f"{item['size']} bytes  sha256={item['sha256']}"
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print()
    print(f"Wrote provenance manifest: {args.manifest}")
    print("Next: python scripts/audit_real_data.py --json-out data/processed/r0_audit.json")


if __name__ == "__main__":
    main()
