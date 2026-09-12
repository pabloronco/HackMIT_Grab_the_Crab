# Local raw-data layout

Raw third-party datasets are intentionally **not committed** to Git. Keep provenance in `docs/REAL_DATA_AUDIT_V0.md` and `configs/data_sources.yaml`.

The preferred path is the reproducible downloader:

```bash
python scripts/fetch_r0_data.py
```

It discovers the latest published Dryad versions through Dryad metadata, selects the exact filenames below, downloads them through Dryad's public file-stream route, checks the API-reported byte size when available, computes SHA-256 checksums, and writes `data/processed/r0_download_manifest.json`.

Expected local layout:

```text
data/raw/
  wsg_cama/
    all_abundance_wide_format.csv
    DailyMaxTemperature.csv
  wsg_effort_geo/
    PAMA.Month.CPUE.csv
    Network_PAMA_CPUE.Map.csv
```

Sources:

- Washington Sea Grant / WDFW / UW Dryad dataset: DOI `10.5061/dryad.0rxwdbsdt`
- Washington Sea Grant / UW Dryad dataset: DOI `10.5061/dryad.sqv9s4ndp`

Important: the second dataset studies *Palaemon macrodactylus*. We reuse only shared Crab Team sampling metadata (`trap.sets`, site coordinates, protocol structure). PAMA outcomes are never treated as European green crab evidence.

If Dryad changes its download interface and the automated fetch fails, do not substitute similar-looking files. Use the DOI landing pages and verify exact filenames before manual download.
