#!/usr/bin/env python3
"""Stage 01 — ingest scraped decision JSON into ``data/01_scraped/``.

Copies the nalus_v2 scraper output into the pipeline's own data tree so the
rest of the pipeline is decoupled from the scraper location (reproducibility).

Usage:
    python -m annotation.llm_annotate.ingest \
        --input scrapers/nalus_v2/data/decisions
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .common import DEFAULT_SCRAPER_DIR, DIR_SCRAPED


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(DEFAULT_SCRAPER_DIR), help="Scraper decisions dir")
    ap.add_argument("--limit", type=int, default=0, help="Ingest only first N (0 = all)")
    args = ap.parse_args()

    src = Path(args.input)
    files = sorted(src.glob("*.json"))
    if not files:
        print(f"No JSON files in {src}", file=sys.stderr)
        sys.exit(1)
    if args.limit:
        files = files[: args.limit]

    DIR_SCRAPED.mkdir(parents=True, exist_ok=True)
    for fp in files:
        shutil.copy2(fp, DIR_SCRAPED / fp.name)

    print(f"Ingested {len(files)} decisions -> {DIR_SCRAPED}")


if __name__ == "__main__":
    main()
