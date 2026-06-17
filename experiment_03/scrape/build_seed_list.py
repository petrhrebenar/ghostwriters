#!/usr/bin/env python3
"""Bootstrap the committed corpus seed list (NALUS file ids) from ECLIs.

Self-containment note
---------------------
This is the **one** place experiment_03 reads the legacy ``subset_disent2.csv``.
It runs once to materialise ``corpus/seed_spis_zn.txt`` — a committed, tracked
artifact that becomes the corpus's source of truth. After that the CSV is no
longer needed (repo loose-end: the root CSV can be removed once experiment_03
reaches parity). The scraper reads the committed seed list, never the CSV.

The only fix over the legacy ``scrapers/extract_spis_zn.py`` is widening the
ECLI *senate* group to accept ``Pl`` (plenum); the previous ``(\\d+)`` silently
dropped every plenary decision, which is why we had 236 instead of the full set.
Plenary decisions are full-court rulings and belong in the analysis.

NALUS file-id format (the ``sz`` GetText param / our JSON filename):
    ECLI:CZ:US:YEAR:SENATE.US.NUMBER.VERSION[.SUBVERSION]
      -> SENATE-NUMBER-VERSION              (4-component ECLI, no subversion)
      -> SENATE-NUMBER-VERSION_SUBVERSION   (5-component ECLI)
    SENATE is "1".."4" for senates (== I.ÚS .. IV.ÚS) and "Pl" for the plenum.

NOTE: the exact GetText ``sz`` token for plenary decisions (``Pl-...`` vs another
encoding) must be confirmed against the live site in the scrape stage; the
mapping here defines the corpus, the scraper resolves the fetch URL.

Usage:
    python -m scrape.build_seed_list --csv ../subset_disent2.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Optional, Tuple

PKG_DIR = Path(__file__).resolve().parent
EXP_DIR = PKG_DIR.parent
DEFAULT_CSV = EXP_DIR.parent / "subset_disent2.csv"
DEFAULT_OUT = EXP_DIR / "corpus" / "seed.tsv"

# Senate group widened from (\d+) to ([0-9A-Za-z]+) so "Pl" (plenum) matches.
RE_ECLI_5 = re.compile(r"ECLI:CZ:US:(\d+):([0-9A-Za-z]+)\.US\.(\d+)\.(\d+)\.(\d+)")
RE_ECLI_4 = re.compile(r"ECLI:CZ:US:(\d+):([0-9A-Za-z]+)\.US\.(\d+)\.(\d+)")


def ecli_to_file_id(ecli: str) -> Optional[str]:
    """Convert an ECLI to a NALUS file id (e.g. ``Pl-7-21_1`` / ``1-139-04``)."""
    m = RE_ECLI_5.match(ecli)
    if m:
        _year, senate, number, version, subversion = m.groups()
        return f"{senate}-{number}-{version}_{subversion}"
    m = RE_ECLI_4.match(ecli)
    if m:
        _year, senate, number, version = m.groups()
        return f"{senate}-{number}-{version}"
    return None


def pairs_from_csv(csv_path: Path) -> List[Tuple[str, str]]:
    """Return sorted unique (ecli, file_id) pairs. ECLI is the scrape search key."""
    csv.field_size_limit(10_000_000)
    pairs = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ecli = (row.get("doc_id") or "").strip()
            fid = ecli_to_file_id(ecli) if ecli else None
            if fid:
                pairs[fid] = ecli  # dedupe by file_id
    return sorted(pairs.items(), key=lambda kv: kv[0])  # (file_id, ecli) sorted by file_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="Legacy subset_disent2.csv (bootstrap source)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output seed TSV path")
    args = ap.parse_args()

    pairs = pairs_from_csv(Path(args.csv))  # (file_id, ecli)
    plenary = [fid for fid, _ in pairs if fid.startswith("Pl-")]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["file_id\tecli"] + [f"{fid}\t{ecli}" for fid, ecli in pairs]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(pairs)} (file_id, ecli) rows -> {out}")
    print(f"  plenary (Pl-*): {len(plenary)}  | senate: {len(pairs) - len(plenary)}")


if __name__ == "__main__":
    main()
