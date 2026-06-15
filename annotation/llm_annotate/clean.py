#!/usr/bin/env python3
"""Stage 02 — clean scraped HTML into numbered plain-text lines.

For each decision in ``data/01_scraped/`` we:
  * extract the body from ``td.DocContent`` and strip HTML,
  * drop blank lines (each remaining line is effectively one paragraph),
  * write three artifacts to ``data/02_cleaned/``:
      <id>.json   canonical {id, meta, lines:[...]}  (machine-readable)
      <id>.txt    metadata header + numbered lines    (human-readable)
      <id>.html   numbered-line preview               (browser-readable)

No annotation tags are inserted here. NO paragraph reconstruction is attempted
(the old prepare_for_tagging.py heuristics proved brittle); we keep whatever
lines the source provides.

Usage:
    python -m annotation.llm_annotate.clean
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from bs4 import BeautifulSoup

from .common import (
    DIR_CLEANED,
    DIR_SCRAPED,
    LABEL_COLORS,
    LABEL_DESC,
    META_FIELDS,
    decision_url,
    esc,
    meta_header_lines,
    numbered_text,
    read_json,
    write_json,
)


def extract_lines(full_text: str) -> List[str]:
    """Body lines from td.DocContent, blanks removed, each stripped."""
    soup = BeautifulSoup(full_text, "html.parser")
    node = soup.find("td", class_="DocContent") or soup.find("body") or soup
    raw = node.get_text(separator="\n")
    return [ln.strip() for ln in raw.split("\n") if ln.strip()]


def preview_html(rec_id: str, meta: Dict, lines: List[str]) -> str:
    header = "<br>".join(esc(h.lstrip("# ").rstrip()) for h in meta_header_lines(meta)[:-1])
    url = meta.get("url_address") or ""
    if url:
        header = header.replace(esc(url), f'<a href="{esc(url)}" target="_blank">{esc(url)}</a>')
    rows = "\n".join(
        f'<tr><td class="ln">{i}</td><td>{esc(ln)}</td></tr>'
        for i, ln in enumerate(lines, 1)
    )
    legend = " ".join(
        f'<span class="badge" style="background:{c}" title="{esc(LABEL_DESC.get(lbl, ""))}">{lbl}</span>'
        for lbl, c in LABEL_COLORS.items()
    )
    return f"""<!doctype html>
<html lang="cs"><head><meta charset="utf-8">
<title>{esc(rec_id)} — cleaned</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.5; }}
 .meta {{ background:#f4f4f4; padding:1rem; border-radius:6px; font-size:.9rem; }}
 .badge {{ color:#fff; padding:.1rem .4rem; border-radius:4px; font-size:.8rem; }}
 table {{ border-collapse: collapse; margin-top:1rem; }}
 td {{ vertical-align: top; padding:.15rem .6rem; }}
 td.ln {{ color:#999; text-align:right; user-select:none; font-variant-numeric: tabular-nums; }}
 tr:hover {{ background:#fafafa; }}
</style></head><body>
<h2>{esc(rec_id)}</h2>
<div class="meta">{header}<br><br>Legenda (po anotaci): {legend}</div>
<table>{rows}</table>
</body></html>
"""


def process(rec_id: str, decision: Dict) -> Dict:
    meta = {k: decision.get(k) for k in META_FIELDS}
    # Override the scraper's broken url_address (built from the human-format
    # spis_zn) with the working file-id URL.
    meta["url_address"] = decision_url(rec_id)
    lines = extract_lines(decision.get("full_text") or "")
    rec = {"id": rec_id, "meta": meta, "lines": lines}

    write_json(DIR_CLEANED / f"{rec_id}.json", rec)

    header = "\n".join(meta_header_lines(meta))
    (DIR_CLEANED / f"{rec_id}.txt").write_text(
        header + "\n\n" + numbered_text(lines) + "\n", encoding="utf-8"
    )
    (DIR_CLEANED / f"{rec_id}.html").write_text(
        preview_html(rec_id, meta, lines), encoding="utf-8"
    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="Process only first N (0 = all)")
    args = ap.parse_args()

    files = sorted(DIR_SCRAPED.glob("*.json"))
    if not files:
        print(f"No JSON files in {DIR_SCRAPED}. Run the ingest stage first.", file=sys.stderr)
        sys.exit(1)
    if args.limit:
        files = files[: args.limit]

    DIR_CLEANED.mkdir(parents=True, exist_ok=True)
    n_ok = n_empty = 0
    for fp in files:
        rec = process(fp.stem, read_json(fp))
        if rec["lines"]:
            n_ok += 1
        else:
            n_empty += 1
            print(f"  WARNING {fp.name}: no body lines extracted")

    print(f"Cleaned {n_ok} decisions ({n_empty} empty) -> {DIR_CLEANED}")


if __name__ == "__main__":
    main()
