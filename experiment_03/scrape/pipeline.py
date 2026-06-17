#!/usr/bin/env python3
"""Stage 1 — scrape NALUS record cards + bodies into ``data/01_scraped/``.

For each (file_id, ecli) in ``corpus/seed.tsv``:
  1. search the ECLI -> ResultDetail id (expect exactly 1 hit);
  2. fetch + cache the record-card HTML (``data/00_corpus/raw_html/``);
  3. normalize it (parse_card) -> metadata (rapporteur, clean separate_opinion, …);
  4. body text: **reuse** an existing ``data/01_scraped/<file_id>.json`` full_text
     if present (the 236 already-annotated decisions — their tags are tied to that
     exact text, so we never refetch/alter it), else GetText the new body;
  5. write the merged record.

Resumable: skips records already enriched (``_meta_source == "record_card"``)
unless ``--overwrite``. Re-parsing from cached HTML never re-hits the network.

Usage (from experiment_03/, in the .venv with selenium):
    .venv/bin/python -m scrape.pipeline --limit 3        # smoke test
    .venv/bin/python -m scrape.pipeline                  # full corpus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from .parse_card import normalize_card
from .scraper import NalusScraper

EXP_DIR = Path(__file__).resolve().parent.parent
SEED = EXP_DIR / "corpus" / "seed.tsv"
DIR_SCRAPED = EXP_DIR / "data" / "01_scraped"
DIR_RAW_HTML = EXP_DIR / "data" / "00_corpus" / "raw_html"

# Fields written to the scraped record (metadata + provenance + body).
OUT_FIELDS = [
    "doc_id", "spis_zn", "date_decision", "date_submission", "date_publication",
    "type_decision", "type_proceedings", "formation", "judge_rapporteur_name",
    "separate_opinion", "type_verdict", "grounds", "applicant", "concerned_body",
    "disputed_act", "subject_proceedings", "subject_register", "popular_name",
    "importance", "url_address", "file_id",
]


def load_seed(path: Path) -> List[Tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("file_id\t"):
            continue
        fid, ecli = line.split("\t")
        rows.append((fid.strip(), ecli.strip()))
    return rows


def _already_enriched(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("_meta_source") == "record_card"
    except Exception:
        return False


def run(limit: int = 0, overwrite: bool = False, headless: bool = True,
        refresh_html: bool = False, only: str = "") -> None:
    DIR_SCRAPED.mkdir(parents=True, exist_ok=True)
    DIR_RAW_HTML.mkdir(parents=True, exist_ok=True)

    seed = load_seed(SEED)
    if only:
        wanted = set(only.split(","))
        seed = [r for r in seed if r[0] in wanted]
    if limit:
        seed = seed[:limit]

    stats = {"done": 0, "skipped": 0, "enriched": 0, "new_body": 0,
             "no_hit": 0, "multi_hit": 0, "id_mismatch": 0, "failed": 0}

    with NalusScraper(headless=headless) as sc:
        for i, (file_id, ecli) in enumerate(seed, 1):
            out_path = DIR_SCRAPED / f"{file_id}.json"
            if _already_enriched(out_path) and not overwrite:
                stats["skipped"] += 1
                continue

            print(f"[{i}/{len(seed)}] {file_id}  ({ecli})")
            try:
                # 1. search -> detail id
                detail_id, total = sc.search_ecli(ecli)
                if total != 1:
                    print(f"    {'no' if total == 0 else total} hits — skipping")
                    stats["no_hit" if total == 0 else "multi_hit"] += 1
                    continue

                # 2. card HTML (cache)
                html_path = DIR_RAW_HTML / f"{file_id}.card.html"
                if html_path.exists() and not refresh_html:
                    html = html_path.read_text(encoding="utf-8")
                else:
                    html = sc.card_html(detail_id)
                    html_path.write_text(html, encoding="utf-8")

                # 3. normalize
                meta = normalize_card(html)
                card_fid = meta.get("file_id")
                if card_fid and card_fid != file_id:
                    print(f"    WARN file_id mismatch: seed={file_id} card={card_fid}")
                    stats["id_mismatch"] += 1

                # 4. body: reuse existing full_text, else fetch
                existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
                full_text = existing.get("full_text")
                if full_text:
                    stats["enriched"] += 1
                else:
                    full_text = sc.fetch_body(file_id)
                    if not full_text:
                        print("    body fetch failed — skipping")
                        stats["failed"] += 1
                        continue
                    stats["new_body"] += 1

                # 5. write merged record
                rec: Dict = {k: meta.get(k) for k in OUT_FIELDS}
                rec["full_text"] = full_text
                rec["_meta_source"] = "record_card"
                rec["_detail_id"] = detail_id
                rec["_warnings"] = meta.get("_warnings", [])
                out_path.write_text(
                    json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
                stats["done"] += 1
                rap = rec.get("judge_rapporteur_name")
                sep = rec.get("separate_opinion")
                print(f"    rapporteur={rap!r} dissents={sep}")
            except Exception as e:  # keep going; one bad record shouldn't abort
                print(f"    ERROR: {type(e).__name__}: {e}")
                stats["failed"] += 1

    print("\n" + "=" * 60)
    print("Scrape summary:", json.dumps(stats, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-headless", dest="headless", action="store_false")
    ap.add_argument("--refresh-html", action="store_true", help="Ignore cached card HTML")
    ap.add_argument("--only", default="", help="Comma-separated file_ids to scrape (testing)")
    args = ap.parse_args()
    run(limit=args.limit, overwrite=args.overwrite, headless=args.headless,
        refresh_html=args.refresh_html, only=args.only)


if __name__ == "__main__":
    main()
