#!/usr/bin/env python3
"""Stage 3 — join annotated spans + scraped metadata into the analysis dataset.

Replaces the dependency on ``subset_disent2.csv``. Reads:
  * ``data/07_spans/<id>.json``   — extract_tags output (labelled spans, in order)
  * ``data/01_scraped/<id>.json`` — scraped metadata (rapporteur, separate_opinion, ...)

Writes two tables to ``data/dataset/`` (parquet + csv):
  * ``decisions``  — one row per decision: concatenated RATIO text (the court's
    own reasoning, authored under the rapporteur) + metadata. This is the
    *scoring* target — the key improvement over experiment-02, which scored the
    whole contaminated decision text.
  * ``dissents``   — one row per DIS/CON span: the separate-opinion prose with
    its attributed judge (see authorship.attribute_spans). This is the per-judge
    *training* data.

Usage:
    python build_dataset.py            # from experiment_03/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from authorship import attribute_spans, heading_name

EXP_DIR = Path(__file__).resolve().parent
DIR_SPANS = EXP_DIR / "data" / "07_spans"
DIR_SCRAPED = EXP_DIR / "data" / "01_scraped"
DIR_OUT = EXP_DIR / "data" / "dataset"

META_FIELDS = [
    "spis_zn", "date_decision", "formation", "type_decision",
    "judge_rapporteur_name", "separate_opinion",
]


def _wc(text: str) -> int:
    return len(text.split())


def _load_meta(doc_id: str) -> Dict:
    fp = DIR_SCRAPED / f"{doc_id}.json"
    if not fp.exists():
        return {}
    raw = json.loads(fp.read_text(encoding="utf-8"))
    return {k: raw.get(k) for k in META_FIELDS}


def build(spans_dir: Path = DIR_SPANS, out_dir: Path = DIR_OUT) -> None:
    decisions: List[Dict] = []
    dissents: List[Dict] = []

    for fp in sorted(spans_dir.glob("*.json")):
        if fp.name == "annotated.jsonl":
            continue
        rec = json.loads(fp.read_text(encoding="utf-8"))
        doc_id = rec["id"]
        spans = rec.get("spans", [])
        meta = _load_meta(doc_id)
        sep = meta.get("separate_opinion") or []

        # --- decisions row: concatenated RATIO (document order) ---
        ratio_spans = [s["text"] for s in spans if s["label"] == "RATIO"]
        ratio_text = "\n\n".join(ratio_spans)
        decisions.append({
            "doc_id": doc_id,
            **{k: meta.get(k) for k in META_FIELDS if k != "separate_opinion"},
            "separate_opinion": "; ".join(sep) if isinstance(sep, list) else sep,
            "n_ratio_spans": len(ratio_spans),
            "ratio_words": _wc(ratio_text),
            "ratio_text": ratio_text,
        })

        # --- dissents rows: one per DIS/CON span, attributed to a judge ---
        for span, author, source in attribute_spans(spans, sep if isinstance(sep, list) else None):
            dissents.append({
                "doc_id": doc_id,
                "author": author,
                "author_source": source,
                "heading_name": heading_name(span["text"]),
                "label": span["label"],
                "words": _wc(span["text"]),
                "text": span["text"],
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    dec_df = pd.DataFrame(decisions)
    dis_df = pd.DataFrame(dissents)
    for name, df in (("decisions", dec_df), ("dissents", dis_df)):
        df.to_csv(out_dir / f"{name}.csv", index=False)
        try:  # parquet preferred (preserves list columns / dtypes) but optional
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
        except Exception as e:
            print(f"  (parquet skipped for {name}: {type(e).__name__}; CSV written)")

    # --- report ---
    print(f"decisions: {len(dec_df)} rows  (with RATIO: {(dec_df['n_ratio_spans'] > 0).sum()})")
    print(f"  rapporteur present: {dec_df['judge_rapporteur_name'].notna().sum()}/{len(dec_df)}")
    print(f"dissents:  {len(dis_df)} rows  (DIS={ (dis_df['label']=='DIS').sum() }, CON={ (dis_df['label']=='CON').sum() })")
    if len(dis_df):
        by_src = dis_df["author_source"].value_counts().to_dict()
        print(f"  author source: {by_src}")
        keep = dis_df[dis_df["author"] != "UNKNOWN"]
        counts = keep["author"].value_counts()
        print(f"  distinct authors: {keep['author'].nunique()}  | unattributed: {(dis_df['author']=='UNKNOWN').sum()}")
        print(f"  authors with >=5 opinions: {(counts >= 5).sum()}")
    print(f"-> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spans", default=str(DIR_SPANS))
    ap.add_argument("--out", default=str(DIR_OUT))
    args = ap.parse_args()
    build(Path(args.spans), Path(args.out))


if __name__ == "__main__":
    main()
