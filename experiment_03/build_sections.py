#!/usr/bin/env python3
"""Per-section table for the section x author "decision cards".

One row per (decision, label): the concatenated text of every span with that
label, in document order, plus length + the scraped metadata needed to annotate
the card (rapporteur, separate_opinion). Distinct from build_dataset.py, which
emits only the RATIO scoring target and the DIS/CON training rows; here we keep
*all* nine labels so a card can show how each section attributes.

Input : data/07_spans/<id>.json (+ data/01_scraped/<id>.json metadata)
Output: data/dataset/sections.{csv,parquet}

Usage:  python build_sections.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
DIR_SPANS = EXP_DIR / "data" / "07_spans"
DIR_SCRAPED = EXP_DIR / "data" / "01_scraped"
DIR_OUT = EXP_DIR / "data" / "dataset"

LABELS = ["HEAD", "REC", "PART", "PROC", "FACT", "RATIO", "DISP", "DIS", "CON"]
LABEL_ORDER = {l: i for i, l in enumerate(LABELS)}
META_FIELDS = ["spis_zn", "date_decision", "formation", "type_decision",
               "judge_rapporteur_name", "separate_opinion"]


def _meta(doc_id: str) -> Dict:
    fp = DIR_SCRAPED / f"{doc_id}.json"
    if not fp.exists():
        return {}
    raw = json.loads(fp.read_text(encoding="utf-8"))
    return {k: raw.get(k) for k in META_FIELDS}


def build(spans_dir: Path = DIR_SPANS, out_dir: Path = DIR_OUT) -> None:
    rows: List[Dict] = []
    for fp in sorted(spans_dir.glob("*.json")):
        if fp.name == "annotated.jsonl":
            continue
        rec = json.loads(fp.read_text(encoding="utf-8"))
        doc_id = rec["id"]
        meta = _meta(doc_id)
        sep = meta.get("separate_opinion") or []
        # group spans by label, preserving document order
        by_label: Dict[str, List[str]] = {}
        for s in rec.get("spans", []):
            by_label.setdefault(s["label"], []).append(s["text"])
        for label, texts in by_label.items():
            text = "\n\n".join(texts)
            rows.append({
                "doc_id": doc_id,
                "label": label,
                "label_order": LABEL_ORDER.get(label, 99),
                "n_spans": len(texts),
                "words": len(text.split()),
                "judge_rapporteur_name": meta.get("judge_rapporteur_name"),
                "separate_opinion": "; ".join(sep) if isinstance(sep, list) else sep,
                "formation": meta.get("formation"),
                "text": text,
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["doc_id", "label_order"]).reset_index(drop=True)
    df.to_csv(out_dir / "sections.csv", index=False)
    try:
        df.to_parquet(out_dir / "sections.parquet", index=False)
    except Exception as e:
        print(f"  (parquet skipped: {type(e).__name__}; CSV written)")

    n_docs = df["doc_id"].nunique()
    print(f"sections: {len(df)} rows across {n_docs} decisions")
    print("  by label: " + ", ".join(
        f"{l}={int((df['label'] == l).sum())}" for l in LABELS if (df['label'] == l).any()))
    print(f"  median words/section: {int(df['words'].median())}")
    print(f"-> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spans", default=str(DIR_SPANS))
    ap.add_argument("--out", default=str(DIR_OUT))
    args = ap.parse_args()
    build(Path(args.spans), Path(args.out))


if __name__ == "__main__":
    main()
