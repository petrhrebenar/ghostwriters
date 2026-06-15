#!/usr/bin/env python3
"""
Extract <RATIO>/<DIS>/<CON> spans from annotated decision text files.

Validates that tags are balanced and not nested, then emits:
  - <output>/<id>.json     spans for one decision
  - <output>/annotated.jsonl   all decisions, one JSON per line

Each span record: {"label": "RATIO|DIS|CON", "text": "..."}.
The author of a DIS/CON is usually the first line of its span
("Odlišné stanovisko soudce ...") and can be parsed downstream.

Usage:
    python annotation/extract_tags.py \
        --input  annotation/decisions_to_annotate \
        --output annotation/annotated
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

LABELS = ["HEAD", "REC", "PART", "PROC", "FACT", "RATIO", "DISP", "DIS", "CON"]
_LBL = "|".join(LABELS)
RE_SPAN = re.compile(rf"<({_LBL})>(.*?)</\1>", re.DOTALL)
RE_ANY_TAG = re.compile(rf"</?({_LBL})>")


def _strip_comments(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.startswith("#"))


def parse_file(path: Path) -> Dict:
    text = _strip_comments(path.read_text(encoding="utf-8"))
    spans: List[Dict] = []
    for m in RE_SPAN.finditer(text):
        spans.append({"label": m.group(1), "text": _norm(m.group(2))})

    # Validate: number of opening/closing tags must match what we captured.
    n_tags = len(RE_ANY_TAG.findall(text))
    problems: List[str] = []
    if n_tags != 2 * len(spans):
        problems.append(f"unbalanced/nested tags ({n_tags} tags, {len(spans)} spans matched)")
    return {"id": path.stem, "spans": spans, "_problems": problems}


def _norm(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Directory of annotated text files")
    ap.add_argument("--output", required=True, help="Directory for parsed output")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.txt"))
    if not files:
        print(f"No text files found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    records: List[Dict] = []
    counts = {l: 0 for l in LABELS}
    n_tagged = n_problem = 0
    for fp in files:
        rec = parse_file(fp)
        problems = rec.pop("_problems")
        if problems:
            n_problem += 1
            print(f"  PROBLEM {fp.name}: {'; '.join(problems)}")
        if rec["spans"]:
            n_tagged += 1
            for s in rec["spans"]:
                counts[s["label"]] += 1
        (out_dir / f"{fp.stem}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records.append(rec)

    with (out_dir / "annotated.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            if rec["spans"]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"Files: {len(files)}  |  with spans: {n_tagged}  |  with tag problems: {n_problem}")
    print(f"Spans: " + ", ".join(f"{l}={counts[l]}" for l in LABELS))
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
