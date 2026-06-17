#!/usr/bin/env python3
"""
Clean scraped Constitutional Court decisions into readable plain text for
tag-based annotation.

Each decision JSON (with raw-HTML ``full_text``) is converted to a UTF-8 text
file with a short metadata header and the decision body split into readable
paragraphs (blank-line separated). NO annotation tags are inserted - the
annotator adds <RATIO>/<DIS>/<CON> tags by hand (see ANOTACE_navod.md).

Usage (run inside the nalus_v2 poetry env so beautifulsoup4 is available):
    poetry run python /abs/path/annotation/prepare_for_tagging.py \
        --input  scrapers/nalus_v2/data/decisions \
        --output annotation/decisions_to_annotate
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup

# A numbered paragraph marker at line start: "1. ", "14. ", "1) ", "[1]", "[5.]".
RE_NUMBERED = re.compile(r"^\[?\s*(\d{1,3})\s*[.\)\]]+\s")
# A roman-numeral section heading at line start, e.g. "I. Skutkové okolnosti".
RE_ROMAN_HEAD = re.compile(r"^[IVXLCDM]{1,5}\.\s+\S")
# A line that is only a roman numeral, e.g. "I." or "II" (standalone heading).
RE_ROMAN_ONLY = re.compile(r"^[IVXLCDM]{1,5}\.?$")
# Common starts of citation/continuation lines that belong to the previous unit.
RE_CONT_START = re.compile(r"^(sp\.\s*zn|č\.\s*j|čj\.|odst\.|písm\.|viz\b)", re.I)
TERMINAL = (".", "!", "?", ":", ")", "]", '"', "“", "”")


def _despace(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def clean_lines(full_text: str) -> List[str]:
    """Body lines (blank lines preserved as '') extracted from td.DocContent."""
    soup = BeautifulSoup(full_text, "html.parser")
    node = soup.find("td", class_="DocContent") or soup.find("body") or soup
    raw = node.get_text(separator="\n")
    return [ln.strip() for ln in raw.split("\n")]


def _is_heading(line: str) -> bool:
    d = _despace(line)
    if d.startswith("odůvodnění") or d.startswith("oduvodneni"):
        return True
    if RE_ROMAN_ONLY.match(line):
        return True
    if RE_ROMAN_HEAD.match(line) and len(line) < 150 and not line.endswith("."):
        return True
    return False


def _is_wrapped(lines: List[str]) -> bool:
    """Old (pre-~2007) decisions are hard-wrapped: almost no line exceeds 100 chars."""
    L = [len(l) for l in lines if l]
    if not L:
        return False
    return sum(1 for x in L if x > 100) / len(L) < 0.10


def paragraphs(full_text: str) -> List[str]:
    """Reconstruct readable paragraphs from the cleaned decision text."""
    lines = clean_lines(full_text)
    return _paras_wrapped(lines) if _is_wrapped(lines) else _paras_flow(lines)


def _paras_flow(lines: List[str]) -> List[str]:
    """Modern numbered docs + early-2000s prose (paragraph-per-line)."""
    out: List[str] = []
    buf: List[str] = []
    expected = 1

    def flush():
        nonlocal buf
        if buf:
            out.append(_norm(" ".join(buf)))
            buf = []

    for l in lines:
        if not l:
            flush()
            continue
        if _is_heading(l):
            flush()
            out.append(_norm(l))
            continue
        m = RE_NUMBERED.match(l)
        if m and expected <= int(m.group(1)) <= expected + 1:
            flush()
            buf = [l]
            expected = int(m.group(1)) + 1
            continue
        first = l[0]
        is_cont = (not first.isalpha()) or first.islower() or len(l) < 3 or bool(RE_CONT_START.match(l))
        prev_terminal = bool(buf) and buf[-1].rstrip().endswith(TERMINAL)
        if buf and (is_cont or not prev_terminal):
            buf.append(l)
        else:
            flush()
            buf = [l]
    flush()
    return [p for p in out if p.strip()]


def _paras_wrapped(lines: List[str]) -> List[str]:
    """Hard-wrapped docs: paragraphs are blank-line separated; join soft wraps."""
    out: List[str] = []
    buf: List[str] = []

    def flush():
        nonlocal buf
        if buf:
            out.append(_norm(" ".join(buf)))
            buf = []

    for l in lines:
        if not l:
            flush()
        else:
            buf.append(l)
    flush()
    return [p for p in out if p.strip()]


HEADER = """\
# spisová značka: {spis_zn}
# typ rozhodnutí:  {type_decision}
# datum:           {date_decision}
# zdroj:           {url}
#
# Anotace: do textu níže vložte značky <RATIO>…</RATIO> a <DIS>…</DIS> / <CON>…</CON>.
# Postup a pravidla viz ANOTACE_navod.md. Text (kromě vkládaných značek) neměňte.
# ============================================================================
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Directory of decision JSON files")
    ap.add_argument("--output", required=True, help="Directory to write clean text files")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.json"))
    if not files:
        print(f"No JSON files found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    n_ok = n_err = 0
    for fp in files:
        try:
            meta = json.loads(fp.read_text(encoding="utf-8"))
            full_text = meta.get("full_text")
            if not full_text:
                print(f"  SKIP {fp.name}: no full_text", file=sys.stderr)
                n_err += 1
                continue
            paras = paragraphs(full_text)
            header = HEADER.format(
                spis_zn=meta.get("spis_zn") or "?",
                type_decision=meta.get("type_decision") or "?",
                date_decision=meta.get("date_decision") or "?",
                url=meta.get("url_address") or "?",
            )
            body = "\n\n".join(paras)
            (out_dir / f"{fp.stem}.txt").write_text(header + "\n" + body + "\n", encoding="utf-8")
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {fp.name}: {e}", file=sys.stderr)
            n_err += 1

    print("=" * 60)
    print(f"Clean text files written: {n_ok}  (errors: {n_err})")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
