#!/usr/bin/env python3
"""Stage 05 — render LLM spans into reviewable tagged artifacts.

Combines ``data/02_cleaned/<id>.json`` (lines) with
``data/04_responses/<id>.json`` (spans) into ``data/05_tagged/``:

  <id>.txt   metadata header + body with <RATIO>/<DIS>/<CON> tags inserted at
             line boundaries. Drop-in input for annotation/extract_tags.py.
  <id>.html  same content, spans highlighted in colour with author + line range
             badges, for fast human review.

The human reviewer edits the .txt (move a tag up/down a line, fix an author),
then re-runs extract_tags.py to get final structured spans.

Usage:
    python -m annotation.llm_annotate.render
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List

from .common import (
    DIR_CLEANED,
    DIR_RESPONSES,
    DIR_TAGGED,
    LABEL_COLORS,
    LABEL_DESC,
    esc,
    meta_header_lines,
    read_json,
)


def build_tagged_txt(meta: Dict, lines: List[str], spans: List[Dict]) -> str:
    """Insert bare <LABEL>…</LABEL> tags around the line ranges.

    Tags are deliberately attribute-free so the output is a drop-in for
    annotation/extract_tags.py (whose regex only matches bare tags). The
    DIS/CON author stays inside the span (its heading line) and in the
    response JSON; it is not duplicated as a tag attribute.
    """
    opens: Dict[int, Dict] = {sp["start_line"]: sp for sp in spans}
    closes: Dict[int, Dict] = {sp["end_line"]: sp for sp in spans}
    out: List[str] = list(meta_header_lines(meta)) + [""]
    for i, ln in enumerate(lines, 1):
        if i in opens:
            out.append(f'<{opens[i]["label"]}>')
        out.append(ln)
        if i in closes:
            out.append(f'</{closes[i]["label"]}>')
    return "\n".join(out) + "\n"


def build_html(rec_id: str, meta: Dict, lines: List[str], spans: List[Dict], problems: List[str]) -> str:
    open_map = {sp["start_line"]: sp for sp in spans}
    close_set = {sp["end_line"] for sp in spans}
    in_span: Dict = {}
    rows: List[str] = []
    for i, ln in enumerate(lines, 1):
        if i in open_map:
            in_span = open_map[i]
            color = LABEL_COLORS[in_span["label"]]
            rows.append(
                f'<tr class="open" style="border-top:3px solid {color}">'
                f'<td class="ln">{i}</td>'
                f'<td><span class="badge" style="background:{color}">'
                f'{in_span["label"]} [{in_span["start_line"]}–{in_span["end_line"]}]'
                f'</span></td></tr>'
            )
        bg = f"background:{LABEL_COLORS[in_span['label']]}1a;" if in_span else ""
        rows.append(f'<tr style="{bg}"><td class="ln">{i}</td><td>{esc(ln)}</td></tr>')
        if i in close_set:
            in_span = {}

    header = "<br>".join(esc(h.lstrip("# ").rstrip()) for h in meta_header_lines(meta)[:-1])
    url = meta.get("url_address") or ""
    if url:
        header = header.replace(esc(url), f'<a href="{esc(url)}" target="_blank">{esc(url)}</a>')
    legend = " ".join(
        f'<span class="badge" style="background:{c}" title="{esc(LABEL_DESC.get(lbl, ""))}">{lbl}</span>'
        for lbl, c in LABEL_COLORS.items()
    )
    prob_html = ""
    if problems:
        items = "".join(f"<li>{esc(p)}</li>" for p in problems)
        prob_html = f'<div class="problems"><b>Validation problems:</b><ul>{items}</ul></div>'

    return f"""<!doctype html>
<html lang="cs"><head><meta charset="utf-8">
<title>{esc(rec_id)} — tagged</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.5; }}
 .meta {{ background:#f4f4f4; padding:1rem; border-radius:6px; font-size:.9rem; }}
 .problems {{ background:#fff3cd; padding:.6rem 1rem; border-radius:6px; margin-top:1rem; font-size:.85rem; }}
 .badge {{ color:#fff; padding:.1rem .45rem; border-radius:4px; font-size:.8rem; font-weight:600; }}
 table {{ border-collapse: collapse; margin-top:1rem; width:100%; }}
 td {{ vertical-align: top; padding:.15rem .6rem; }}
 td.ln {{ color:#999; text-align:right; user-select:none; font-variant-numeric: tabular-nums; width:3rem; }}
</style></head><body>
<h2>{esc(rec_id)}</h2>
<div class="meta">{header}<br><br>Legenda: {legend}</div>
{prob_html}
<table>{''.join(rows)}</table>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="Render only first N (0 = all)")
    args = ap.parse_args()

    files = sorted(DIR_RESPONSES.glob("*.json"))
    if not files:
        print(f"No responses in {DIR_RESPONSES}. Run run_llm first.", file=sys.stderr)
        sys.exit(1)
    if args.limit:
        files = files[: args.limit]

    DIR_TAGGED.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in files:
        resp = read_json(fp)
        cleaned = read_json(DIR_CLEANED / fp.name)
        meta, lines = cleaned["meta"], cleaned["lines"]
        spans = resp["spans"]
        (DIR_TAGGED / f"{resp['id']}.txt").write_text(
            build_tagged_txt(meta, lines, spans), encoding="utf-8"
        )
        (DIR_TAGGED / f"{resp['id']}.html").write_text(
            build_html(resp["id"], meta, lines, spans, resp.get("problems", [])),
            encoding="utf-8",
        )
        n += 1

    print(f"Rendered {n} tagged decisions -> {DIR_TAGGED}")


if __name__ == "__main__":
    main()
