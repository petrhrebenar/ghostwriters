#!/usr/bin/env python3
"""Stage 04 — send prompts to OpenRouter and parse/validate the spans.

Reads ``data/03_prompts/<id>.json``, calls the OpenRouter chat-completions API
(default model: Claude Sonnet), extracts the JSON span list from the reply,
validates it against the line count, and writes ``data/04_responses/<id>.json``
with the raw reply, parsed+validated spans, token usage and any problems.

Requires the ``OPENROUTER_API_KEY`` environment variable.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python -m annotation.llm_annotate.run_llm           # all prompts, skip done
    python -m annotation.llm_annotate.run_llm --limit 3 # first 3 only
    python -m annotation.llm_annotate.run_llm --dry-run # validate prompts, no API
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests

from .common import (
    DIR_PROMPTS,
    DIR_RESPONSES,
    LABELS,
    read_json,
    write_json,
)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
RE_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


# ---------------------------------------------------------------------------
# Parsing + validation
# ---------------------------------------------------------------------------

def extract_json(content: str) -> Optional[Dict]:
    """Pull the first JSON object out of a model reply (tolerates fences/prose)."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.DOTALL)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = RE_JSON_OBJ.search(content)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def validate_spans(spans: List[Dict], n_lines: int) -> Tuple[List[Dict], List[str]]:
    """Return (clean_spans, problems). Drops malformed spans; flags overlaps."""
    problems: List[str] = []
    clean: List[Dict] = []
    for i, sp in enumerate(spans):
        label = sp.get("label")
        s, e = sp.get("start_line"), sp.get("end_line")
        if label not in LABELS:
            problems.append(f"span {i}: bad label {label!r}")
            continue
        if not isinstance(s, int) or not isinstance(e, int):
            problems.append(f"span {i} ({label}): non-integer line refs")
            continue
        if not (1 <= s <= e <= n_lines):
            problems.append(f"span {i} ({label}): out-of-range [{s},{e}] (n={n_lines})")
            continue
        clean.append({"label": label, "start_line": s, "end_line": e})

    clean.sort(key=lambda x: x["start_line"])
    for a, b in zip(clean, clean[1:]):
        if b["start_line"] <= a["end_line"]:
            problems.append(
                f"overlap: {a['label']}[{a['start_line']},{a['end_line']}] & "
                f"{b['label']}[{b['start_line']},{b['end_line']}]"
            )
    return clean, problems


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_openrouter(prompt: Dict, api_key: str, timeout: int = 120) -> Dict:
    payload = {
        "model": prompt["model"],
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        "temperature": prompt["params"].get("temperature", 0.0),
        "max_tokens": prompt["params"].get("max_tokens", 4096),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "ghostwriters-annotation",
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def process_prompt(prompt: Dict, api_key: str) -> Dict:
    data = call_openrouter(prompt, api_key)
    content = data["choices"][0]["message"]["content"]
    parsed = extract_json(content)
    if parsed is None:
        spans, problems = [], ["could not parse JSON from reply"]
    else:
        spans, problems = validate_spans(parsed.get("spans", []), prompt["n_lines"])
    return {
        "id": prompt["id"],
        "model": prompt["model"],
        "n_lines": prompt["n_lines"],
        "spans": spans,
        "problems": problems,
        "usage": data.get("usage", {}),
        "raw_content": content,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="Process only first N (0 = all)")
    ap.add_argument("--overwrite", action="store_true", help="Re-run prompts already done")
    ap.add_argument("--dry-run", action="store_true", help="Validate prompts; no API calls")
    ap.add_argument("--delay", type=float, default=0.0, help="Seconds between calls")
    args = ap.parse_args()

    files = sorted(DIR_PROMPTS.glob("*.json"))
    if not files:
        print(f"No prompts in {DIR_PROMPTS}. Run build_prompts first.", file=sys.stderr)
        sys.exit(1)
    if args.limit:
        files = files[: args.limit]

    if args.dry_run:
        total_chars = 0
        for fp in files:
            p = read_json(fp)
            total_chars += len(p["system"]) + len(p["user"])
        print(f"DRY RUN: {len(files)} prompts, ~{total_chars:,} chars "
              f"(~{total_chars // 4:,} tokens est.). No API calls made.")
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY in the environment.", file=sys.stderr)
        sys.exit(2)

    DIR_RESPONSES.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_err = n_problem = 0
    for i, fp in enumerate(files, 1):
        out_path = DIR_RESPONSES / fp.name
        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue
        prompt = read_json(fp)
        try:
            rec = process_prompt(prompt, api_key)
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"[{i}/{len(files)}] ERROR {prompt['id']}: {e}", file=sys.stderr)
            continue
        write_json(out_path, rec)
        n_ok += 1
        tag = ""
        if rec["problems"]:
            n_problem += 1
            tag = f"  [problems: {len(rec['problems'])}]"
        counts = {l: sum(1 for s in rec["spans"] if s["label"] == l) for l in LABELS}
        print(f"[{i}/{len(files)}] {prompt['id']}: "
              + ", ".join(f"{l}={counts[l]}" for l in LABELS) + tag)
        if args.delay:
            time.sleep(args.delay)

    print("=" * 60)
    print(f"Done: {n_ok} called, {n_skip} skipped, {n_err} errors, "
          f"{n_problem} with validation problems -> {DIR_RESPONSES}")


if __name__ == "__main__":
    main()
