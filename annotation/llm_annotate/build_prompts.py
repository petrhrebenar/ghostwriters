#!/usr/bin/env python3
"""Stage 03 — build the exact LLM prompt for each cleaned decision.

Writes ``data/03_prompts/<id>.json`` capturing everything needed to reproduce
the call: model id, params, system message, and the fully-rendered user
message (rules + metadata + numbered text). The actual API call happens in
stage 04 (run_llm.py), which consumes these files verbatim.

Usage:
    python -m annotation.llm_annotate.build_prompts --model anthropic/claude-sonnet-4.6
"""

from __future__ import annotations

import argparse
import sys

from .common import DIR_CLEANED, DIR_PROMPTS, read_json, write_json
from .prompts import SYSTEM_MSG, build_user_message

DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help="Build only first N (0 = all)")
    args = ap.parse_args()

    files = sorted(DIR_CLEANED.glob("*.json"))
    if not files:
        print(f"No cleaned files in {DIR_CLEANED}. Run the clean stage first.", file=sys.stderr)
        sys.exit(1)
    if args.limit:
        files = files[: args.limit]

    DIR_PROMPTS.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in files:
        rec = read_json(fp)
        if not rec.get("lines"):
            continue
        prompt = {
            "id": rec["id"],
            "model": args.model,
            "params": {"temperature": args.temperature, "max_tokens": args.max_tokens},
            "n_lines": len(rec["lines"]),
            "system": SYSTEM_MSG,
            "user": build_user_message(rec["meta"], rec["lines"]),
        }
        write_json(DIR_PROMPTS / f"{rec['id']}.json", prompt)
        n += 1

    print(f"Built {n} prompts (model={args.model}) -> {DIR_PROMPTS}")


if __name__ == "__main__":
    main()
