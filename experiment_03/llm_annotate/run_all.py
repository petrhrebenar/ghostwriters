#!/usr/bin/env python3
"""Run the whole LLM annotation pipeline end to end.

Stages: ingest -> clean -> build_prompts -> run_llm -> render.
Each stage is also runnable on its own (see the module docstrings).

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python -m annotation.llm_annotate.run_all --limit 5
    python -m annotation.llm_annotate.run_all --model anthropic/claude-sonnet-4.6
"""

from __future__ import annotations

import argparse
import sys

from . import build_prompts, clean, ingest, render, run_llm
from .build_prompts import DEFAULT_MODEL


def _run(module, argv):
    saved = sys.argv
    sys.argv = [module.__name__] + argv
    try:
        module.main()
    finally:
        sys.argv = saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="Stop before any API calls")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-ingest", action="store_true", help="01_scraped already populated")
    args = ap.parse_args()

    lim = ["--limit", str(args.limit)] if args.limit else []

    if not args.skip_ingest:
        _run(ingest, lim)
    _run(clean, lim)
    _run(build_prompts, ["--model", args.model] + lim)

    if args.dry_run:
        _run(run_llm, ["--dry-run"] + lim)
        print("\nDry run complete. Re-run without --dry-run to call the API.")
        return

    llm_argv = lim + (["--overwrite"] if args.overwrite else [])
    _run(run_llm, llm_argv)
    _run(render, lim)
    print("\nPipeline complete. Review data/05_tagged/*.html, edit *.txt as needed.")


if __name__ == "__main__":
    main()
