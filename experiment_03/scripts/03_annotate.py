#!/usr/bin/env python3
"""Stage 2 — LLM 9-category annotation (clean -> prompts -> LLM -> render).

The scraper writes straight to data/01_scraped/, so ingest is skipped. Resumable:
run_llm skips decisions already in data/04_responses/, so this only annotates the
73 new plenary decisions. Needs OPENROUTER_API_KEY.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_annotate import run_all
if __name__ == "__main__":
    if "--skip-ingest" not in sys.argv:
        sys.argv.append("--skip-ingest")
    run_all.main()
