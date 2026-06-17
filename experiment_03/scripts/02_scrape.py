#!/usr/bin/env python3
"""Stage 1 — scrape NALUS record cards + bodies -> data/01_scraped/.

Needs the .venv with selenium + Firefox/geckodriver. Pass-through to
scrape.pipeline (e.g. --limit, --overwrite, --only). Resumable.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scrape.pipeline import main
if __name__ == "__main__":
    main()
