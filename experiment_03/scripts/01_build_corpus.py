#!/usr/bin/env python3
"""Stage 0 — bootstrap the committed corpus seed list (corpus/seed.tsv)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scrape.build_seed_list import main
if __name__ == "__main__":
    main()
