#!/usr/bin/env python3
"""Stage 3b — join spans + scraped metadata -> data/dataset/{decisions,dissents}."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_dataset
if __name__ == "__main__":
    build_dataset.main()
