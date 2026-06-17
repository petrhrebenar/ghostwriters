#!/usr/bin/env python3
"""Stage 3a — parse tagged decisions (data/05_tagged) into spans (data/07_spans)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import extract_tags
if __name__ == "__main__":
    sys.argv = ["extract_tags", "--input", str(ROOT / "data" / "05_tagged"),
                "--output", str(ROOT / "data" / "07_spans")]
    extract_tags.main()
