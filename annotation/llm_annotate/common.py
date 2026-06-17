"""Shared helpers and path config for the LLM annotation pipeline.

Stages (see README.md):
    01_scraped   <id>.json            scraper output (full_text + metadata)
    02_cleaned   <id>.json/.txt/.html HTML stripped, lines numbered
    03_prompts   <id>.json            exact {model, system, user, params} sent
    04_responses <id>.json            raw LLM reply + parsed spans + usage
    05_tagged    <id>.txt/.html       <RATIO>/<DIS>/<CON> tags + colour preview
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PKG_DIR = Path(__file__).resolve().parent
ANNOTATION_DIR = PKG_DIR.parent
DATA_DIR = ANNOTATION_DIR / "data"

DIR_SCRAPED = DATA_DIR / "01_scraped"
DIR_CLEANED = DATA_DIR / "02_cleaned"
DIR_PROMPTS = DATA_DIR / "03_prompts"
DIR_RESPONSES = DATA_DIR / "04_responses"
DIR_TAGGED = DATA_DIR / "05_tagged"

# Default location of the nalus_v2 scraper output (used by the ingest stage).
DEFAULT_SCRAPER_DIR = ANNOTATION_DIR.parent / "scrapers" / "nalus_v2" / "data" / "decisions"

# NALUS source URL. The working query param is the *file id* (the JSON
# filename / record id, e.g. "1-1056-07_2"), NOT the human-readable spisová
# značka that the scraper's parse.py wrote into url_address.
NALUS_BASE_URL = "https://nalus.usoud.cz/Search/GetText.aspx"


def decision_url(rec_id: str) -> str:
    """Working NALUS URL for a decision, built from its file id."""
    return f"{NALUS_BASE_URL}?sz={quote(rec_id, safe='')}"

# Full 9-category structural scheme (see Poznámky/anotacni_schema_US.md).
# Listed in canonical document order.
LABELS = ["HEAD", "REC", "PART", "PROC", "FACT", "RATIO", "DISP", "DIS", "CON"]

LABEL_COLORS = {
    "HEAD": "#7f7f7f",   # grey   - header
    "REC": "#ff7f0e",    # orange - recapitulation of petition (applicant)
    "PART": "#8c564b",   # brown  - other parties' statements
    "PROC": "#9467bd",   # purple - procedural prerequisites
    "FACT": "#17becf",   # cyan   - factual background (court, neutral)
    "RATIO": "#2c7fb8",  # blue   - ratio decidendi (primary target)
    "DISP": "#bcbd22",   # olive  - disposition (verdict)
    "DIS": "#d62728",    # red    - dissenting opinion
    "CON": "#2ca02c",    # green  - concurring opinion
}

# Short Czech glosses for the HTML legend.
LABEL_DESC = {
    "HEAD": "záhlaví",
    "REC": "rekapitulace stížnosti (stěžovatel)",
    "PART": "vyjádření ostatních účastníků",
    "PROC": "procesní předpoklady",
    "FACT": "skutkové/procesní pozadí (řeč soudu)",
    "RATIO": "vlastní odůvodnění soudu",
    "DISP": "výrok",
    "DIS": "odlišné (disentní) stanovisko",
    "CON": "souhlasné (konkurující) stanovisko",
}

# Metadata fields copied from the scraped JSON into the cleaned artifact.
META_FIELDS = [
    "spis_zn",
    "type_decision",
    "date_decision",
    "formation",
    "judge_rapporteur_name",
    "separate_opinion",
    "url_address",
]


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------

def read_json(path: Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def ensure_dirs() -> None:
    for d in (DIR_SCRAPED, DIR_CLEANED, DIR_PROMPTS, DIR_RESPONSES, DIR_TAGGED):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Line numbering / rendering
# ---------------------------------------------------------------------------

def numbered_text(lines: List[str], sep: str = ": ") -> str:
    """Render lines as ``"<n>: <text>"`` (1-based) for prompts/preview."""
    width = len(str(len(lines)))
    return "\n".join(f"{i:>{width}}{sep}{ln}" for i, ln in enumerate(lines, 1))


def meta_header_lines(meta: Dict) -> List[str]:
    """Human-readable comment header (``#`` lines) for cleaned/tagged .txt."""
    return [
        f"# spisová značka: {meta.get('spis_zn') or '?'}",
        f"# typ rozhodnutí:  {meta.get('type_decision') or '?'}",
        f"# datum:           {meta.get('date_decision') or '?'}",
        f"# soudce zpravodaj:{meta.get('judge_rapporteur_name') or '?'}",
        f"# odlišná stan.:   {', '.join(meta.get('separate_opinion') or []) or '-'}",
        f"# zdroj:           {meta.get('url_address') or '?'}",
        "# " + "=" * 74,
    ]


def esc(s: str) -> str:
    return html.escape(s, quote=True)
