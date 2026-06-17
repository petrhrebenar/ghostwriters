#!/usr/bin/env python3
"""Parse a NALUS ResultDetail record card (``table.recordCardTable``) into a
normalized metadata dict. **Transport-agnostic**: input is card HTML, output is
a dict — no network here, so the fetch layer (Selenium today, requests/Playwright
tomorrow) can change without touching this module.

Mirrors the field mapping of the reference ``stepanpaulik/ccc_dataset``
(``scripts/ccc_web_scraping.R``): the card is a two-column label→value table;
names are stored surname-first and reordered; the GetText file id comes from the
"URL adresa" cell; formation is derived from the ECLI prefix.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup

# NALUS card label (Czech) -> our field name. Only the fields we use downstream
# plus a few nice-to-haves; unmapped labels are ignored.
LABEL_MAP = {
    "Identifikátor evropské judikatury": "doc_id",        # ECLI
    "Spisová značka": "spis_zn",
    "Populární název": "popular_name",
    "Datum rozhodnutí": "date_decision",
    "Datum vyhlášení": "date_publication",
    "Datum podání": "date_submission",
    "Forma rozhodnutí": "type_decision",
    "Typ řízení": "type_proceedings",
    "Význam": "importance",
    "Navrhovatel": "applicant",
    "Dotčený orgán": "concerned_body",
    "Soudce zpravodaj": "judge_rapporteur_name",
    "Napadený akt": "disputed_act",
    "Typ výroku": "type_verdict",
    "Odlišné stanovisko": "separate_opinion",
    "Předmět řízení": "subject_proceedings",
    "Věcný rejstřík": "subject_register",
    "URL adresa": "url_address",
}

# Fields that are newline-separated multi-value cells -> lists.
MULTIVALUE = {"applicant", "concerned_body", "separate_opinion", "disputed_act",
              "type_verdict", "subject_proceedings", "subject_register"}

# Fields that hold a surname-first personal name -> reorder to "First Last".
NAME_FIELDS = {"judge_rapporteur_name", "separate_opinion"}

_MONTHS = None  # dates already come as "8. 8. 2006" (numeric) — no month names


def card_label_value(html: str) -> Dict[str, str]:
    """Raw label -> value map from ``table.recordCardTable`` (values keep newlines)."""
    soup = BeautifulSoup(html, "html.parser")
    tbl = soup.select_one(".recordCardTable")
    out: Dict[str, str] = {}
    if tbl is None:
        return out
    for tr in tbl.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) >= 2:
            label = tds[0].get_text(" ", strip=True)
            if label:
                out[label] = tds[1].get_text("\n", strip=True)
    return out


def reorder_name(name: str) -> str:
    """"Güttler Vojen" -> "Vojen Güttler" (surname-first -> given-first).

    NALUS stores names surname-first. The reference takes word(2) word(1); we
    swap the first two tokens and keep any remainder (rare multi-part names).
    """
    parts = name.split()
    if len(parts) < 2:
        return name.strip()
    return " ".join([parts[1], parts[0], *parts[2:]]).strip()


def parse_czech_date(value: str) -> Optional[str]:
    """"8. 8. 2006" -> "2006-08-08" (ISO). Returns None if unparseable."""
    m = re.search(r"(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})", value)
    if not m:
        return None
    d, mo, y = (int(x) for x in m.groups())
    return f"{y:04d}-{mo:02d}-{d:02d}"


def formation_from_ecli(ecli: str) -> Optional[str]:
    """Senate / plenum from the ECLI prefix (e.g. ':Pl.US.', ':1.US.')."""
    m = re.search(r":([0-9A-Za-z]+)\.US\.", ecli or "")
    if not m:
        return None
    code = m.group(1)
    if code == "Pl":
        return "Plénum"
    roman = {"1": "I. senát", "2": "II. senát", "3": "III. senát", "4": "IV. senát"}
    return roman.get(code, code)


def file_id_from_url(url: str) -> Optional[str]:
    """Extract the GetText ``sz`` file id from the card's URL adresa cell."""
    if not url:
        return None
    qs = parse_qs(urlparse(url).query)
    sz = qs.get("sz")
    return sz[0] if sz else None


def _grounds(type_verdict: List[str]) -> Optional[str]:
    s = " ".join(type_verdict).lower()
    if not s:
        return None
    if "vyhověno" in s or "zamítnuto" in s:
        return "merits"
    if "procesní" in s:
        return "procedural"
    return "admissibility"


def normalize_card(html: str) -> Dict:
    """Card HTML -> normalized metadata dict (no ``full_text``; added by fetch)."""
    raw = card_label_value(html)
    meta: Dict = {"_warnings": []}

    for label, value in raw.items():
        field = LABEL_MAP.get(label)
        if not field:
            continue
        if field in MULTIVALUE:
            vals = [v.strip() for v in value.split("\n") if v.strip()]
            if field in NAME_FIELDS:
                vals = [reorder_name(v) for v in vals]
            meta[field] = vals
        elif field in NAME_FIELDS:
            meta[field] = reorder_name(value) if value else None
        elif field.startswith("date_"):
            meta[field] = parse_czech_date(value)
        else:
            meta[field] = value or None

    # Derived / cross-checked fields.
    meta["formation"] = formation_from_ecli(meta.get("doc_id", ""))
    meta["file_id"] = file_id_from_url(meta.get("url_address", ""))
    meta["grounds"] = _grounds(meta.get("type_verdict") or [])

    if not meta.get("judge_rapporteur_name"):
        meta["_warnings"].append("missing judge_rapporteur_name")
    if not meta.get("file_id"):
        meta["_warnings"].append("could not derive file_id from URL adresa")
    return meta
