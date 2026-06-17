#!/usr/bin/env python3
"""Attribute DIS/CON spans to the judge who authored them.

Two signals:
  * the **span heading** — DIS/CON spans begin with "Odlišné stanovisko
    soudce/soudkyně <Name>", with the name in the Czech *genitive* (declined),
    e.g. "soudce Jana Musila", "JUDr. Ivy Brožové";
  * the scraped **separate_opinion** metadata — the authoritative, clean
    *nominative* list of dissenting judges for the decision (from the record
    card; see scrape/).

Rule (see IMPLEMENTATION_PLAN §5 Stage 3):
  - 1 listed judge  -> every DIS/CON span is that judge's (unambiguous);
  - N listed judges -> match each span to the listed judge whose surname
    appears in the span heading;
  - 0 listed / metadata missing -> fall back to the raw name parsed from the
    heading (lossy: still genitive). This path dominates until the record-card
    scraper replaces the buggy legacy metadata.

The annotation schema requires one tag-pair per judge (ANOTACE_navod.md), so
multi-judge decisions already arrive as separate spans — the matcher only has
to assign each span to the right name, not split text.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

# Honorifics / academic titles to drop before reading a name.
_TITLES = r"(?:JUDr|Mgr|Ing|prof|doc|Dr|PhDr|CSc|DrSc|Ph\.?D|LL\.?M|MBA|et)\.?"

# Heading: "Odlišné stanovisko [soudce|soudkyně|...] <name phrase>" up to a
# delimiter (newline, "ve věci", "k výroku/odůvodnění/nálezu", "sp. zn.", "(").
_RE_HEADING = re.compile(
    r"(?i)(?:částečně\s+)?(?:odlišn[éá]|souhlasn[éá])\s+stanovisk[oa]\s+"
    r"(?:soudce|soudkyně|soudců)?\s*"
    r"(?P<name>.+?)"
    r"(?=\s*(?:\n|ve\s+věci|k\s+výrok|k\s+odůvodn|k\s+nález|sp\.\s*zn|\(|$))",
    re.DOTALL,
)


def _strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def heading_name(span_text: str) -> Optional[str]:
    """Best-effort raw name phrase from a DIS/CON span heading (still declined)."""
    head = span_text.lstrip()
    # A few decisions prefix a bare "Odlišná stanoviska" line; skip it.
    head = re.sub(r"(?i)^\s*odlišná\s+stanoviska\s*\n", "", head, count=1)
    m = _RE_HEADING.match(head)
    if not m:
        return None
    name = re.sub(_TITLES, "", m.group("name"), flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip(" .,")
    return name or None


def surname_key(name: str) -> str:
    """Diacritics-folded lowercase last token — a coarse, declension-tolerant key.

    Genitive surnames keep a recognisable stem (Musila/Musil, Ševčíka/Ševčík),
    so for *matching against a known nominative list* a prefix compare on this
    key is usually enough; it is NOT a clean author id on its own.
    """
    if not name:
        return ""
    toks = _strip_diacritics(name).lower().split()
    return toks[-1] if toks else ""


def _matches(meta_name: str, span_key: str) -> bool:
    """Does a nominative metadata name plausibly match a genitive span key?"""
    mkey = surname_key(meta_name)
    if not mkey or not span_key:
        return False
    short, long = sorted((mkey, span_key), key=len)
    # Genitive adds a suffix to the nominative stem -> prefix containment both ways.
    return long.startswith(short[: max(3, len(short) - 2)])


def attribute_spans(
    spans: List[dict],
    separate_opinion: Optional[List[str]] = None,
) -> List[Tuple[dict, str, str]]:
    """Return (span, author, source) for each DIS/CON span.

    ``source`` is "meta" when the author came from the scraped separate_opinion
    list, else "heading".
    """
    sep = [s for s in (separate_opinion or []) if s and s.lower() != "unknown"]
    out: List[Tuple[dict, str, str]] = []
    dc = [s for s in spans if s["label"] in ("DIS", "CON")]

    for span in dc:
        raw = heading_name(span["text"])
        if len(sep) == 1:
            out.append((span, sep[0], "meta"))
        elif len(sep) > 1:
            key = surname_key(raw or "")
            hit = next((m for m in sep if _matches(m, key)), None)
            out.append((span, hit or (raw or "UNKNOWN"), "meta" if hit else "heading"))
        else:
            out.append((span, raw or "UNKNOWN", "heading"))
    return out
