#!/usr/bin/env python3
"""
Parse NALUS decision text to extract metadata using regex patterns.
Based on variable_reference.md schema, extracting what's available from GetText.aspx text.
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class DecisionMetadata:
    """Metadata extracted from NALUS decision text."""
    
    # Core identifiers
    spis_zn: Optional[str] = None
    doc_id: Optional[str] = None  # ECLI - rarely in text, will be null
    
    # Dates
    date_decision: Optional[str] = None
    date_submission: Optional[str] = None
    
    # Decision form and proceedings
    type_decision: Optional[str] = None  # NÁLEZ, USNESENÍ, STANOVISKO PLÉNA
    type_proceedings: Optional[str] = None
    formation: Optional[str] = None  # e.g., "I. senát", "Plénum"
    
    # Judges
    judge_rapporteur_name: Optional[str] = None
    composition: List[str] = None  # Panel members (simplified list of names)
    
    # Parties
    applicant: List[str] = None
    concerned_body: List[str] = None
    
    # Verdicts
    type_verdict: List[str] = None  # Simplified: ["zamítnuto", "vyhověno", etc.]
    grounds: Optional[str] = None  # Computed from type_verdict
    
    # Disputed acts
    disputed_act: List[str] = None
    
    # Separate opinions
    separate_opinion: List[str] = None  # Names of dissenting judges (no grouping)
    
    # URL
    url_address: Optional[str] = None
    
    # Full text
    full_text: Optional[str] = None
    
    # Warnings for missing fields
    _warnings: List[str] = None
    
    def __post_init__(self):
        if self.composition is None:
            self.composition = []
        if self.applicant is None:
            self.applicant = []
        if self.concerned_body is None:
            self.concerned_body = []
        if self.type_verdict is None:
            self.type_verdict = []
        if self.disputed_act is None:
            self.disputed_act = []
        if self.separate_opinion is None:
            self.separate_opinion = []
        if self._warnings is None:
            self._warnings = []


def compute_grounds(type_verdict: List[str]) -> str:
    """
    Compute grounds from type_verdict using Paulík's hierarchical rule.
    merits > procedural > admissibility
    """
    if not type_verdict:
        return None
    
    verdict_str = " ".join(type_verdict).lower()
    
    if "vyhověno" in verdict_str or "zamítnuto" in verdict_str:
        return "merits"
    elif "procesní" in verdict_str:
        return "procedural"
    else:
        return "admissibility"


def parse_decision(text: str, spis_zn: str) -> DecisionMetadata:
    """
    Extract metadata from decision text using regex patterns.
    
    Args:
        text: Full decision text from GetText.aspx
        spis_zn: Spisová značka (file number) used to fetch the decision
    
    Returns:
        DecisionMetadata object with extracted fields
    """
    meta = DecisionMetadata(spis_zn=spis_zn)
    meta.full_text = text
    
    # 1. Spisová značka (case ID)
    spis_match = re.search(r'([IVXLCDM]+\.?\s*ÚS\s+\d+/\d+)', text)
    if spis_match:
        meta.spis_zn = spis_match.group(1)
    else:
        meta._warnings.append("Could not extract spisová značka from text")
    
    # 2. Decision type (NÁLEZ, USNESENÍ, STANOVISKO)
    form_match = re.search(r'(NÁLEZ|USNESENÍ|STANOVISKO\s+PLÉNA)', text, re.IGNORECASE)
    if form_match:
        meta.type_decision = form_match.group(1).upper()
    else:
        meta._warnings.append("Could not extract decision type (NÁLEZ/USNESENÍ)")
    
    # 3. Formation (panel)
    formation_match = re.search(r'([IVXLCDM]+\.?\s*senát|plénum)', text, re.IGNORECASE)
    if formation_match:
        meta.formation = formation_match.group(1).capitalize()
    
    # 4. Date decision
    date_match = re.search(r'ze dne\s+(\d{1,2}\.\s*\w+\s+\d{4})', text)
    if date_match:
        meta.date_decision = date_match.group(1)
    else:
        # Try alternative pattern
        date_match = re.search(r'(\d{1,2}\.\s*\w+\s+\d{4})', text)
        if date_match:
            meta.date_decision = date_match.group(1)
            meta._warnings.append("Date extracted but may not be decision date")
    
    # 5. Composition (panel members)
    # Pattern: "složeného z předsedkyně senátu X a soudců Y a Z"
    comp_match = re.search(r'složen[ýé] z\s+([^\.]+)', text, re.IGNORECASE)
    if comp_match:
        comp_text = comp_match.group(1)
        # Extract names - look for capitalized names with titles
        # Pattern: "JUDr. X", "prof. Y", "Mgr. Z", or just full names
        names = re.findall(r'(?:JUDr\.|prof\.|Mgr\.|Ing\.|Dr\.)?\s*([A-ZÁÉÍÓÚÝČĎŇŘŠŤŽ][a-záéíóúýčďňřšťž]+(?:\s+[A-ZÁÉÍÓÚÝČĎŇŘŠŤŽ][a-záéíóúýčďňřšťž]+)*)', comp_text)
        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)
        meta.composition = unique_names
    
    # 6. Applicant (from "ústavní stížnosti X proti")
    applicant_match = re.search(r'ústavní stížnosti\s+([^,\n]+?)\s+proti', text, re.IGNORECASE)
    if applicant_match:
        applicant = applicant_match.group(1).strip()
        meta.applicant = [applicant]
    
    # 7. Concerned body (from "proti usnesení X" or "proti rozsudku X")
    # Capture only the court name, not the full citation
    concerned_matches = re.findall(r'proti\s+(?:usnesení|rozsudku|platebnímu rozkazu|rozhodnutí)\s+([A-ZÁÉÍÓÚÝČĎŇŘŠŤŽ][^,\n]+?)(?:\s+č\. j\.|\s+ze dne)', text, re.IGNORECASE)
    if concerned_matches:
        # Clean up and deduplicate
        unique_bodies = list(set([b.strip() for b in concerned_matches]))
        meta.concerned_body = unique_bodies
    
    # 8. Type verdict (simplified)
    verdict_patterns = [
        (r'ústavní stížnost se zamítá', 'zamítnuto'),
        (r'ústavní stížnosti se vyhovuje', 'vyhověno'),
        (r'odmítá se', 'odmítnuto'),
        (r'zastavuje se', 'zastaveno'),
    ]
    for pattern, verdict in verdict_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            meta.type_verdict.append(verdict)
    
    if not meta.type_verdict:
        meta._warnings.append("Could not extract verdict type")
    
    # 9. Compute grounds
    meta.grounds = compute_grounds(meta.type_verdict)
    
    # 10. Disputed acts (challenged legal acts)
    # Look for patterns like "zrušení ustanovení § X zákona č. Y"
    disputed_matches = re.findall(r'zrušení\s+([^,\n]+?zákona[^,\n]*|[^,\n]+?vyhlášky[^,\n]*)', text, re.IGNORECASE)
    if disputed_matches:
        meta.disputed_act = disputed_matches
    
    # 11. Separate opinion (dissenting judges)
    dissent_match = re.search(r'odlišné stanovisko', text, re.IGNORECASE)
    if dissent_match:
        # Try to extract judge name from "Odlišné stanovisko soudkyně X"
        dissent_name_match = re.search(r'odlišné stanovisko\s+(?:soudkyně|soudce)\s+([A-ZÁÉÍÓÚÝČĎŇŘŠŤŽ][a-záéíóúýčďňřšťž]+(?:\s+[A-ZÁÉÍÓÚÝČĎŇŘŠŤŽ][a-záéíóúýčďňřšťž]+)*)', text, re.IGNORECASE)
        if dissent_name_match:
            meta.separate_opinion = [dissent_name_match.group(1)]
        else:
            meta.separate_opinion = ["unknown"]  # Flag that dissent exists but name not found
    
    # 12. URL (construct from spis_zn)
    if meta.spis_zn:
        # Convert spis_zn to URL-safe format
        url_safe = meta.spis_zn.replace(" ", "%20").replace("/", "%2F")
        meta.url_address = f"https://nalus.usoud.cz/Search/GetText.aspx?sz={url_safe}"
    
    return meta


def to_dict(meta: DecisionMetadata) -> Dict[str, Any]:
    """Convert DecisionMetadata to dictionary for JSON serialization."""
    d = asdict(meta)
    # Remove internal _warnings from output (or keep it as metadata)
    # Keeping it for transparency
    return d
