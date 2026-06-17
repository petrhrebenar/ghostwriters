"""Morphological feature extraction from UDPipe output.

Extracts features from the XPOS and Feats fields of tokens:
- Morphological category distributions (Case, Number, Gender, Tense, Voice, Mood)
- XPOS bigrams (finer-grained than UPOS bigrams)
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from fingerprint.preprocessing import Document

# Morphological categories to extract from the Feats field.
# Each category maps to its known values in Czech (UD Czech-PDT).
MORPH_CATEGORIES: dict[str, list[str]] = {
    "Case": ["Nom", "Gen", "Dat", "Acc", "Voc", "Loc", "Ins"],
    "Number": ["Sing", "Plur", "Dual"],
    "Gender": ["Masc", "Fem", "Neut"],
    "Tense": ["Past", "Pres", "Fut"],
    "Voice": ["Act", "Pass"],
    "Mood": ["Ind", "Imp", "Cnd"],
    "Degree": ["Pos", "Cmp", "Sup"],
    "Polarity": ["Pos", "Neg"],
}


def _parse_feats(feats_str: str) -> dict[str, str]:
    """Parse a UDPipe Feats string like 'Case=Nom|Gender=Masc|Number=Sing'.

    Returns
    -------
    dict[str, str]
        Mapping from category to value, e.g. {'Case': 'Nom', 'Gender': 'Masc'}.
    """
    if not feats_str or feats_str == "_":
        return {}
    result = {}
    for pair in feats_str.split("|"):
        if "=" in pair:
            key, val = pair.split("=", 1)
            result[key] = val
    return result


def morphological_category_features(doc: Document) -> dict[str, float]:
    """Compute distributions over morphological categories.

    For each category (Case, Number, etc.), computes the relative frequency of
    each value among tokens that carry that category.

    Returns
    -------
    dict[str, float]
        Feature names like 'morph_Case_Nom', 'morph_Voice_Pass', etc.
    """
    # Count occurrences of each (category, value) pair
    category_counts: dict[str, Counter] = {
        cat: Counter() for cat in MORPH_CATEGORIES
    }

    for token in doc.all_tokens:
        parsed = _parse_feats(token.feats)
        for cat in MORPH_CATEGORIES:
            if cat in parsed:
                category_counts[cat][parsed[cat]] += 1

    features: dict[str, float] = {}
    for cat, known_values in MORPH_CATEGORIES.items():
        counts = category_counts[cat]
        total = sum(counts.values())
        for val in known_values:
            if total > 0:
                features[f"morph_{cat}_{val}"] = counts.get(val, 0) / total
            else:
                features[f"morph_{cat}_{val}"] = 0.0

    return features


def passive_voice_ratio(doc: Document) -> dict[str, float]:
    """Compute the ratio of passive to active voice among verbs.

    Returns
    -------
    dict[str, float]
        Single feature: 'passive_ratio'.
    """
    active = 0
    passive = 0
    for token in doc.all_tokens:
        parsed = _parse_feats(token.feats)
        voice = parsed.get("Voice")
        if voice == "Act":
            active += 1
        elif voice == "Pass":
            passive += 1

    total = active + passive
    return {"passive_ratio": passive / total if total > 0 else 0.0}


def xpos_bigram_profile(
    doc: Document,
    vocab: Optional[list[str]] = None,
) -> dict[str, float]:
    """Compute XPOS tag bigram relative frequencies.

    Parameters
    ----------
    doc : Document
        Preprocessed document.
    vocab : list[str], optional
        Fixed vocabulary of XPOS bigrams (joined with "_").
        If None, returns all observed bigrams.

    Returns
    -------
    dict[str, float]
        Mapping from XPOS bigram to relative frequency.
    """
    counts: Counter = Counter()

    for sent in doc.sentences:
        xpos_tags = [t.xpos for t in sent.tokens]
        if len(xpos_tags) < 2:
            continue
        for i in range(len(xpos_tags) - 1):
            bigram = f"{xpos_tags[i]}_{xpos_tags[i + 1]}"
            counts[bigram] += 1

    total = sum(counts.values())
    if total == 0:
        if vocab is not None:
            return {bg: 0.0 for bg in vocab}
        return {}

    if vocab is not None:
        return {bg: counts.get(bg, 0) / total for bg in vocab}

    return {bg: c / total for bg, c in counts.items()}


def build_xpos_bigram_vocab(
    documents: list[Document],
    top_k: int = 150,
) -> list[str]:
    """Build a shared XPOS bigram vocabulary from the corpus.

    Parameters
    ----------
    documents : list[Document]
        All preprocessed documents.
    top_k : int
        Number of top bigrams to keep.

    Returns
    -------
    list[str]
        Sorted list of top_k XPOS bigrams by total frequency.
    """
    global_counts: Counter = Counter()
    for doc in documents:
        profile = xpos_bigram_profile(doc)
        for bg, freq in profile.items():
            global_counts[bg] += freq

    return [bg for bg, _ in global_counts.most_common(top_k)]


def all_morphological_features(
    doc: Document,
    xpos_vocab: Optional[list[str]] = None,
) -> dict[str, float]:
    """Combine all morphological features into a single dict.

    Parameters
    ----------
    doc : Document
        Preprocessed document.
    xpos_vocab : list[str], optional
        Fixed XPOS bigram vocabulary. If None, XPOS bigrams are skipped.

    Returns
    -------
    dict[str, float]
        All morphological features.
    """
    feats: dict[str, float] = {}
    feats.update(morphological_category_features(doc))
    feats.update(passive_voice_ratio(doc))
    if xpos_vocab is not None:
        feats.update(xpos_bigram_profile(doc, vocab=xpos_vocab))
    return feats


def morphological_feature_names(
    xpos_vocab: Optional[list[str]] = None,
) -> list[str]:
    """Return ordered feature names for the morphological feature vector.

    Parameters
    ----------
    xpos_vocab : list[str], optional
        XPOS bigram vocabulary (determines XPOS bigram feature names).

    Returns
    -------
    list[str]
        Feature names in the same order as all_morphological_features().
    """
    names: list[str] = []
    # Category distributions
    for cat, known_values in MORPH_CATEGORIES.items():
        for val in known_values:
            names.append(f"morph_{cat}_{val}")
    # Passive ratio
    names.append("passive_ratio")
    # XPOS bigrams
    if xpos_vocab is not None:
        names.extend([f"xpos2_{bg}" for bg in xpos_vocab])
    return names
