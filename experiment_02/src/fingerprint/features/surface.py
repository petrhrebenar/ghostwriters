"""Surface-level stylometric features: sentence/word length, punctuation."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from fingerprint.preprocessing import Document

PUNCTUATION_MARKS = [",", ".", ";", ":", "!", "?", "-", "(", ")", '"', "…"]


def sentence_length_features(doc: Document) -> dict[str, float]:
    """Compute sentence length statistics (in tokens)."""
    lengths = [len(s.tokens) for s in doc.sentences]
    if not lengths:
        return {
            "sent_len_mean": 0.0,
            "sent_len_std": 0.0,
            "sent_len_median": 0.0,
            "sent_len_min": 0.0,
            "sent_len_max": 0.0,
            "sent_count": 0.0,
        }
    arr = np.array(lengths, dtype=float)
    return {
        "sent_len_mean": float(arr.mean()),
        "sent_len_std": float(arr.std()),
        "sent_len_median": float(np.median(arr)),
        "sent_len_min": float(arr.min()),
        "sent_len_max": float(arr.max()),
        "sent_count": float(len(lengths)),
    }


def word_length_features(doc: Document) -> dict[str, float]:
    """Compute word length statistics (in characters)."""
    lengths = [len(t.form) for t in doc.all_tokens if t.upos != "PUNCT"]
    if not lengths:
        return {
            "word_len_mean": 0.0,
            "word_len_std": 0.0,
            "word_len_median": 0.0,
        }
    arr = np.array(lengths, dtype=float)
    return {
        "word_len_mean": float(arr.mean()),
        "word_len_std": float(arr.std()),
        "word_len_median": float(np.median(arr)),
    }


def punctuation_features(doc: Document) -> dict[str, float]:
    """Compute punctuation frequencies per 1000 tokens."""
    all_forms = doc.all_forms
    total = len(all_forms)
    if total == 0:
        return {f"punct_{p}_per1k": 0.0 for p in PUNCTUATION_MARKS}

    counts = Counter(all_forms)
    return {
        f"punct_{p}_per1k": counts.get(p, 0) / total * 1000
        for p in PUNCTUATION_MARKS
    }


def vocabulary_richness_features(doc: Document) -> dict[str, float]:
    """Compute vocabulary richness measures."""
    forms = [t.form.lower() for t in doc.all_tokens if t.upos != "PUNCT"]
    lemmas = [t.lemma.lower() for t in doc.all_tokens if t.upos != "PUNCT"]
    n = len(forms)
    if n == 0:
        return {"ttr_form": 0.0, "ttr_lemma": 0.0, "hapax_ratio": 0.0}

    unique_forms = set(forms)
    unique_lemmas = set(lemmas)
    hapax = sum(1 for w, c in Counter(forms).items() if c == 1)

    return {
        "ttr_form": len(unique_forms) / n,
        "ttr_lemma": len(unique_lemmas) / n,
        "hapax_ratio": hapax / n,
    }


def all_surface_features(doc: Document) -> dict[str, float]:
    """Combine all surface features into a single dict."""
    feats: dict[str, float] = {}
    feats.update(sentence_length_features(doc))
    feats.update(word_length_features(doc))
    feats.update(punctuation_features(doc))
    feats.update(vocabulary_richness_features(doc))
    return feats
