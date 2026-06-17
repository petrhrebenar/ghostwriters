"""Character n-gram and POS n-gram feature extraction."""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from fingerprint.preprocessing import Document


def character_ngram_profile(
    doc: Document,
    n: int = 3,
    top_k: Optional[int] = None,
    vocab: Optional[list[str]] = None,
) -> dict[str, float]:
    """Compute character n-gram relative frequencies.

    Parameters
    ----------
    doc : Document
        Preprocessed document.
    n : int
        Size of character n-grams.
    top_k : int, optional
        If vocab is None, return only the top_k most frequent n-grams.
    vocab : list[str], optional
        Fixed vocabulary of n-grams. If provided, only count these.

    Returns
    -------
    dict[str, float]
        Mapping from n-gram to relative frequency.
    """
    # Concatenate all token forms with space separators (preserving word boundaries)
    text = " ".join(t.form.lower() for t in doc.all_tokens)
    if len(text) < n:
        return {}

    ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]
    total = len(ngrams)
    counts = Counter(ngrams)

    if vocab is not None:
        return {ng: counts.get(ng, 0) / total for ng in vocab}

    if top_k is not None:
        counts = dict(counts.most_common(top_k))

    return {ng: c / total for ng, c in counts.items()}


def pos_ngram_profile(
    doc: Document,
    n: int = 2,
    top_k: Optional[int] = None,
    vocab: Optional[list[str]] = None,
) -> dict[str, float]:
    """Compute POS tag n-gram relative frequencies.

    Parameters
    ----------
    doc : Document
        Preprocessed document with POS tags.
    n : int
        Size of POS n-grams.
    top_k : int, optional
        If vocab is None, return only top_k most frequent n-grams.
    vocab : list[str], optional
        Fixed vocabulary of POS n-grams (joined with "_").

    Returns
    -------
    dict[str, float]
        Mapping from POS n-gram (e.g. "NOUN_VERB") to relative frequency.
    """
    results: dict[str, float] = {}

    for sent in doc.sentences:
        tags = sent.upos_tags
        if len(tags) < n:
            continue
        for i in range(len(tags) - n + 1):
            ng = "_".join(tags[i : i + n])
            results[ng] = results.get(ng, 0) + 1

    total = sum(results.values())
    if total == 0:
        return {}

    if vocab is not None:
        return {ng: results.get(ng, 0) / total for ng in vocab}

    results = {ng: c / total for ng, c in results.items()}

    if top_k is not None:
        sorted_items = sorted(results.items(), key=lambda x: -x[1])[:top_k]
        results = dict(sorted_items)

    return results


def build_ngram_vocab_from_corpus(
    documents: list[Document],
    ngram_fn,
    n: int,
    top_k: int,
    **kwargs,
) -> list[str]:
    """Build a shared vocabulary of top_k n-grams across a corpus.

    Parameters
    ----------
    documents : list[Document]
        All preprocessed documents.
    ngram_fn : callable
        One of character_ngram_profile or pos_ngram_profile.
    n : int
        N-gram size.
    top_k : int
        Number of top n-grams to keep.

    Returns
    -------
    list[str]
        Sorted list of top_k n-grams by total frequency.
    """
    global_counts: Counter = Counter()
    for doc in documents:
        profile = ngram_fn(doc, n=n, **kwargs)
        for ng, freq in profile.items():
            global_counts[ng] += freq

    return [ng for ng, _ in global_counts.most_common(top_k)]
