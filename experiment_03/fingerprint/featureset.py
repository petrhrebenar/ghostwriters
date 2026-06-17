"""Reusable feature-extraction + UDPipe helpers shared by the analysis script.

Factored out of experiment_02's ``run_pipeline.py`` so both the dissent
(training) and decision/RATIO (scoring) passes use identical logic.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .features.function_words import (
    function_word_feature_names,
    function_word_frequencies,
)
from .features.morphology import (
    all_morphological_features,
    build_xpos_bigram_vocab,
    morphological_feature_names,
)
from .features.ngrams import (
    build_ngram_vocab_from_corpus,
    character_ngram_profile,
    pos_ngram_profile,
)
from .features.surface import all_surface_features
from .preprocessing import clean_text

DEFAULT_FEATURES = ["function_words", "surface", "char_ngrams", "pos_ngrams", "morphology"]


def process_texts(processor, texts: List[str], doc_ids: List[str]) -> list:
    """Tokenize/tag a list of texts with a UDPipe processor."""
    documents = []
    n = len(texts)
    for i, (text, doc_id) in enumerate(zip(texts, doc_ids)):
        documents.append(processor.process(clean_text(str(text)), doc_id=str(doc_id)))
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  Processed {i + 1}/{n}", flush=True)
    return documents


def _single_doc_features(doc, feature_sets, char_vocab, pos_vocab, xpos_vocab,
                         collect_names=False):
    vec: List[float] = []
    names: List[str] = []
    if "function_words" in feature_sets:
        vec.extend(function_word_frequencies(doc).tolist())
        if collect_names:
            names.extend(function_word_feature_names())
    if "surface" in feature_sets:
        sf = all_surface_features(doc)
        vec.extend(sf.values())
        if collect_names:
            names.extend(sf.keys())
    if "char_ngrams" in feature_sets and char_vocab:
        cng = character_ngram_profile(doc, n=3, vocab=char_vocab)
        vec.extend(cng.values())
        if collect_names:
            names.extend([f"char3_{ng}" for ng in char_vocab])
    if "pos_ngrams" in feature_sets and pos_vocab:
        png = pos_ngram_profile(doc, n=2, vocab=pos_vocab)
        vec.extend(png.values())
        if collect_names:
            names.extend([f"pos2_{ng}" for ng in pos_vocab])
    if "morphology" in feature_sets:
        morph = all_morphological_features(doc, xpos_vocab=xpos_vocab)
        vec.extend(morph.values())
        if collect_names:
            names.extend(morphological_feature_names(xpos_vocab=xpos_vocab))
    return vec, names


def build_vocabs(documents: list, features: List[str]) -> dict:
    """Build char/pos/xpos vocabularies from the training corpus."""
    char_vocab = pos_vocab = xpos_vocab = None
    if "char_ngrams" in features:
        char_vocab = build_ngram_vocab_from_corpus(documents, character_ngram_profile, n=3, top_k=200)
    if "pos_ngrams" in features:
        pos_vocab = build_ngram_vocab_from_corpus(documents, pos_ngram_profile, n=2, top_k=100)
    if "morphology" in features:
        xpos_vocab = build_xpos_bigram_vocab(documents, top_k=150)
    return {"char_vocab": char_vocab, "pos_vocab": pos_vocab, "xpos_vocab": xpos_vocab}


def extract_features(documents: list, features: List[str], vocabs: dict):
    """Feature matrix X + feature_names using the given (prebuilt) vocabularies."""
    cv, pv, xv = vocabs.get("char_vocab"), vocabs.get("pos_vocab"), vocabs.get("xpos_vocab")
    feature_names: List[str] = []
    rows: List[List[float]] = []
    for doc in documents:
        vec, names = _single_doc_features(doc, features, cv, pv, xv, collect_names=(not feature_names))
        rows.append(vec)
        if not feature_names:
            feature_names = names
    return np.array(rows), feature_names


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
