#!/usr/bin/env python3
"""Score every decision section with the trained per-judge classifier.

Reuses the model saved by ``scripts/06_analyze.py`` (``trained_classifiers_*.pkl``)
— no retraining. For each (decision, section) in ``data/dataset/sections.csv`` we
UDPipe-tag the section text, extract the same features, and emit the per-judge
probability vector (raw + row-normalized softmax).

This is the basis for the section x author "decision cards" and for the key
control: **DIS** sections are the same genre as the training data with a known
author, so their self-attribution isolates genre-transfer from ghostwriting,
whereas RATIO sections test cross-genre transfer.

Output: outputs/section_scores_<clf>.csv
        (doc_id, label, words, judge_rapporteur_name, separate_opinion,
         predicted_author, prob_<author>..., norm_<author>...)

Usage:
    poetry run python score_sections.py --classifier logistic
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

from fingerprint.featureset import extract_features, process_texts, softmax
from fingerprint.preprocessing import UDPipeProcessor

OUT = EXP_DIR / "outputs"
SECTIONS = EXP_DIR / "data" / "dataset" / "sections.csv"
DEFAULT_FEATURES = ["function_words", "surface", "char_ngrams", "pos_ngrams", "morphology"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classifier", default="logistic")
    ap.add_argument("--model-path", default=str(EXP_DIR / "models" / "czech-pdt-ud-2.5-191206.udpipe"))
    ap.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    args = ap.parse_args()

    bundle = pickle.load(open(OUT / f"trained_classifiers_{args.classifier}.pkl", "rb"))
    pac, scaler, authors = bundle["classifier"], bundle["scaler"], bundle["authors"]
    vocabs = pickle.load(open(OUT / "dissent_features.pkl", "rb"))["vocabs"]

    sec = pd.read_csv(SECTIONS)
    print(f"Scoring {len(sec)} sections across {sec['doc_id'].nunique()} decisions "
          f"with {len(authors)} judge classifiers ({args.classifier}).")

    proc = UDPipeProcessor(model_path=Path(args.model_path))
    docs = process_texts(proc, sec["text"].fillna("").tolist(),
                         [f"{r.doc_id}:{r.label}" for r in sec.itertuples()])
    X, _ = extract_features(docs, args.features, vocabs)
    Xs = scaler.transform(X)
    probas = pac.predict_proba(Xs)
    norm = softmax(probas)

    out = sec[["doc_id", "label", "words", "judge_rapporteur_name", "separate_opinion"]].copy()
    out["predicted_author"] = [authors[j] for j in np.argmax(probas, axis=1)]
    out["max_probability"] = probas.max(axis=1)
    a_idx = {a: i for i, a in enumerate(authors)}
    # probability the section's text matches the decision's rapporteur (if trained)
    out["prob_rapporteur"] = [
        probas[r, a_idx[rp]] if isinstance(rp, str) and rp in a_idx else np.nan
        for r, rp in enumerate(out["judge_rapporteur_name"])
    ]
    for i, a in enumerate(authors):
        out[f"prob_{a}"] = probas[:, i]
    for i, a in enumerate(authors):
        out[f"norm_{a}"] = norm[:, i]

    path = OUT / f"section_scores_{args.classifier}.csv"
    out.to_csv(path, index=False)
    print(f"-> {path.name}")

    # quick control read: in-genre (DIS) vs cross-genre (RATIO) self-attribution
    print("\nSelf-attribution by section type (rapporteur is a trained judge):")
    for lab in ["RATIO", "FACT", "PROC", "REC"]:
        s = out[(out["label"] == lab) & out["prob_rapporteur"].notna()]
        if len(s):
            match = (s["predicted_author"] == s["judge_rapporteur_name"]).mean()
            print(f"  {lab:5s} n={len(s):3d}  pred==rapporteur {match:.3f}  median prob_rapporteur {s['prob_rapporteur'].median():.3f}")
    # DIS: known author is the separate-opinion judge, not the rapporteur — handled in plots


if __name__ == "__main__":
    main()
