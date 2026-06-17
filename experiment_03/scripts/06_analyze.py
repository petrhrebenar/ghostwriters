#!/usr/bin/env python3
"""Stage 4 — per-writer authorship analysis on span-isolated text.

Adapts experiment_02's pipeline to experiment_03's two-table dataset:
  * train per-judge binary classifiers on **DIS/CON spans** (dissents table);
  * LOO-CV evaluate them;
  * score each decision's **RATIO** (decisions table) — the court's own
    reasoning authored under the rapporteur — instead of the contaminated full
    decision text;
  * attribute the predicted author and compare it to the scraped
    ``judge_rapporteur_name`` to surface possible ghostwriting.

Steps (resume with --from-step):
  load · udpipe_dissents · features_dissents · evaluate ·
  udpipe_decisions · features_decisions · apply

Run from experiment_03/ with the project env (UDPipe model in models/):
    poetry run python scripts/06_analyze.py
    poetry run python scripts/06_analyze.py --from-step apply
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))

from fingerprint.classifiers import PerAuthorClassifier
from fingerprint.data_loader import author_summary, load_decisions, load_dissents
from fingerprint.evaluation import (
    format_results_summary,
    loo_per_author_evaluation,
    print_results,
)
from fingerprint.feature_importance import (
    extract_feature_importance,
    format_feature_importance_report,
)
from fingerprint.featureset import (
    DEFAULT_FEATURES,
    build_vocabs,
    extract_features,
    process_texts,
    softmax,
)
from fingerprint.preprocessing import UDPipeProcessor
from sklearn.preprocessing import StandardScaler

STEPS = ["load", "udpipe_dissents", "features_dissents", "evaluate",
         "udpipe_decisions", "features_decisions", "apply"]
OUT = EXP_DIR / "outputs"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-dissents", type=int, default=5)
    p.add_argument("--model-path", default=None, help="UDPipe Czech model (.udpipe)")
    p.add_argument("--features", nargs="+", default=DEFAULT_FEATURES, choices=DEFAULT_FEATURES)
    p.add_argument("--classifier", default="logistic", choices=["logistic", "xgboost", "both"])
    p.add_argument("--from-step", default="load", choices=STEPS)
    return p.parse_args()


def hdr(name): print(f"\n{'=' * 60}\nStep: {name}\n{'=' * 60}")
def save(obj, path): pickle.dump(obj, open(path, "wb")); print(f"  -> {path.name}")
def load(path): print(f"  <- {path.name}"); return pickle.load(open(path, "rb"))


def _udpipe(args):
    mp = Path(args.model_path) if args.model_path else None
    try:
        proc = UDPipeProcessor(model_path=mp)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}"); sys.exit(1)
    print("  UDPipe model loaded.")
    return proc


def _classifiers(args):
    return ["logistic", "xgboost"] if args.classifier == "both" else [args.classifier]


def main():
    args = parse_args()
    OUT.mkdir(exist_ok=True)
    start = STEPS.index(args.from_step)

    # 1. load ────────────────────────────────────────────────
    if start <= 0:
        hdr("load")
        dis_df = load_dissents(min_dissents=args.min_dissents)
        dec_df = load_decisions()
        print(f"  dissents: {len(dis_df)} from {dis_df['separate_opinion'].nunique()} authors")
        print(f"  decisions to score (with RATIO): {len(dec_df)}\n")
        print(author_summary(dis_df).to_string())
        save(dis_df, OUT / "corpus_dissents.pkl")
        save(dec_df, OUT / "corpus_decisions.pkl")
    else:
        dis_df = load(OUT / "corpus_dissents.pkl")
        dec_df = load(OUT / "corpus_decisions.pkl")

    # 2. udpipe_dissents ─────────────────────────────────────
    if start <= 1:
        hdr("udpipe_dissents")
        docs = process_texts(_udpipe(args), dis_df["separate_opinion_extracted"].tolist(),
                             dis_df["doc_id"].tolist())
        save(docs, OUT / "dissent_documents.pkl")
    else:
        docs = load(OUT / "dissent_documents.pkl")

    # 3. features_dissents ───────────────────────────────────
    if start <= 2:
        hdr("features_dissents")
        vocabs = build_vocabs(docs, args.features)
        X, feat_names = extract_features(docs, args.features, vocabs)
        y = dis_df["separate_opinion"].values
        print(f"  X: {X.shape}")
        save({"X": X, "y": y, "feature_names": feat_names, "vocabs": vocabs},
             OUT / "dissent_features.pkl")
    else:
        d = load(OUT / "dissent_features.pkl")
        X, y, feat_names, vocabs = d["X"], d["y"], d["feature_names"], d["vocabs"]

    # 4. evaluate ────────────────────────────────────────────
    if start <= 3:
        hdr("evaluate")
        for clf in _classifiers(args):
            print(f"\n--- {clf} ---")
            res = loo_per_author_evaluation(X, y, classifier_type=clf)
            print_results(res)
            (OUT / f"loo_summary_{clf}.txt").write_text(format_results_summary(res))
            sc = StandardScaler(); Xs = sc.fit_transform(X)
            pac = PerAuthorClassifier(classifier_type=clf); pac.fit(Xs, y)
            fi = extract_feature_importance(pac, feat_names, top_k=20)
            fi.to_csv(OUT / f"feature_importance_{clf}.csv", index=False)
            print(format_feature_importance_report(fi, top_k=5))

    # 5. udpipe_decisions (RATIO) ────────────────────────────
    if start <= 4:
        hdr("udpipe_decisions")
        ddocs = process_texts(_udpipe(args), dec_df["text"].tolist(), dec_df["doc_id"].tolist())
        save(ddocs, OUT / "decision_documents.pkl")
    else:
        ddocs = load(OUT / "decision_documents.pkl")

    # 6. features_decisions ──────────────────────────────────
    if start <= 5:
        hdr("features_decisions")
        Xd, _ = extract_features(ddocs, args.features, vocabs)
        save({"X": Xd}, OUT / "decision_features.pkl")
    else:
        Xd = load(OUT / "decision_features.pkl")["X"]

    # 7. apply (score RATIO, attribute vs rapporteur) ────────
    hdr("apply")
    for clf in _classifiers(args):
        print(f"\n--- {clf} ---")
        sc = StandardScaler()
        pac = PerAuthorClassifier(classifier_type=clf)
        pac.fit(sc.fit_transform(X), y)
        authors = pac.get_author_names()
        probas = pac.predict_proba(sc.transform(Xd))
        norm = softmax(probas)

        out = pd.DataFrame({"doc_id": dec_df["doc_id"].values})
        out["judge_rapporteur"] = dec_df.get("judge_rapporteur_name")
        out["formation"] = dec_df.get("formation")
        for i, a in enumerate(authors):
            out[f"prob_{a}"] = probas[:, i]
        out["predicted_author"] = [authors[j] for j in np.argmax(probas, axis=1)]
        out["max_probability"] = probas.max(axis=1)
        # probability assigned to the actual rapporteur (when they are a trained author)
        a_index = {a: i for i, a in enumerate(authors)}
        out["prob_rapporteur"] = [
            probas[r, a_index[rp]] if rp in a_index else np.nan
            for r, rp in enumerate(out["judge_rapporteur"].fillna(""))
        ]
        out["rapporteur_is_trained_author"] = out["judge_rapporteur"].isin(authors)
        out.to_csv(OUT / f"authorship_probabilities_{clf}.csv", index=False)
        print(f"  -> authorship_probabilities_{clf}.csv  ({len(out)} decisions, {len(authors)} authors)")
        save({"classifier": pac, "scaler": sc, "authors": authors, "feature_names": feat_names},
             OUT / f"trained_classifiers_{clf}.pkl")

    print(f"\n{'=' * 60}\nDone. Outputs in {OUT}/")


if __name__ == "__main__":
    main()
