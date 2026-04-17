#!/usr/bin/env python3
"""Run the per-writer authorship classification pipeline.

The pipeline has 7 steps:

  1. load              Load & filter CSV                    -> outputs/corpus.pkl
  2. udpipe_dissents   Tokenize/tag dissents with UDPipe    -> outputs/dissent_documents.pkl
  3. features_dissents Extract feature matrix from dissents  -> outputs/dissent_features.pkl
  4. evaluate          LOO-CV with per-author classifiers    -> outputs/loo_results.pkl
  5. udpipe_decisions  Tokenize/tag decisions with UDPipe    -> outputs/decision_documents.pkl
  6. features_decisions Extract features from decisions       -> outputs/decision_features.pkl
  7. apply             Score decisions with trained models    -> outputs/authorship_probabilities.csv

Use --from-step to resume from any step (earlier outputs are loaded from disk).

Examples:
    poetry run python scripts/run_pipeline.py                         # full run
    poetry run python scripts/run_pipeline.py --from-step evaluate    # re-evaluate
    poetry run python scripts/run_pipeline.py --from-step apply       # re-apply only
    poetry run python scripts/run_pipeline.py --classifier xgboost    # use XGBoost
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fingerprint.classifiers import PerAuthorClassifier
from fingerprint.data_loader import author_summary, load_dissents
from fingerprint.evaluation import (
    format_results_summary,
    loo_per_author_evaluation,
    print_results,
)
from fingerprint.feature_importance import (
    extract_feature_importance,
    format_feature_importance_report,
)
from fingerprint.features.function_words import (
    function_word_feature_names,
    function_word_frequencies,
)
from fingerprint.features.morphology import (
    all_morphological_features,
    build_xpos_bigram_vocab,
    morphological_feature_names,
)
from fingerprint.features.ngrams import (
    build_ngram_vocab_from_corpus,
    character_ngram_profile,
    pos_ngram_profile,
)
from fingerprint.features.surface import all_surface_features
from fingerprint.preprocessing import UDPipeProcessor, clean_text

STEPS = [
    "load", "udpipe_dissents", "features_dissents", "evaluate",
    "udpipe_decisions", "features_decisions", "apply",
]
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-writer authorship classification pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--min-dissents", type=int, default=5,
        help="Minimum dissents per author (default: 5)",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to UDPipe Czech model (.udpipe file)",
    )
    parser.add_argument(
        "--features", type=str, nargs="+",
        default=["function_words", "surface", "char_ngrams", "pos_ngrams", "morphology"],
        choices=["function_words", "surface", "char_ngrams", "pos_ngrams", "morphology"],
        help="Feature sets to use",
    )
    parser.add_argument(
        "--classifier", type=str, default="logistic",
        choices=["logistic", "xgboost", "both"],
        help="Classifier type (default: logistic)",
    )
    parser.add_argument(
        "--from-step", type=str, default="load",
        choices=STEPS,
        help="Resume pipeline from this step (default: load = full run)",
    )
    return parser.parse_args()


# ── Helpers ──────────────────────────────────────────────────

def step_header(name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Step: {name}")
    print("=" * 60)


def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  -> Saved {path.name}")


def load_pickle(path: Path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"  <- Loaded {path.name}")
    return obj


def _get_udpipe_processor(args):
    """Create UDPipe processor, exit on failure."""
    model_path = Path(args.model_path) if args.model_path else None
    try:
        processor = UDPipeProcessor(model_path=model_path)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)
    print("  UDPipe model loaded.")
    return processor


def _process_texts(processor, texts: list[str], doc_ids: list[str]) -> list:
    """Process a list of texts with UDPipe."""
    documents = []
    n = len(texts)
    for i, (text, doc_id) in enumerate(zip(texts, doc_ids)):
        cleaned = clean_text(str(text))
        doc = processor.process(cleaned, doc_id=str(doc_id))
        documents.append(doc)
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  Processed {i + 1}/{n}", flush=True)
    return documents


def _extract_features(args, documents: list, label: str = "") -> tuple:
    """Extract feature matrix from documents.

    Returns
    -------
    X : np.ndarray
    feature_names : list[str]
    vocabs : dict (char_vocab, pos_vocab, xpos_vocab)
    """
    feature_names: list[str] = []

    # Build vocabularies
    char_vocab = None
    pos_vocab = None
    xpos_vocab = None

    if "char_ngrams" in args.features:
        print(f"  Building character 3-gram vocabulary{' (' + label + ')' if label else ''}...")
        char_vocab = build_ngram_vocab_from_corpus(
            documents, character_ngram_profile, n=3, top_k=200,
        )
    if "pos_ngrams" in args.features:
        print(f"  Building POS 2-gram vocabulary{' (' + label + ')' if label else ''}...")
        pos_vocab = build_ngram_vocab_from_corpus(
            documents, pos_ngram_profile, n=2, top_k=100,
        )
    if "morphology" in args.features:
        print(f"  Building XPOS bigram vocabulary{' (' + label + ')' if label else ''}...")
        xpos_vocab = build_xpos_bigram_vocab(documents, top_k=150)

    vocabs = {"char_vocab": char_vocab, "pos_vocab": pos_vocab, "xpos_vocab": xpos_vocab}

    feature_matrix: list[list[float]] = []
    for doc in documents:
        vec, names = _extract_single_document_features(
            doc, args.features, char_vocab, pos_vocab, xpos_vocab,
            collect_names=(not feature_names),
        )
        feature_matrix.append(vec)
        if not feature_names:
            feature_names = names

    X = np.array(feature_matrix)
    print(f"  Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"  Feature sets: {args.features}")
    return X, feature_names, vocabs


def _extract_features_with_vocabs(args, documents: list, vocabs: dict) -> tuple:
    """Extract features using pre-built vocabularies (for decision texts).

    Returns
    -------
    X : np.ndarray
    feature_names : list[str]
    """
    char_vocab = vocabs.get("char_vocab")
    pos_vocab = vocabs.get("pos_vocab")
    xpos_vocab = vocabs.get("xpos_vocab")
    feature_names: list[str] = []
    feature_matrix: list[list[float]] = []

    for doc in documents:
        vec, names = _extract_single_document_features(
            doc, args.features, char_vocab, pos_vocab, xpos_vocab,
            collect_names=(not feature_names),
        )
        feature_matrix.append(vec)
        if not feature_names:
            feature_names = names

    X = np.array(feature_matrix)
    print(f"  Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    return X, feature_names


def _extract_single_document_features(
    doc, feature_sets, char_vocab, pos_vocab, xpos_vocab, collect_names=False,
) -> tuple[list[float], list[str]]:
    """Extract feature vector for a single document."""
    vec: list[float] = []
    names: list[str] = []

    if "function_words" in feature_sets:
        fw = function_word_frequencies(doc)
        vec.extend(fw.tolist())
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


def _softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax normalization."""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ── Step implementations ─────────────────────────────────────

def step_load(args) -> pd.DataFrame:
    """Step 1: Load and filter the CSV dataset."""
    step_header("load")
    df = load_dissents(min_dissents=args.min_dissents)
    print(f"  {len(df)} dissents from {df['separate_opinion'].nunique()} authors\n")
    print(author_summary(df).to_string())
    save_pickle(df, OUTPUTS_DIR / "corpus.pkl")
    return df


def step_udpipe_dissents(args, df: pd.DataFrame) -> list:
    """Step 2: Process dissent texts with UDPipe."""
    step_header("udpipe_dissents")
    processor = _get_udpipe_processor(args)
    documents = _process_texts(
        processor,
        df["separate_opinion_extracted"].tolist(),
        df["doc_id"].tolist(),
    )
    save_pickle(documents, OUTPUTS_DIR / "dissent_documents.pkl")
    return documents


def step_features_dissents(args, df: pd.DataFrame, documents: list) -> tuple:
    """Step 3: Extract feature matrix from dissent documents."""
    step_header("features_dissents")
    X, feature_names, vocabs = _extract_features(args, documents, label="dissents")
    y = df["separate_opinion"].values

    # Save as CSV for inspection
    feat_df = pd.DataFrame(X, columns=feature_names)
    feat_df.insert(0, "doc_id", df["doc_id"].values)
    feat_df.insert(1, "author", y)
    feat_df.to_csv(OUTPUTS_DIR / "dissent_feature_matrix.csv", index=False)
    print(f"  -> Saved dissent_feature_matrix.csv")

    save_pickle(
        {"X": X, "y": y, "feature_names": feature_names, "vocabs": vocabs},
        OUTPUTS_DIR / "dissent_features.pkl",
    )
    return X, y, feature_names, vocabs


def step_evaluate(args, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> list[dict]:
    """Step 4: Run LOO-CV evaluation."""
    step_header("evaluate")
    all_results = []

    classifier_types = (
        ["logistic", "xgboost"] if args.classifier == "both"
        else [args.classifier]
    )

    for clf_type in classifier_types:
        print(f"\n--- Evaluating: {clf_type} ---")
        results = loo_per_author_evaluation(X, y, classifier_type=clf_type)
        print()
        print_results(results)
        print()
        all_results.append(results)

        # Save LOO probability matrix
        prob_df = pd.DataFrame(
            results["probabilities"],
            columns=[f"prob_{a}" for a in results["authors"]],
        )
        prob_df.insert(0, "true_author", results["true_labels"])

        # Add softmax-normalized probabilities
        norm_probas = _softmax(results["probabilities"])
        for i, author in enumerate(results["authors"]):
            prob_df[f"norm_{author}"] = norm_probas[:, i]

        prob_df.to_csv(
            OUTPUTS_DIR / f"loo_probabilities_{clf_type}.csv", index=False,
        )
        print(f"  -> Saved loo_probabilities_{clf_type}.csv")

    # Save summary
    summary_path = OUTPUTS_DIR / "results_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Per-writer authorship classification results\n")
        f.write(f"Authors: {len(np.unique(y))} (min {args.min_dissents} dissents)\n")
        f.write(f"Samples: {X.shape[0]}, Features: {X.shape[1]}\n")
        f.write(f"Feature sets: {args.features}\n")
        f.write(f"{'=' * 60}\n\n")
        for r in all_results:
            f.write(format_results_summary(r))
            f.write(f"\n\n{'=' * 60}\n\n")
    print(f"  -> Saved results_summary.txt")

    # Feature importance (train on full data)
    for clf_type in classifier_types:
        print(f"\n--- Feature importance: {clf_type} ---")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pac = PerAuthorClassifier(classifier_type=clf_type)
        pac.fit(X_scaled, y)
        fi_df = extract_feature_importance(pac, feature_names, top_k=20)
        fi_df.to_csv(
            OUTPUTS_DIR / f"feature_importance_{clf_type}.csv", index=False,
        )
        print(f"  -> Saved feature_importance_{clf_type}.csv")
        print(format_feature_importance_report(fi_df, top_k=5))

    save_pickle(all_results, OUTPUTS_DIR / "loo_results.pkl")
    return all_results


def step_udpipe_decisions(args, df: pd.DataFrame) -> list:
    """Step 5: Process full decision texts with UDPipe."""
    step_header("udpipe_decisions")
    processor = _get_udpipe_processor(args)
    documents = _process_texts(
        processor,
        df["text"].tolist(),
        df["doc_id"].tolist(),
    )
    save_pickle(documents, OUTPUTS_DIR / "decision_documents.pkl")
    return documents


def step_features_decisions(args, documents: list, vocabs: dict) -> tuple:
    """Step 6: Extract feature matrix from decision documents."""
    step_header("features_decisions")
    X, feature_names = _extract_features_with_vocabs(args, documents, vocabs)
    save_pickle(
        {"X": X, "feature_names": feature_names},
        OUTPUTS_DIR / "decision_features.pkl",
    )
    return X, feature_names


def step_apply(
    args,
    df: pd.DataFrame,
    X_dissents: np.ndarray,
    y_dissents: np.ndarray,
    X_decisions: np.ndarray,
    feature_names: list[str],
) -> None:
    """Step 7: Score decision texts with trained classifiers."""
    step_header("apply")
    from sklearn.preprocessing import StandardScaler

    classifier_types = (
        ["logistic", "xgboost"] if args.classifier == "both"
        else [args.classifier]
    )

    for clf_type in classifier_types:
        print(f"\n--- Applying: {clf_type} ---")

        # Train on all dissents
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_dissents)
        X_apply = scaler.transform(X_decisions)

        pac = PerAuthorClassifier(classifier_type=clf_type)
        pac.fit(X_train, y_dissents)
        authors = pac.get_author_names()

        # Predict probabilities
        probas = pac.predict_proba(X_apply)
        norm_probas = _softmax(probas)

        # Build output DataFrame
        out_df = pd.DataFrame()
        out_df["doc_id"] = df["doc_id"].values
        if "judge_rapporteur_name" in df.columns:
            out_df["judge_rapporteur"] = df["judge_rapporteur_name"].values

        for i, author in enumerate(authors):
            out_df[f"prob_{author}"] = probas[:, i]
        for i, author in enumerate(authors):
            out_df[f"norm_{author}"] = norm_probas[:, i]

        # Add predicted author (highest raw probability)
        out_df["predicted_author"] = [authors[j] for j in np.argmax(probas, axis=1)]
        out_df["max_probability"] = probas.max(axis=1)

        out_path = OUTPUTS_DIR / f"authorship_probabilities_{clf_type}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  -> Saved {out_path.name}")
        print(f"  {len(out_df)} decisions scored with {len(authors)} author classifiers")

        # Save trained classifiers for reuse
        save_pickle(
            {"classifier": pac, "scaler": scaler, "feature_names": feature_names, "authors": authors},
            OUTPUTS_DIR / f"trained_classifiers_{clf_type}.pkl",
        )


# ── Main ─────────────────────────────────────────────────────

def main():
    args = parse_args()
    OUTPUTS_DIR.mkdir(exist_ok=True)

    start_idx = STEPS.index(args.from_step)

    # ── Step 1: load ──
    if start_idx <= 0:
        df = step_load(args)
    else:
        print(f"\n  Skipping step 'load' — loading from outputs/corpus.pkl")
        df = load_pickle(OUTPUTS_DIR / "corpus.pkl")

    # ── Step 2: udpipe_dissents ──
    if start_idx <= 1:
        dissent_docs = step_udpipe_dissents(args, df)
    else:
        print(f"  Skipping step 'udpipe_dissents' — loading from outputs/dissent_documents.pkl")
        dissent_docs = load_pickle(OUTPUTS_DIR / "dissent_documents.pkl")

    # ── Step 3: features_dissents ──
    if start_idx <= 2:
        X, y, feature_names, vocabs = step_features_dissents(args, df, dissent_docs)
    else:
        print(f"  Skipping step 'features_dissents' — loading from outputs/dissent_features.pkl")
        data = load_pickle(OUTPUTS_DIR / "dissent_features.pkl")
        X, y, feature_names, vocabs = data["X"], data["y"], data["feature_names"], data["vocabs"]

    # ── Step 4: evaluate ──
    if start_idx <= 3:
        step_evaluate(args, X, y, feature_names)

    # ── Step 5: udpipe_decisions ──
    if start_idx <= 4:
        decision_docs = step_udpipe_decisions(args, df)
    else:
        print(f"  Skipping step 'udpipe_decisions' — loading from outputs/decision_documents.pkl")
        decision_docs = load_pickle(OUTPUTS_DIR / "decision_documents.pkl")

    # ── Step 6: features_decisions ──
    if start_idx <= 5:
        X_decisions, _ = step_features_decisions(args, decision_docs, vocabs)
    else:
        print(f"  Skipping step 'features_decisions' — loading from outputs/decision_features.pkl")
        data = load_pickle(OUTPUTS_DIR / "decision_features.pkl")
        X_decisions = data["X"]

    # ── Step 7: apply ──
    step_apply(args, df, X, y, X_decisions, feature_names)

    print(f"\n{'=' * 60}")
    print("Done. All outputs in outputs/")


if __name__ == "__main__":
    main()
