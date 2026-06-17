"""Evaluation: leave-one-out cross-validation with per-author binary metrics."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from fingerprint.classifiers import PerAuthorClassifier


def loo_per_author_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    classifier_type: str = "logistic",
    scale: bool = True,
) -> dict:
    """Leave-one-out evaluation with per-author binary classifiers.

    For each LOO fold:
    1. Fit scaler on training data
    2. Train 19 binary classifiers (one per author)
    3. Score the held-out sample with all classifiers → 19 probabilities

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    classifier_type : str
        "logistic" or "xgboost".
    scale : bool
        Whether to StandardScale features.

    Returns
    -------
    dict with keys:
        - method: str
        - authors: list[str]
        - probabilities: np.ndarray (n_samples, n_authors) — LOO probabilities
        - true_labels: np.ndarray (n_samples,)
        - per_author_metrics: dict[str, dict] — per-author ROC AUC, PR AUC, F1, etc.
        - rank1_accuracy: float
        - rank3_accuracy: float
        - mrr: float
    """
    authors = sorted(np.unique(y).tolist())
    n_authors = len(authors)
    loo = LeaveOneOut()
    n_total = X.shape[0]

    # Collect LOO probabilities: (n_samples, n_authors)
    all_probas = np.zeros((n_total, n_authors))
    true_labels = np.empty(n_total, dtype=object)

    for i, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        pac = PerAuthorClassifier(
            classifier_type=classifier_type,
            authors=authors,
        )
        pac.fit(X_train, y_train)
        probas = pac.predict_proba(X_test)  # (1, n_authors)
        all_probas[i, :] = probas[0]
        true_labels[i] = y_test[0]

        if (i + 1) % 50 == 0 or i == n_total - 1:
            print(f"  LOO-CV ({classifier_type}): {i + 1}/{n_total}", flush=True)

    # Compute per-author binary metrics
    per_author_metrics = _compute_per_author_metrics(all_probas, true_labels, authors)

    # Compute ranking metrics
    rank1_acc, rank3_acc, mrr = _compute_ranking_metrics(all_probas, true_labels, authors)

    return {
        "method": f"per_author_{classifier_type}",
        "authors": authors,
        "probabilities": all_probas,
        "true_labels": true_labels,
        "per_author_metrics": per_author_metrics,
        "rank1_accuracy": rank1_acc,
        "rank3_accuracy": rank3_acc,
        "mrr": mrr,
    }


def _compute_per_author_metrics(
    probas: np.ndarray,
    true_labels: np.ndarray,
    authors: list[str],
) -> dict[str, dict]:
    """Compute per-author binary classification metrics.

    Parameters
    ----------
    probas : np.ndarray, shape (n_samples, n_authors)
    true_labels : np.ndarray, shape (n_samples,)
    authors : list[str]

    Returns
    -------
    dict[str, dict]
        Per-author metrics: roc_auc, avg_precision, f1, precision, recall, n_samples.
    """
    metrics = {}
    for i, author in enumerate(authors):
        y_true_binary = (true_labels == author).astype(int)
        y_scores = probas[:, i]
        y_pred_binary = (y_scores >= 0.5).astype(int)

        n_pos = int(y_true_binary.sum())
        author_metrics = {"n_samples": n_pos}

        # ROC AUC (requires both classes present)
        if n_pos > 0 and n_pos < len(y_true_binary):
            try:
                author_metrics["roc_auc"] = float(roc_auc_score(y_true_binary, y_scores))
            except ValueError:
                author_metrics["roc_auc"] = float("nan")
            try:
                author_metrics["avg_precision"] = float(
                    average_precision_score(y_true_binary, y_scores)
                )
            except ValueError:
                author_metrics["avg_precision"] = float("nan")
        else:
            author_metrics["roc_auc"] = float("nan")
            author_metrics["avg_precision"] = float("nan")

        author_metrics["f1"] = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
        author_metrics["precision"] = float(
            precision_score(y_true_binary, y_pred_binary, zero_division=0)
        )
        author_metrics["recall"] = float(
            recall_score(y_true_binary, y_pred_binary, zero_division=0)
        )

        metrics[author] = author_metrics

    return metrics


def _compute_ranking_metrics(
    probas: np.ndarray,
    true_labels: np.ndarray,
    authors: list[str],
) -> tuple[float, float, float]:
    """Compute ranking-based metrics from probability matrix.

    Parameters
    ----------
    probas : np.ndarray, shape (n_samples, n_authors)
    true_labels : np.ndarray, shape (n_samples,)
    authors : list[str]

    Returns
    -------
    rank1_accuracy : float
    rank3_accuracy : float
    mrr : float (mean reciprocal rank)
    """
    n = len(true_labels)
    rank1_correct = 0
    rank3_correct = 0
    reciprocal_ranks = []

    for i in range(n):
        # Rank authors by descending probability
        ranked_idx = np.argsort(-probas[i])
        ranked_authors = [authors[j] for j in ranked_idx]

        true_author = true_labels[i]
        if true_author in ranked_authors:
            rank = ranked_authors.index(true_author) + 1  # 1-indexed
        else:
            rank = len(authors) + 1

        if rank == 1:
            rank1_correct += 1
        if rank <= 3:
            rank3_correct += 1
        reciprocal_ranks.append(1.0 / rank)

    rank1_acc = rank1_correct / n if n > 0 else 0.0
    rank3_acc = rank3_correct / n if n > 0 else 0.0
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    return rank1_acc, rank3_acc, mrr


def format_results_summary(results: dict) -> str:
    """Format evaluation results as a human-readable string.

    Parameters
    ----------
    results : dict
        Output of loo_per_author_evaluation().

    Returns
    -------
    str
        Formatted summary.
    """
    lines = []
    lines.append(f"Method: {results['method']}")
    lines.append(f"Rank-1 accuracy: {results['rank1_accuracy']:.3f}")
    lines.append(f"Rank-3 accuracy: {results['rank3_accuracy']:.3f}")
    lines.append(f"MRR: {results['mrr']:.3f}")
    lines.append("")
    lines.append("Per-author metrics:")
    lines.append(f"{'Author':<25} {'N':>4} {'ROC AUC':>8} {'PR AUC':>8} {'F1':>6} {'Prec':>6} {'Rec':>6}")
    lines.append("-" * 75)

    for author in results["authors"]:
        m = results["per_author_metrics"][author]
        roc = f"{m['roc_auc']:.3f}" if not np.isnan(m.get("roc_auc", float("nan"))) else "  N/A"
        pr = f"{m['avg_precision']:.3f}" if not np.isnan(m.get("avg_precision", float("nan"))) else "  N/A"
        lines.append(
            f"{author:<25} {m['n_samples']:>4} {roc:>8} {pr:>8} "
            f"{m['f1']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f}"
        )

    # Macro averages (only for authors with valid metrics)
    valid = [m for m in results["per_author_metrics"].values() if not np.isnan(m.get("roc_auc", float("nan")))]
    if valid:
        avg_roc = np.mean([m["roc_auc"] for m in valid])
        avg_pr = np.mean([m["avg_precision"] for m in valid])
        avg_f1 = np.mean([m["f1"] for m in valid])
        lines.append("-" * 75)
        lines.append(
            f"{'Macro average':<25} {'':>4} {avg_roc:>8.3f} {avg_pr:>8.3f} {avg_f1:>6.3f}"
        )

    return "\n".join(lines)


def print_results(results: dict) -> None:
    """Pretty-print evaluation results."""
    print(format_results_summary(results))
