"""Feature importance extraction and reporting.

Extracts per-author discriminative features from trained classifiers:
- Logistic regression: coefficient weights
- XGBoost: gain-based feature importances
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from fingerprint.classifiers import PerAuthorClassifier


def extract_feature_importance(
    pac: PerAuthorClassifier,
    feature_names: list[str],
    top_k: int = 20,
) -> pd.DataFrame:
    """Extract per-author top-k most discriminative features.

    Parameters
    ----------
    pac : PerAuthorClassifier
        A fitted classifier.
    feature_names : list[str]
        Ordered list of feature names matching the feature matrix columns.
    top_k : int
        Number of top features to report per author.

    Returns
    -------
    pd.DataFrame
        Columns: author, rank, feature, weight/importance, direction.
    """
    rows = []

    if pac.classifier_type == "logistic":
        coefficients = pac.get_coefficients()
        for author in pac.get_author_names():
            coefs = coefficients[author]
            # Sort by absolute value
            top_idx = np.argsort(np.abs(coefs))[::-1][:top_k]
            for rank, idx in enumerate(top_idx, 1):
                rows.append({
                    "author": author,
                    "rank": rank,
                    "feature": feature_names[idx],
                    "weight": float(coefs[idx]),
                    "abs_weight": float(np.abs(coefs[idx])),
                    "direction": "positive" if coefs[idx] > 0 else "negative",
                })

    elif pac.classifier_type == "xgboost":
        importances = pac.get_feature_importances()
        for author in pac.get_author_names():
            imp = importances[author]
            top_idx = np.argsort(imp)[::-1][:top_k]
            for rank, idx in enumerate(top_idx, 1):
                rows.append({
                    "author": author,
                    "rank": rank,
                    "feature": feature_names[idx],
                    "weight": float(imp[idx]),
                    "abs_weight": float(imp[idx]),
                    "direction": "N/A",
                })

    return pd.DataFrame(rows)


def format_feature_importance_report(
    df: pd.DataFrame,
    top_k: int = 10,
) -> str:
    """Format feature importance as a human-readable report.

    Parameters
    ----------
    df : pd.DataFrame
        Output of extract_feature_importance().
    top_k : int
        Number of features to show per author in the report.

    Returns
    -------
    str
        Formatted report.
    """
    lines = ["Feature Importance Report", "=" * 60, ""]

    for author in df["author"].unique():
        author_df = df[df["author"] == author].head(top_k)
        lines.append(f"Author: {author}")
        lines.append(f"  {'Rank':>4}  {'Feature':<30} {'Weight':>10} {'Dir':<10}")
        lines.append(f"  {'-' * 60}")
        for _, row in author_df.iterrows():
            lines.append(
                f"  {row['rank']:>4}  {row['feature']:<30} "
                f"{row['weight']:>10.4f} {row['direction']:<10}"
            )
        lines.append("")

    return "\n".join(lines)
