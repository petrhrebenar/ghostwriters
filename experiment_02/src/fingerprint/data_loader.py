"""Load and filter the dissent dataset."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# CSV contains very long HTML text fields
csv.field_size_limit(sys.maxsize)

DATA_PATH = Path(__file__).resolve().parents[3] / "subset_disent2.csv"


def load_dissents(
    path: Optional[Path] = None,
    min_dissents: int = 5,
) -> pd.DataFrame:
    """Load subset_disent2.csv and optionally filter by minimum dissent count.

    Parameters
    ----------
    path : Path, optional
        Path to subset_disent2.csv. Defaults to repo-level file.
    min_dissents : int
        Minimum number of dissents an author must have to be included.
        Set to 0 to include all authors.

    Returns
    -------
    pd.DataFrame
        Columns: doc_id, text, date_decision, date_submission, type_decision,
        separate_opinion, formation, length_proceeding,
        separate_opinion_extracted
    """
    if path is None:
        path = DATA_PATH

    df = pd.read_csv(path, engine="python")

    # Basic cleaning
    df["separate_opinion"] = df["separate_opinion"].astype(str).str.strip()
    df = df[df["separate_opinion"].notna() & (df["separate_opinion"] != "")]

    if "date_decision" in df.columns:
        df["date_decision"] = pd.to_datetime(df["date_decision"], errors="coerce")

    # Filter by minimum dissent count
    if min_dissents > 0:
        counts = df["separate_opinion"].value_counts()
        keep = counts[counts >= min_dissents].index
        df = df[df["separate_opinion"].isin(keep)]

    df = df.reset_index(drop=True)
    return df


def author_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-author statistics."""
    stats = (
        df.assign(
            word_count=df["separate_opinion_extracted"]
            .fillna("")
            .str.split()
            .apply(len)
        )
        .groupby("separate_opinion")
        .agg(
            n_dissents=("doc_id", "count"),
            avg_words=("word_count", "mean"),
            median_words=("word_count", "median"),
            min_words=("word_count", "min"),
            max_words=("word_count", "max"),
            total_words=("word_count", "sum"),
        )
        .sort_values("n_dissents", ascending=False)
    )
    return stats


if __name__ == "__main__":
    df = load_dissents(min_dissents=5)
    print(f"Loaded {len(df)} dissents from {df['separate_opinion'].nunique()} authors")
    print()
    print(author_summary(df).to_string())
