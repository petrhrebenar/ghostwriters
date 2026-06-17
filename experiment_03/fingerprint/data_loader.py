"""Load the experiment_03 analysis dataset (self-contained, span-derived).

Replaces experiment_02's ``subset_disent2.csv`` loader. Reads the two tables
produced by ``build_dataset.py`` from annotated spans + scraped metadata:

  * ``data/dataset/dissents.parquet``  — per-judge separate-opinion prose
    (training data). Columns include ``doc_id, author, label, text``.
  * ``data/dataset/decisions.parquet`` — per-decision RATIO text (scoring
    target) + metadata. Columns include ``doc_id, ratio_text,
    judge_rapporteur_name``.

For drop-in compatibility with the ported fingerprint pipeline, the returned
frames expose experiment_02's column names:
  - dissents : ``separate_opinion`` (author label) + ``separate_opinion_extracted`` (text)
  - decisions: ``text`` (= RATIO) + ``judge_rapporteur_name``
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

DATASET_DIR = Path(__file__).resolve().parents[1] / "data" / "dataset"


def _read_table(stem: Path) -> pd.DataFrame:
    """Read ``<stem>.parquet`` if available (and pyarrow present), else ``.csv``."""
    pq = stem.with_suffix(".parquet")
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    return pd.read_csv(stem.with_suffix(".csv"))


def load_dissents(
    path: Optional[Path] = None,
    min_dissents: int = 5,
    labels: tuple[str, ...] = ("DIS", "CON"),
) -> pd.DataFrame:
    """Load separate-opinion training rows, filtered by minimum count per author.

    Parameters
    ----------
    path : Path, optional
        Path to ``dissents.parquet``. Defaults to ``data/dataset/``.
    min_dissents : int
        Minimum opinions an author must have to be kept (0 = keep all).
    labels : tuple
        Which span labels to include (default DIS+CON; pass ("DIS",) to train
        on dissents only).
    """
    df = _read_table(Path(path) if path else DATASET_DIR / "dissents")

    df = df[df["label"].isin(labels)].copy()
    df = df[df["author"].notna() & (df["author"].astype(str).str.strip() != "")]
    df = df[df["author"] != "UNKNOWN"]

    # experiment_02-compatible names
    df["separate_opinion"] = df["author"].astype(str).str.strip()
    df["separate_opinion_extracted"] = df["text"]

    if min_dissents > 0:
        counts = df["separate_opinion"].value_counts()
        keep = counts[counts >= min_dissents].index
        df = df[df["separate_opinion"].isin(keep)]

    return df.reset_index(drop=True)


def load_decisions(
    path: Optional[Path] = None,
    require_ratio: bool = True,
) -> pd.DataFrame:
    """Load per-decision scoring rows (RATIO text + metadata)."""
    df = _read_table(Path(path) if path else DATASET_DIR / "decisions")

    df["text"] = df["ratio_text"]
    if require_ratio:
        df = df[df["text"].astype(str).str.strip() != ""]

    if "date_decision" in df.columns:
        df["date_decision"] = pd.to_datetime(df["date_decision"], errors="coerce")

    return df.reset_index(drop=True)


def author_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-author statistics over a dissents frame (from ``load_dissents``)."""
    stats = (
        df.assign(
            word_count=df["separate_opinion_extracted"].fillna("").str.split().apply(len)
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
    print(f"Loaded {len(df)} opinions from {df['separate_opinion'].nunique()} authors")
    print()
    print(author_summary(df).to_string())
