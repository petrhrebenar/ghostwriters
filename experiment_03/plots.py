#!/usr/bin/env python3
"""Visualizations for the authorship / ghostwriting analysis.

Produces:
  1. fig_validation_confusion.png  — LOO true vs predicted dissent author
     (the validity gate: must be diagonal-heavy before trusting RATIO scores).
  2. fig_rapporteur_x_author.png   — corpus matrix, rows = trained rapporteurs,
     cols = judge classifiers, cell = mean normalized RATIO probability. A strong
     diagonal = judges write their own reasoning; off-diagonal = style attributed
     elsewhere (ghostwriting candidate, or genre transfer — see report).
  3. fig_selfattr_by_section.png   — median self-attribution per section type:
     DIS (in-genre, known author) vs RATIO/FACT/... (cross-genre). Separates
     genre-transfer from ghostwriting.
  4. cards/<doc_id>.png            — per-decision section x author heatmap
     (all decisions). Rapporteur column boxed; known dissent author marked on
     DIS/CON rows; short (<MIN_WORDS) sections greyed as unreliable.

Run from experiment_03/ with the analysis env:
    poetry run python plots.py --classifier logistic
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
OUT = EXP_DIR / "outputs"
CARDS = OUT / "cards"

LABEL_ORDER = ["HEAD", "REC", "PART", "PROC", "FACT", "RATIO", "DISP", "DIS", "CON"]
MIN_WORDS = 200  # below this, a section is too short for reliable stylometry


# ── 1. validation confusion (recompute LOO, deterministic) ──────────────
def validation_confusion(clf: str) -> None:
    from fingerprint.evaluation import loo_per_author_evaluation
    d = pickle.load(open(OUT / "dissent_features.pkl", "rb"))
    res = loo_per_author_evaluation(d["X"], d["y"], classifier_type=clf)
    authors = list(res["authors"])
    true = np.asarray(res["true_labels"])
    pred = np.array([authors[i] for i in np.argmax(res["probabilities"], axis=1)])

    idx = {a: i for i, a in enumerate(authors)}
    cm = np.zeros((len(authors), len(authors)))
    for t, p in zip(true, pred):
        cm[idx[t], idx[p]] += 1
    row = cm.sum(1, keepdims=True)
    cmn = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cmn, annot=cm.astype(int), fmt="d", cmap="Blues", vmin=0, vmax=1,
                xticklabels=authors, yticklabels=authors, ax=ax,
                cbar_kws={"label": "row-normalized (recall)"})
    acc = (true == pred).mean()
    ax.set_title(f"LOO validation — dissent author confusion ({clf})\n"
                 f"rank-1 accuracy = {acc:.3f}  (n={len(true)}, {len(authors)} judges)")
    ax.set_xlabel("predicted author"); ax.set_ylabel("true author")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    fig.savefig(OUT / "fig_validation_confusion.png", dpi=130); plt.close(fig)
    print("  -> fig_validation_confusion.png")


# ── helpers for normalized prob matrices ────────────────────────────────
def _prob_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("prob_") and c != "prob_rapporteur"]


def _row_normalize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    m = df[cols].to_numpy(dtype=float)
    e = np.exp(m - m.max(axis=1, keepdims=True))
    return pd.DataFrame(e / e.sum(axis=1, keepdims=True), columns=cols, index=df.index)


# ── 2. corpus rapporteur x author (RATIO) ───────────────────────────────
def corpus_matrix(clf: str) -> None:
    df = pd.read_csv(OUT / f"authorship_probabilities_{clf}.csv")
    df = df[df["rapporteur_is_trained_author"] == True].copy()  # noqa: E712
    cols = _prob_cols(df)
    authors = [c[len("prob_"):] for c in cols]
    # raw mean per-author probability over each rapporteur's RATIOs (independent
    # one-vs-rest classifiers; raw prob is the honest "fingerprint match strength").
    g = df.copy()
    g["rapporteur"] = df["judge_rapporteur"].values
    mat = g.groupby("rapporteur")[cols].mean()
    mat = mat.reindex([a for a in authors if a in mat.index])  # order rows like cols
    mat.columns = authors

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(mat, cmap="magma", vmin=0, vmax=1,
                xticklabels=authors, yticklabels=mat.index, ax=ax,
                cbar_kws={"label": "mean P(author | RATIO)"})
    # box the diagonal (rapporteur == predicted author)
    for i, rap in enumerate(mat.index):
        if rap in authors:
            j = authors.index(rap)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="lime", lw=2))
    ax.set_title(f"Corpus: RATIO attribution by rapporteur ({clf})\n"
                 f"green box = self (rapporteur's own column); n={len(df)} decisions")
    ax.set_xlabel("predicted author (fingerprint)"); ax.set_ylabel("rapporteur")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    fig.savefig(OUT / "fig_rapporteur_x_author.png", dpi=130); plt.close(fig)
    print("  -> fig_rapporteur_x_author.png")


# ── 3. self-attribution by section type ─────────────────────────────────
def selfattr_by_section(clf: str) -> None:
    """Match rate of each section to its APPROPRIATE known author:
       court-voice sections (HEAD..DISP) -> rapporteur; DIS/CON -> dissent author.
    DIS/CON is the in-genre control (same genre as training); RATIO is the
    cross-genre test. This is why DIS is high and RATIO low."""
    sec = pd.read_csv(OUT / f"section_scores_{clf}.csv")
    OP = {"DIS", "CON"}
    rows = []
    for lab in LABEL_ORDER:
        s = sec[sec["label"] == lab]
        if not len(s):
            continue
        if lab in OP:  # match predicted author to a listed dissent judge
            def _hit(r):
                judges = [j.strip() for j in str(r.separate_opinion or "").split(";") if j.strip()]
                return _surname_match(r.predicted_author, judges) is not None
            valid = s[s["separate_opinion"].notna()]
            rate = valid.apply(_hit, axis=1).mean() if len(valid) else float("nan")
            rows.append({"section": lab, "n": len(valid), "ref": "dissent author",
                         "match_rate": rate})
        else:  # court voice -> rapporteur
            v = s[s["prob_rapporteur"].notna()]
            rate = (v["predicted_author"] == v["judge_rapporteur_name"]).mean() if len(v) else float("nan")
            rows.append({"section": lab, "n": len(v), "ref": "rapporteur",
                         "match_rate": rate})
    d = pd.DataFrame(rows)
    palette = {"rapporteur": "#2c7fb8", "dissent author": "#d62728"}
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(d, x="section", y="match_rate", hue="ref", palette=palette, dodge=False, ax=ax)
    for i, r in d.iterrows():
        ax.text(i, r["match_rate"] + 0.01, f"{r['match_rate']:.0%}\nn={r['n']}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Attribution to the correct known author, by section type ({clf})\n"
                 "DIS/CON (red) = dissent author = IN-GENRE control;  RATIO (blue) = rapporteur = CROSS-GENRE test")
    ax.set_ylabel("P(predicted author == known author)")
    ax.legend(title="scored against", loc="upper left")
    plt.tight_layout()
    fig.savefig(OUT / "fig_selfattr_by_section.png", dpi=130); plt.close(fig)
    print("  -> fig_selfattr_by_section.png")
    d.to_csv(OUT / "selfattr_by_section.csv", index=False)


# ── 4. per-decision section x author cards ──────────────────────────────
def _surname_match(name: str, candidates: list[str]) -> str | None:
    if not isinstance(name, str):
        return None
    key = name.split()[-1].lower()
    for c in candidates:
        if c.split()[-1].lower() == key:
            return c
    return None


def section_cards(clf: str) -> None:
    sec = pd.read_csv(OUT / f"section_scores_{clf}.csv")
    prob_cols = [c for c in sec.columns if c.startswith("prob_") and c != "prob_rapporteur"]
    authors = [c[len("prob_"):] for c in prob_cols]
    CARDS.mkdir(parents=True, exist_ok=True)
    n = 0
    for doc_id, g in sec.groupby("doc_id"):
        g = g.assign(_o=g["label"].map({l: i for i, l in enumerate(LABEL_ORDER)})).sort_values("_o")
        M = g[prob_cols].to_numpy(dtype=float)
        rlabels = [f"{r.label} ({int(r.words)}w)" for r in g.itertuples()]
        rap = g["judge_rapporteur_name"].iloc[0]
        sep = str(g["separate_opinion"].iloc[0] or "")
        sep_judges = [s.strip() for s in sep.split(";") if s.strip()]

        fig, ax = plt.subplots(figsize=(max(8, len(authors) * 0.5), max(3, len(g) * 0.5 + 1)))
        sns.heatmap(M, cmap="magma", vmin=0, vmax=1, xticklabels=authors,
                    yticklabels=rlabels, ax=ax, cbar_kws={"label": "P(author | section)"})
        # grey out short sections
        for i, w in enumerate(g["words"].to_numpy()):
            if w < MIN_WORDS:
                ax.add_patch(plt.Rectangle((0, i), len(authors), 1, fill=True,
                                           color="white", alpha=0.45, zorder=3))
        # rapporteur column (boxed) + known dissent authors on DIS/CON rows
        rap_m = _surname_match(rap, authors)
        if rap_m:
            j = authors.index(rap_m)
            ax.add_patch(plt.Rectangle((j, 0), 1, len(g), fill=False, edgecolor="lime", lw=2))
        for i, r in enumerate(g.itertuples()):
            if r.label in ("DIS", "CON"):
                for jm in (_surname_match(s, authors) for s in sep_judges):
                    if jm:
                        ax.add_patch(plt.Rectangle((authors.index(jm), i), 1, 1, fill=False,
                                                   edgecolor="cyan", lw=2))
        title = f"{doc_id} — rapporteur: {rap or '?'}"
        if sep_judges:
            title += f"  |  dissent: {', '.join(sep_judges)}"
        ax.set_title(title + "\n(green=rapporteur col, cyan=known dissent author, grey=short/unreliable)",
                     fontsize=9)
        plt.xticks(rotation=45, ha="right", fontsize=8); plt.yticks(fontsize=8)
        plt.tight_layout()
        fig.savefig(CARDS / f"{doc_id}.png", dpi=110); plt.close(fig)
        n += 1
    print(f"  -> {n} cards in {CARDS}/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classifier", default="logistic")
    ap.add_argument("--only", nargs="+", default=["validation", "corpus", "sections", "cards"],
                    choices=["validation", "corpus", "sections", "cards"])
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if "validation" in args.only: validation_confusion(args.classifier)
    if "corpus" in args.only: corpus_matrix(args.classifier)
    if "sections" in args.only: selfattr_by_section(args.classifier)
    if "cards" in args.only: section_cards(args.classifier)
    print("Done.")


if __name__ == "__main__":
    main()
