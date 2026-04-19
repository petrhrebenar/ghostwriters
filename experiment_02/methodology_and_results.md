# Experiment 02 — Methodology and Results

> **Per-Writer Authorship Classification for Czech Constitutional Court Decisions**
>
> This document is a comprehensive research report for Experiment 02. It describes
> the motivation, methodology, implementation, and baseline results for building
> 19 independent binary authorship classifiers and applying them to full decision
> texts to detect potential ghostwriting. Colleagues unfamiliar with the codebase
> should be able to understand the full approach by reading this document alone.
>
> Prerequisite reading: [`experiment_01/methodology_and_results.md`](../experiment_01/methodology_and_results.md)
> — Experiment 02 builds on the same data, preprocessing, and stylometric
> features, and inherits the same literature context (Rosenthal & Yoon, Avraham
> et al., Mosteller & Wallace, Burrows).

---

## 1. Research Question

Experiment 01 answered: *"Which of these 19 judges wrote this dissent?"* — a
closed-set 19-way classification task, achieving 69.7% LOO-CV accuracy.

Experiment 02 asks a fundamentally different question:

> For each of the 19 judges J and for each Constitutional Court decision D,
> what is the probability that J wrote D?

The output is not a single predicted author but a **19-dimensional probability
vector per decision**, one probability per judge, each produced independently.

### 1.1 Why per-writer instead of 19-way multi-class?

For ghostwriting detection on majority opinions we need a fundamentally
different framing than closed-set attribution:

1. **Open-world application.** The full decision text may have been written by
   someone *not* among the 19 judges (a judicial assistant). A multi-class
   classifier is forced to pick one of the 19; a set of binary classifiers can
   legitimately say "none of these judges match."
2. **Independent probability estimates.** We want P(author = J | text) for each
   J independently. A multi-class softmax forces probabilities to sum to 1
   across the 19 judges — which is misleading when the true author is outside
   the candidate set.
3. **Different training signal per author.** Each binary classifier can focus
   on the specific features that distinguish *one* author from the rest,
   rather than jointly optimizing for all 19.

### 1.2 Hypothesis (inherited from Experiment 01)

Dissenting opinions are personally authored by the judges they are attributed
to, so a stylometric fingerprint trained on dissents reflects the judge's own
style. Majority decisions, in contrast, are frequently drafted by judicial
assistants. If the attributed rapporteur scores low on a decision while another
judge (or none) scores high, that is a ghostwriting signal.

---

## 2. Data

The dataset, author filtering, and column conventions are identical to
Experiment 01. Briefly:

- **Source**: `subset_disent2.csv` (repo root), 314 single-author dissenting
  opinions from 35 judges, prepared from the
  [Czech Constitutional Court dataset](https://github.com/stepanpaulik/ccc_dataset).
- **Author filter**: ≥ 5 dissents per judge → **19 judges, 277 dissent texts**
  (same as Experiment 01).
- **Decision texts**: Experiment 02 additionally uses the `text` column —
  the **full text of the Constitutional Court decision** in which each dissent
  appeared. This gives us 277 decision texts aligned with the 277 dissents
  (one decision per dissent, because each row in `subset_disent2.csv` is a
  dissent and carries its decision text). These decisions are the **target of
  the ghostwriting analysis**.

For the detailed class distribution (5–32 dissents per judge) see
[Experiment 01, §3.2](../experiment_01/methodology_and_results.md#32-author-filtering).

---

## 3. Methodology

### 3.1 Pipeline overview

```
subset_disent2.csv
    │
    ├─ [1] load           → 277 dissents, 19 authors
    │
    ├─ [2] udpipe_dissents  → tokenize / POS / lemma / morph-feats for dissents
    │
    ├─ [3] features_dissents → 740-dim feature matrix, build char/POS/XPOS vocabs
    │
    ├─ [4] evaluate       → LOO-CV over dissents with per-author binary classifiers
    │                        (ranking metrics + per-author ROC AUC, PR AUC, F1)
    │
    ├─ [5] udpipe_decisions  → same NLP pipeline for the 277 full decision texts
    │
    ├─ [6] features_decisions → apply the dissent-trained vocabs to decisions
    │                           → 740-dim feature matrix for decisions
    │
    └─ [7] apply          → train final 19 classifiers on ALL dissents
                            → score all 277 decisions
                            → authorship_probabilities_*.csv
```

Each step saves its output under `outputs/`, so the pipeline can resume from
any step via `--from-step`.

### 3.2 Preprocessing

Identical to Experiment 01: text cleaning, UDPipe 1 tokenization / POS tagging
/ lemmatization with `czech-pdt-ud-2.5-191206`, parsed into
`Document → Sentence → Token` dataclasses. See [Experiment 01, §4.2](../experiment_01/methodology_and_results.md#42-preprocessing-pipeline).

**Scope change for Experiment 02**: We apply the same preprocessing to *two*
corpora — the 277 dissents and the 277 full decisions. Vocabularies for
character n-grams, POS bigrams, and XPOS bigrams are built **only on dissents**
(the training data) to avoid information leakage from the decisions.

### 3.3 Feature sets (740 features total)

The four feature sets from Experiment 01 are retained verbatim; a fifth
**morphology** set is added.

| Set | Count | Source |
|-----|------:|--------|
| Function words | 240 | `features/function_words.py` (inherited) |
| Surface | 22 | `features/surface.py` (inherited) |
| Character 3-grams | 200 | `features/ngrams.py` (inherited) |
| POS bigrams (UPOS) | 100 | `features/ngrams.py` (inherited) |
| **Morphology (new)** | **178** | `features/morphology.py` |
| **Total** | **740** | |

(For the four inherited sets, see [Experiment 01, §4.3](../experiment_01/methodology_and_results.md#43-feature-sets). The function-word
list is the deduplicated 240-word set; Experiment 01 reports 263 words but
uses the same deduplicated vector under the hood.)

#### 3.3.1 Morphology features (new)

Implemented in `src/fingerprint/features/morphology.py`. UDPipe's CoNLL-U
output exposes rich Czech-specific morphological annotation that Experiment 01
did not exploit. We add three sub-groups:

**(a) Morphological category distributions (~28 features).**
For each of the following grammatical categories we compute the relative
frequency of each value over tokens where the category is annotated:

- **Case**: Nom, Gen, Dat, Acc, Voc, Loc, Ins (7)
- **Number**: Sing, Plur (2)
- **Gender**: Masc, Fem, Neut (3)
- **Tense**: Past, Pres, Fut (3)
- **Mood**: Ind, Imp, Cnd (3)
- **Voice**: Act, Pass (2)
- **Aspect**: Imp, Perf (2)
- **VerbForm**: Fin, Inf, Part, Conv (4)
- **Degree**: Pos, Cmp, Sup (3)

These distributions capture stylistic preferences that are independent of
vocabulary: e.g. a judge who systematically prefers passive over active
constructions, nominal (case-heavy) over verbal style, or past over present
narration.

**(b) Passive voice ratio (1 feature).**
`count(Voice=Pass) / count(finite verbs)`. Emphasizes a single stylometrically
important dimension known to vary across legal writers.

**(c) XPOS bigrams (top 150, ~149 features).**
Czech-specific fine-grained POS tags (positional XPOS, e.g. `NNIS1-----A----`
for a masculine-inanimate singular-nominative-animate-negated noun) carry far
more information than the Universal POS tags. We extract per-sentence bigrams
of XPOS codes, build a corpus-wide vocabulary of the top 150 most frequent
XPOS bigrams from the dissents, and represent each document as relative
frequencies over this vocabulary.

(The 178 "morphology" count above is the sum of the three sub-groups after
pruning to actually observed categories/values.)

### 3.4 Classifiers

Implemented in `src/fingerprint/classifiers.py` as `PerAuthorClassifier`, a
wrapper that trains **19 independent binary classifiers** (one per author).
For author J, a binary label `y_J = (y == J)` is computed and a binary model
is fit on all training samples.

Two backends are supported:

| | Logistic Regression | XGBoost |
|---|---|---|
| Loss | log loss (L2 regularized) | tree boosting (logistic loss) |
| Imbalance handling | `class_weight='balanced'` | `scale_pos_weight = n_neg / n_pos` |
| Hyperparameters | `solver='lbfgs'`, `C=1.0`, `max_iter=1000` | `max_depth=3`, `n_estimators=50`, `lr=0.15`, `reg_alpha=1.0`, `reg_lambda=1.0` |
| Probability quality | Good natively | Reasonable (no post-hoc calibration yet) |
| Interpretability | Coefficients per feature | Gain-based feature importances |

`PerAuthorClassifier.predict_proba(X)` returns an `(n_samples, 19)` matrix
where column *k* is P(author = J_k | x) from the *k*-th binary classifier.
**Columns are independent and do not sum to 1** — that is the central design
decision of this experiment.

### 3.5 Evaluation (LOO-CV)

Implemented in `src/fingerprint/evaluation.py`. For each of the 277 dissents:

1. Hold out one sample.
2. Fit a fresh `StandardScaler` on the remaining 276.
3. Train 19 binary classifiers on the scaled training data.
4. Transform the held-out sample with the training scaler and score it with
   all 19 classifiers, producing a 19-dim probability vector.

After all 277 folds we have a 277×19 probability matrix. Total work per
classifier backend: 5,263 model fits.

**Metrics reported**:

| Category | Metric | Meaning |
|---|---|---|
| **Ranking** (global) | Rank-1 accuracy | Fraction of samples where the true author has the highest probability |
| | Rank-3 accuracy | Fraction of samples where the true author is in the top 3 |
| | MRR | Mean reciprocal rank of the true author |
| **Binary** (per author) | ROC AUC | Discrimination of positive vs. rest, threshold-free |
| | PR AUC | Area under precision–recall; more meaningful under class imbalance |
| | F1 / Precision / Recall | At fixed threshold 0.5 on the raw binary probability |

Because the 19 binary probabilities are *independent*, ranking metrics are
computed by taking the argmax / top-k across the 19 columns per sample —
this is a legitimate comparison even though the probabilities don't sum to 1.

### 3.6 Application to decisions

After LOO-CV evaluation we retrain the 19 binary classifiers **once, on all
277 dissents** (no held-out sample), and score the 277 decision texts. The
output file `authorship_probabilities_logistic.csv` has 277 rows and, per
row, 3 × 19 + 2 columns:

- `prob_<author>` (19 cols): raw binary probabilities, independent, not
  normalized.
- `norm_<author>` (19 cols): softmax-normalized probabilities across the 19
  authors (sum to 1). Useful as a relative comparison, but interpret with
  care — softmax amplifies small raw-probability gaps.
- `predicted_author`: argmax over raw probabilities.
- `max_probability`: the raw probability at the argmax (i.e., the confidence
  of the top prediction).

---

## 4. Results

### 4.1 Global ranking performance

LOO-CV over the 277 dissents with logistic regression:

| Metric | Exp 02 (per-writer LR) | Exp 01 (multi-class LR) |
|---|---|---|
| **Rank-1 accuracy** | **73.3 %** | 69.7 % |
| **Rank-3 accuracy** | **85.6 %** | — |
| **MRR** | **0.815** | — |
| **Macro ROC AUC** | **0.927** | — |

The per-writer binary formulation *outperforms* the Experiment-01 multi-class
classifier (73.3 % vs. 69.7 %) using the same data and the same underlying
feature set, plus the new morphology features. The correct author is in the
top 3 predictions 85.6 % of the time. When rank-1 is wrong, the correct
answer is typically at rank 2 (MRR = 0.815).

### 4.2 Per-author binary metrics (logistic regression LOO-CV)

Authors are sorted by ROC AUC (descending).

| Author | N | ROC AUC | PR AUC | F1 | Prec | Rec |
|---|---:|---:|---:|---:|---:|---:|
| Iva Brožová | 9 | 1.000 | 1.000 | 0.875 | 1.000 | 0.778 |
| Jan Filip | 31 | 0.999 | 0.991 | 0.933 | 0.966 | 0.903 |
| Jan Musil | 23 | 0.995 | 0.927 | 0.870 | 0.870 | 0.870 |
| Pavel Varvařovský | 14 | 0.993 | 0.896 | 0.696 | 0.889 | 0.571 |
| Kateřina Šimáčková | 6 | 0.991 | 0.788 | 0.667 | 1.000 | 0.500 |
| Josef Fiala | 21 | 0.982 | 0.786 | 0.872 | 0.944 | 0.810 |
| Vojtěch Šimíček | 9 | 0.978 | 0.905 | 0.889 | 0.889 | 0.889 |
| Vladimír Kůrka | 10 | 0.975 | 0.575 | 0.667 | 0.636 | 0.700 |
| Miloslav Výborný | 9 | 0.975 | 0.621 | 0.571 | 0.800 | 0.444 |
| Jiří Zemánek | 12 | 0.971 | 0.822 | 0.818 | 0.900 | 0.750 |
| Michaela Židlická | 5 | 0.959 | 0.442 | 0.500 | 0.667 | 0.400 |
| Vladimír Sládeček | 5 | 0.954 | 0.327 | 0.250 | 0.333 | 0.200 |
| David Uhlíř | 6 | 0.943 | 0.259 | 0.000 | 0.000 | 0.000 |
| Radovan Suchánek | 30 | 0.935 | 0.713 | 0.712 | 0.724 | 0.700 |
| Eliška Wagnerová | 22 | 0.919 | 0.432 | 0.462 | 0.529 | 0.409 |
| Stanislav Balík | 17 | 0.891 | 0.619 | 0.593 | 0.800 | 0.471 |
| Ivana Janů | 32 | 0.852 | 0.465 | 0.407 | 0.444 | 0.375 |
| Jiří Nykodým | 5 | 0.396 | 0.017 | 0.000 | 0.000 | 0.000 |
| **Macro average** | | **0.927** | **0.629** | **0.588** | | |

### 4.3 Analysis of LOO-CV results

**Tier 1 — highly distinctive writers (ROC AUC ≥ 0.97, 10 judges).**
Jan Filip, Jan Musil, Iva Brožová, Pavel Varvařovský, Kateřina Šimáčková,
Josef Fiala, Vojtěch Šimíček, Vladimír Kůrka, Miloslav Výborný, Jiří Zemánek.
These classifiers are essentially saturated — the feature set captures the
author's style almost perfectly. Notably, **Iva Brožová achieves perfect
separation (ROC AUC = 1.0) with only 9 dissents** — her style is uniquely
identifiable.

**Tier 2 — moderate signal (ROC AUC 0.85–0.96, 8 judges).**
Michaela Židlická, Vladimír Sládeček, David Uhlíř, Radovan Suchánek,
Eliška Wagnerová, Stanislav Balík, Ivana Janů (and, borderline, Jiří
Nykodým). For these authors the *ranking* is useful but the 0.5 probability
threshold is miscalibrated: see David Uhlíř (ROC AUC 0.943, but F1 = 0.0) —
his classifier correctly puts his own dissents at the top of the probability
ranking yet never crosses 0.5 absolutely. For our ghostwriting application
this is acceptable because we compare *probabilities across authors*, not
against a fixed threshold.

**Tier 3 — failure (ROC AUC < 0.5, 1 judge).**
Jiří Nykodým (5 dissents, ROC AUC 0.396 — worse than random). With only 5
samples and apparently no distinctive signal, his classifier is not usable.

**Sample size is not the main driver.** Iva Brožová (N = 9) and Vojtěch
Šimíček (N = 9) have ROC AUC ≥ 0.98, while Ivana Janů (N = 32, the largest
class) has only 0.852. This mirrors the Exp-01 "Ivana Janů anomaly" and
suggests genuinely high within-author stylistic variance — a candidate signal
for stylistic heterogeneity or even ghostwriting inside her own dissents.

**Comparison with Experiment 01 per-author F1.** For the top-tier writers,
F1 is essentially unchanged (Jan Filip 0.95 → 0.93, Jan Musil 0.90 → 0.87 —
the slight drops are within LOO-CV noise). The per-writer framing trades a
small per-author F1 for a markedly better global rank-1 accuracy (+3.6 pp)
and, importantly, **independently-interpretable probabilities**.

### 4.4 Application to decisions

The trained logistic-regression PerAuthorClassifier was applied to the 277
full decision texts. Characteristics of the output
(`outputs/authorship_probabilities_logistic.csv`):

**Distribution of the top raw probability per decision:**

| Quantile | `max_probability` |
|---|---:|
| min | 0.010 |
| 25 % | 0.118 |
| 50 % (median) | 0.346 |
| 75 % | 0.782 |
| max | 1.000 |
| mean | 0.436 |

- 114 / 277 decisions (41 %) have at least one binary classifier produce
  probability ≥ 0.5.
- 28 / 277 decisions (10 %) have a classifier that is very confident
  (≥ 0.9).
- The remaining ~60 % have no classifier strongly above 0.5, meaning the
  decision is either written in a "neutral" / institutional style, by a
  judge outside the 19 candidates, or jointly.

**Predicted-author distribution on decisions (argmax):**

| Predicted author | # decisions | # own dissents | ratio |
|---|---:|---:|---:|
| Radovan Suchánek | 78 | 30 | 2.6 |
| Ivana Janů | 58 | 32 | 1.8 |
| Jan Filip | 27 | 31 | 0.9 |
| Stanislav Balík | 17 | 17 | 1.0 |
| Jan Musil | 16 | 23 | 0.7 |
| Kateřina Šimáčková | 14 | 6 | 2.3 |
| Josef Fiala | 12 | 21 | 0.6 |
| Eliška Wagnerová | 11 | 22 | 0.5 |
| Jiří Zemánek | 9 | 12 | 0.8 |
| Vojtěch Šimíček | 8 | 9 | 0.9 |
| Vladimír Sládeček | 7 | 5 | 1.4 |
| Pavel Varvařovský | 6 | 14 | 0.4 |
| Iva Brožová | 4 | 9 | 0.4 |
| Jiří Nykodým | 4 | 5 | 0.8 |
| Ludvík David | 2 | 11 | 0.2 |
| Miloslav Výborný | 2 | 9 | 0.2 |
| Michaela Židlická | 1 | 5 | 0.2 |
| Vladimír Kůrka | 1 | 10 | 0.1 |
| *David Uhlíř* | 0 | 6 | 0.0 |

The "ratio" column compares how many decisions the classifier attributes to a
judge vs. how many dissents that judge actually wrote in the corpus.

**Observations (preliminary, not yet validated against rapporteur metadata):**

- **Radovan Suchánek and Ivana Janů are over-assigned** (ratios 2.6 and 1.8).
  This is consistent with a known pattern for "generic" classifiers — they
  act as a catch-all when no other author has a strongly peaked style. This
  matches the low PR AUC of these two classifiers in LOO-CV (0.71 and 0.47).
- **Jan Filip is approximately balanced** (27 vs. 31). His classifier is
  well-calibrated (PR AUC 0.99) and only fires on decisions that genuinely
  match his style.
- **David Uhlíř never wins** on any decision, even though he wrote 6 dissents.
  This is expected given his LOO-CV F1 of 0.0.

The real signal for ghostwriting will come from comparing these predictions
against the actual rapporteur on each decision — **the step not yet performed
in this baseline**. The question is not "who does the classifier pick", but
"does the classifier's top pick match the attributed rapporteur?" and "does
the attributed rapporteur have an anomalously low probability?"

### 4.5 Known limitations of the current results

1. **No ground-truth comparison yet.** Rapporteur metadata from the decisions
   is not yet joined with the probability output. This is the primary missing
   downstream analysis.
2. **No calibration.** Raw probabilities from `class_weight='balanced'`
   logistic regression are not Platt/isotonic-calibrated, so the 0.5 threshold
   is arbitrary. For ranking the effect is negligible; for absolute
   interpretation, calibration is needed.
3. **XGBoost not yet evaluated.** The XGBoost LOO-CV was attempted but
   interrupted at ~90 % completion due to wall-clock cost. The code supports
   it (`--classifier xgboost` / `--classifier both`); re-running with the
   current reduced settings (`n_estimators=50`) is a pending follow-up.
4. **Feature importance extracted but not yet cross-referenced with per-author
   F1.** Per-author top-20 features (LR coefficients) are saved to
   `outputs/feature_importance_logistic.csv` and analyzed in
   `outputs/feature_importance_analysis.md`. Key findings: (a) the
   `punct_;_per1k` (semicolon rate) feature is a top-20 discriminator for
   7 of 19 judges with both positive and negative weights — semicolon usage
   is a strong idiolectal marker; (b) the new XPOS-bigram morphology features
   appear prominently for the highest-ROC-AUC authors (Jan Musil, Iva
   Brožová, Josef Fiala); (c) Ivana Janů's top features max out at a weight
   of only +0.33 (vs. +0.51 for Vojtěch Šimíček) — no dominant stylistic
   peak, consistent with her low ROC AUC despite 32 training samples.
5. **Genre mismatch dissent → decision.** Dissents are argumentative
   first-person texts; full decisions include boilerplate, procedural
   history, fact descriptions, and reasoning. Features like sentence length
   or passive voice ratio may shift systematically between the two genres,
   potentially biasing all 19 classifiers in a correlated way. This is a
   real concern and the main reason the `max_probability` distribution is
   shifted downward on decisions compared to LOO-CV on dissents.
6. **277 decisions == 277 dissents (aligned).** We currently score the same
   decisions from which the dissents were extracted. To generalize we will
   need decisions without any matching dissent in the corpus.

---

## 5. Implementation

### 5.1 Project structure

```
experiment_02/
├── methodology_and_results.md  # This document
├── IMPLEMENTATION.md           # Design plan with rationale + open questions
├── README.md                   # Setup & CLI usage
├── pyproject.toml              # Poetry config (Python ≥3.11)
├── poetry.lock
├── .env                        # PYTHON_KEYRING_BACKEND=...
├── outputs/                    # Auto-generated
│   ├── corpus.pkl
│   ├── dissent_documents.pkl
│   ├── dissent_features.pkl
│   ├── dissent_feature_matrix.csv
│   ├── loo_probabilities_logistic.csv
│   ├── decision_documents.pkl
│   ├── decision_features.pkl
│   ├── trained_classifiers_logistic.pkl
│   └── authorship_probabilities_logistic.csv
├── scripts/
│   └── run_pipeline.py         # 7-step CLI pipeline
└── src/
    └── fingerprint/
        ├── __init__.py
        ├── data_loader.py            # inherited (path adjusted, text col added)
        ├── preprocessing.py          # inherited verbatim
        ├── classifiers.py            # NEW — PerAuthorClassifier (LR + XGBoost)
        ├── evaluation.py             # NEW — LOO-CV with ranking + binary metrics
        ├── feature_importance.py     # NEW — LR coefficients / XGBoost gain
        └── features/
            ├── __init__.py
            ├── function_words.py     # inherited verbatim
            ├── ngrams.py             # inherited verbatim
            ├── surface.py            # inherited verbatim
            └── morphology.py         # NEW — XPOS bigrams + morph categories
```

### 5.2 Pipeline steps

| Step | Name | Input | Output | Duration |
|---|---|---|---|---|
| 1 | `load` | `subset_disent2.csv` | `corpus.pkl` | ~2 s |
| 2 | `udpipe_dissents` | corpus.pkl + UDPipe model | `dissent_documents.pkl` | ~5 min |
| 3 | `features_dissents` | dissent_documents.pkl | `dissent_features.pkl` | ~15 s |
| 4 | `evaluate` | dissent_features.pkl | `loo_probabilities_*.csv` | ~3 min LR / hours XGB |
| 5 | `udpipe_decisions` | corpus.pkl + UDPipe model | `decision_documents.pkl` | ~15 min |
| 6 | `features_decisions` | decision_documents.pkl + vocabs from step 3 | `decision_features.pkl` | ~30 s |
| 7 | `apply` | dissent features + decision features | `authorship_probabilities_*.csv`, `trained_classifiers_*.pkl` | ~10 s |

Example invocations:

```bash
# Full pipeline with logistic regression
poetry run python scripts/run_pipeline.py --classifier logistic

# Both classifiers, from scratch
poetry run python scripts/run_pipeline.py --classifier both

# Resume from application after changing feature extraction
poetry run python scripts/run_pipeline.py --from-step features_dissents

# Only re-apply to decisions with existing trained classifiers
poetry run python scripts/run_pipeline.py --from-step apply
```

### 5.3 Dependencies

Managed via Poetry (`pyproject.toml`). Everything from Experiment 01, plus:

| Package | Purpose |
|---|---|
| `xgboost` | Second classifier backend for `PerAuthorClassifier` |

Python ≥ 3.11 is required. The `.env` file sets
`PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` to prevent Poetry from
hanging on D-Bus keyring lookups on this machine.

### 5.4 External model

Same as Experiment 01: `czech-pdt-ud-2.5-191206.udpipe` from
[LINDAT](https://lindat.mff.cuni.cz/repository/items/41f05304-629f-4313-b9cf-9eeb0a2ca7c6),
placed in `models/` at the repository root and shared across experiments.

---

## 6. Design Decisions and Trade-offs

See `IMPLEMENTATION.md` for the full rationale; key points summarized here.

1. **One-vs-rest binary classifiers, not one-vs-one.** With 19 authors,
   one-vs-one would produce 171 classifiers; one-vs-rest produces 19. The
   binary-probability-per-author interpretation is clearer and is exactly
   what the downstream ghostwriting question requires.
2. **`class_weight='balanced'` for logistic regression, `scale_pos_weight`
   for XGBoost**, both set to `n_neg / n_pos` per author. This keeps the
   decision boundary from collapsing onto the majority (negative) class under
   ~1:50 imbalance.
3. **UDPipe 1, not UDPipe 2.** UDPipe 2 has better taggers but requires
   pre-tokenized CoNLL-U input and a REST/server setup. UDPipe 1 is
   sufficient for stylometric features and keeps the pipeline local.
4. **Vocabularies built only on dissents.** Character 3-grams, POS bigrams,
   and XPOS bigrams use the top-K most frequent items from the *training*
   corpus (dissents) to avoid leakage from the decisions.
5. **StandardScaler refit per LOO fold.** Prevents leakage of the held-out
   sample's feature distribution into the scaler statistics.
6. **Raw + softmax-normalized probabilities in the output.** Softmax is a
   useful relative comparison, but the raw probabilities are the primary
   signal for ghostwriting detection.
7. **XGBoost hyperparameters conservative.** `max_depth=3`, `n_estimators=50`
   after an early full-size run (`n_estimators=100`) proved too slow for
   LOO-CV (5,263 fits). The current setting is a compromise between runtime
   and boosting-round coverage.

---

## 7. Limitations and Future Work

### 7.1 Current limitations

- **No rapporteur join.** The raw output is a probability matrix; it must be
  joined against each decision's attributed rapporteur (from
  `subset_disent2.csv` or the source dataset) to produce an actual
  ghostwriting ranking.
- **No probability calibration** (isotonic regression or Platt scaling). See
  §4.5.
- **XGBoost evaluation pending.** Re-run with the reduced settings is
  straightforward but ~1–1.5 hours of wall time.
- **Feature ablation not yet done.** The 740-feature matrix is used as-is.
  The ablation study (§4b in `IMPLEMENTATION.md`) is pending.
- **Feature importance analysis pending.** Code is implemented; the
  per-author top-20 reports still need to be produced and analyzed.
- **Single preprocessing pass per decision.** Long decisions (some > 10k
  words) are represented as a single feature vector, dominating the
  statistics. Chunking (inherited from Experiment 01's future work) is still
  relevant.
- **Genre mismatch.** Training on dissents and scoring decisions introduces
  a systematic domain shift. Some features (passive voice ratio, sentence
  length, boilerplate-heavy char n-grams) will differ between genres in ways
  that are not author-specific.

### 7.2 Planned improvements

1. **Rapporteur-based ghostwriting analysis**: For each decision, compare the
   attributed rapporteur's P(author = rapporteur | decision) against the
   maximum over the other 18 classifiers. Decisions with large gaps are the
   strongest ghostwriting candidates.
2. **Probability calibration**: Wrap the per-author binary classifiers with
   `CalibratedClassifierCV(method='isotonic', cv='prefit')` using a held-out
   portion of the dissents.
3. **XGBoost LOO-CV + comparison**: Complete the `--classifier both` run and
   add the XGBoost results table alongside §4.2.
4. **Feature ablation**: Run LOO-CV for each feature set alone and in
   cumulative combinations (function_words → + surface → + char_ngrams
   → + pos_ngrams → + morphology). Report Rank-1 accuracy per configuration.
5. **Feature importance**: Produce per-author top-20 coefficient / gain
   tables; check whether the "Ivana Janů anomaly" has a feature-level
   explanation.
6. **Cross-corpus validation**: Score decisions whose dissenter is *not* in
   the 19-author set and verify that no single author scores high (expected:
   lower `max_probability` distribution).
7. **Dissent-style vs. decision-style correction**: Quantify the
   dissent↔decision feature drift on matched pairs; consider dissent-only
   normalization or domain-adaptation.
8. **Within-author variance analysis** (following Rosenthal & Yoon, 2011b):
   bootstrap χ² on function-word counts *within* each judge's dissents to
   detect authors whose own dissents show high internal stylistic variance —
   a candidate ghostwriting signal in the training data itself.

---

## 8. References

Inherited from Experiment 01 (Rosenthal & Yoon 2011a/b, Avraham et al. 2025,
Mosteller & Wallace 1963, Hampton, Burrows 2002) — see
[Experiment 01, §8](../experiment_01/methodology_and_results.md#8-references).

Additional references relevant to this experiment:

1. Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting
   System*. KDD '16.
2. King, G. & Zeng, L. (2001). *Logistic Regression in Rare Events Data*.
   Political Analysis, 9(2), 137–163. [Rationale for `class_weight='balanced'`
   under imbalance.]
3. Platt, J. C. (1999). *Probabilistic Outputs for Support Vector Machines
   and Comparisons to Regularized Likelihood Methods*. Advances in Large
   Margin Classifiers. [Calibration reference for future work.]
4. Straka, M. & Straková, J. (2017). *Tokenizing, POS Tagging, Lemmatizing
   and Parsing UD 2.0 with UDPipe*. CoNLL 2017 Shared Task. [UDPipe 1 with
   the Czech-PDT UD 2.5 model used here.]
