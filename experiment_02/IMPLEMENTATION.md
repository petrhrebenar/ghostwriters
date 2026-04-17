# Experiment 02 — Per-Writer Classification

## 0. Original Prompt

> alright. yes, the idea for experiment_02
> - instead of choosing 1 of the 19 authors we need to build classifier for each author separately - the reason is that we want to apply this on the decisions ("text" column in `subset_disent2.csv`) to check who is the most likely author (or to rephrase it: what is the probability of authorship of each of the 19 judges)
> - please create dir `experiment_02` and therein file IMPLEMENTATION.md; otherwise follow the structure of experiment_01 (to the extent it makes sense)
> - into that file write in detail your proposed course of action; include technical details, key design decisions, assumptions etc.
> - keep experiment_02 completely isolated from experiment_01 (i.e. also create dedicated poetry env in experiment_02) and the rest of the repo with one exception: input dataset `subset_disent2.csv` stays in the repo root
> - any questions you may have - add them to the bottom of the IMPLEMENTATION.md file
> - add this prompt verbatim to the top of the IMPLEMENTATION.md file

> --- please add this prompt verbatim to the top of the document; also, add my answers to your questions at the bottom and add my questions with your answers as well

### 0.2 Follow-up Prompt

> answers to questions:
> 1. should be plain text (but containing linebreaks, unformatted headers, lists etc.)
> 2. `subset.csv` just contains some metadata, not real data; i think the only dataset we can work with for now is `subset_disent2.csv`
> 3. for now, let's not worry about authors with few dissents - let's build good foundation for authors with more dissents and deal with the rest later
> 4. i'm not sure: in the end, we'll be comparing probabilities of authorship for different authors - would that work with/without normalization?
> 5. not at this point, LOO-CV sufficient tfor now
> 6. yes, please add to the scope
>
> a few questions/suggestions of my own:
> - model: only logistic regression now, right? how about XGboost?
> - using UDPIPE2 (https://ufal.mff.cuni.cz/udpipe/2) instead of UDPIPE?
> - adding more UDPIPE2 features, i.e. POS, morphology etc.?
> - feature pruning, feature importance? (for interpretability)
>
> --- please add this prompt verbatim to the top of the document; also, add my answers to your questions at the bottom and add my questions with your answers as well

---

## 1. Goal

Build 19 independent binary classifiers — one per judge — trained on dissenting
opinions (`separate_opinion_extracted`). Each classifier answers: *"How likely is
it that judge J wrote this text?"* Then apply all 19 classifiers to the full
decision texts (`text` column) to produce a probability vector over judges for
every decision.

This is the bridge from fingerprinting (experiment 01) to actual **ghostwriting
detection**: if the attributed judge rapporteur scores low and another judge
scores high, that is a ghostwriting signal.

---

## 2. Approach: One-vs-Rest Binary Classifiers with Calibrated Probabilities

### 2.1 Why per-writer instead of 19-way multi-class?

Experiment 01 used a single 19-way classifier (logistic regression, LOO-CV,
69.7% accuracy). That framing answers: *"Which of these 19 judges wrote this
dissent?"* — a closed-set classification problem.

For ghostwriting detection we need a fundamentally different question: *"Does this
text match judge J's style?"* Reasons:

1. **Open-world application**: The full decision text may have been written by
   someone *not* among the 19 judges (an assistant). A multi-class classifier is
   forced to pick one of the 19; a set of binary classifiers can legitimately say
   "none of these judges match."
2. **Independent probability estimates**: We want P(author = J | text) for each J
   independently. A multi-class softmax forces probabilities to sum to 1 across
   the 19 judges — which is misleading when the true author is outside the
   candidate set.
3. **Per-author tuning**: Each binary classifier can be tuned and calibrated
   independently, accommodating the wide variance in sample sizes (5–32 dissents
   per author).

### 2.2 Training data

- **Source**: `subset_disent2.csv` (repo root), column `separate_opinion_extracted`
- **Labels**: `separate_opinion` (judge name)
- **Filter**: judges with ≥ 5 dissents → 19 authors, 277 texts (same as exp 01)

For judge J's binary classifier:
- **Positive class**: all dissents by J (e.g., 31 for Jan Filip)
- **Negative class**: all dissents by the other 18 judges (e.g., 246)

### 2.3 Class imbalance

The positive:negative ratio ranges from roughly 1:5 (Jan Filip, 31 vs 246) to
1:54 (judges with 5 dissents vs 272). This is severe imbalance. Strategy:

- **`class_weight='balanced'`** in `LogisticRegression` — automatically adjusts
  the loss function to weight samples inversely proportional to class frequency.
  Simple, effective, no data augmentation artifacts.
- **Evaluation metric**: per-author ROC AUC and average precision (PR AUC), not
  accuracy. Accuracy is meaningless at 5:272 ratios.

### 2.4 Classifier choice

**Logistic Regression** (scikit-learn `LogisticRegression`, solver `lbfgs`):

- Directly outputs calibrated-ish probabilities via `predict_proba()`
- Linear model — low overfitting risk with 563 features and 277 samples
- Regularization via C parameter (default C=1.0, tune if needed)
- `class_weight='balanced'` for imbalance handling
- Deterministic, interpretable (feature weights per author)

We do **not** use `SGDClassifier` here (unlike exp 01) because `LogisticRegression`
with `lbfgs` gives proper `predict_proba()` natively. SGD's probability estimates
require additional calibration.

### 2.5 Probability calibration

Even though logistic regression probabilities are "natively" probabilistic, they
may be poorly calibrated for extreme imbalance. We apply **isotonic regression**
calibration using `CalibratedClassifierCV` with method `'isotonic'` and inner
cross-validation (5-fold stratified, not LOO — too expensive for calibration).

Rationale for isotonic over Platt (sigmoid): Platt assumes the uncalibrated
scores follow a sigmoid — may not hold for heavily imbalanced binary problems.
Isotonic is non-parametric and more flexible.

**Fallback**: If sample sizes are too small for isotonic (< ~50 positive samples),
we fall back to `'sigmoid'` (Platt scaling) or skip calibration for that author
and report raw logistic regression probabilities.

### 2.6 Feature extraction

Reuse the **same 4 feature sets** from experiment 01 (563 features total):

| Feature set | Count | Description |
|-------------|-------|-------------|
| `function_words` | 263 | Czech function word relative frequencies (lemma-matched) |
| `surface` | 22 | Sentence/word length stats, punctuation per 1k, TTR |
| `char_ngrams` | 200 | Top-200 character trigram relative frequencies |
| `pos_ngrams` | 78 | POS bigram relative frequencies via UDPipe |
| `morphology` | ~50–80 | **NEW**: Morphological feature distributions + XPOS bigrams (see §14 Q3) |

The feature extraction code will be **copied** from experiment 01 into
experiment 02's `src/` (complete isolation). Any modifications (e.g., vocabulary
handling) will be made in the experiment 02 copy.

**Important**: The character n-gram and POS n-gram vocabularies must be built from
the **training set only** (dissents), not from the application set (full decisions).
The same vocabulary is then used to extract features from both dissents and
decisions.

### 2.7 Feature standardization

`StandardScaler` fitted on the **full training set** (all 277 dissents). The same
scaler is applied to decision texts at inference time. During LOO-CV evaluation,
the scaler is fitted on the LOO training fold only (276 samples).

---

## 3. Evaluation

### 3.1 Leave-one-out cross-validation

For each LOO fold (hold out 1 dissent):
1. Fit scaler on the 276 training dissents
2. For each of the 19 judges, train a binary classifier on the 276 training
   dissents
3. Score the held-out dissent with all 19 classifiers → 19 probabilities
4. Record the predicted probabilities and the true author

This gives 277 probability vectors (each of length 19). From these we compute:

### 3.2 Metrics

**Per-author (binary) metrics:**
- ROC AUC
- Average precision (PR AUC)
- F1, precision, recall at threshold 0.5
- Optimal threshold (Youden's J or F1-maximizing)

**Global (multi-label → multi-class) metrics:**
- **Rank-1 accuracy**: fraction of texts where the true author has the highest
  probability among the 19 classifiers (comparable to exp 01's accuracy)
- **Rank-3 accuracy**: true author in top-3 by probability
- **Mean reciprocal rank (MRR)**: average of 1/rank of the true author

### 3.3 Comparison with experiment 01

We report rank-1 accuracy and rank-3 accuracy to directly compare with exp 01
(69.7% and ~83% respectively). The per-author binary metrics are new and specific
to the per-writer framing.

---

## 4. Application to Full Decision Texts

### 4.1 Pipeline

After evaluation, we train 19 **final classifiers** on all 277 dissents (no
hold-out). Then:

1. Process each full decision text (`text` column in `subset_disent2.csv`) through
   UDPipe
2. Extract the same 563 features using the same vocabulary
3. Standardize using the scaler fitted on all dissents
4. Score with each of the 19 binary classifiers → 19 probabilities per decision
5. Output a CSV: `doc_id, judge_rapporteur, prob_judge_1, ..., prob_judge_19`

### 4.2 Interpretation

For each decision we can then ask:
- Does the judge rapporteur have the highest probability? (= likely self-authored)
- Does another judge score highest? (= possible ghostwriting by that judge or
  their assistant, or stylistic similarity)
- Are all probabilities low? (= likely written by an assistant not in the judge set)

### 4.3 Scope

The `text` column in `subset_disent2.csv` contains the **full decision text** for
the same cases that have dissenting opinions. This is a natural first application
target: we can compare the dissent author's style with the majority opinion's
style within the same case.

---

## 5. Pipeline Steps

| Step | Name | Input | Output | Duration (est.) |
|------|------|-------|--------|-----------------|
| 1 | `load` | `subset_disent2.csv` | `outputs/corpus.pkl` | ~2 s |
| 2 | `udpipe_dissents` | corpus + UDPipe model | `outputs/dissent_documents.pkl` | ~5 min |
| 3 | `features_dissents` | dissent_documents | `outputs/dissent_features.pkl` | ~5 s |
| 4 | `evaluate` | dissent_features | `outputs/loo_results.pkl`, `outputs/loo_probabilities.csv` | ~3 min |
| 5 | `udpipe_decisions` | corpus + UDPipe model | `outputs/decision_documents.pkl` | ~10 min |
| 6 | `features_decisions` | decision_documents | `outputs/decision_features.pkl` | ~5 s |
| 7 | `apply` | dissent_features + decision_features | `outputs/authorship_probabilities.csv` | ~10 s |

Steps 2–3 (dissent processing) and 5–6 (decision processing) are independent and
could in principle run in parallel, but for simplicity we run them sequentially.
The `--from-step` flag allows resuming from any step.

**UDPipe document reuse**: If `experiment_01/outputs/documents.pkl` exists and
contains the same dissent documents, we could in theory load them. However, for
complete isolation, experiment 02 processes its own documents from scratch. The
user can skip expensive UDPipe steps via `--from-step` if outputs already exist.

---

## 6. Project Structure

```
experiment_02/
├── IMPLEMENTATION.md              # This file
├── README.md                      # Replication instructions
├── pyproject.toml                 # Poetry config (Python ≥ 3.11)
├── outputs/                       # Pipeline outputs (auto-generated)
├── scripts/
│   └── run_pipeline.py            # CLI pipeline runner (7 steps)
└── src/
    └── fingerprint/               # Python package
        ├── __init__.py
        ├── data_loader.py         # CSV loading, author filtering
        ├── preprocessing.py       # Text cleaning, UDPipe wrapper
        ├── features/
        │   ├── __init__.py
        │   ├── function_words.py  # Czech function word frequencies (263)
        │   ├── ngrams.py          # Character 3-grams (200) + POS bigrams (78)
        │   ├── surface.py         # Sentence/word length, punctuation, TTR (22)
        │   └── morphology.py      # NEW: morphological feature distributions + XPOS bigrams
        ├── classifiers.py         # Per-author binary LogisticRegression + XGBoost
        ├── evaluation.py          # LOO-CV with binary + ranking metrics
        ├── feature_importance.py  # NEW: LR coefficients, XGBoost importance, reporting
        └── application.py         # Score decision texts with trained classifiers
```

### Modules copied from experiment 01 (verbatim or near-verbatim)

- `data_loader.py` — same CSV loading logic, minor changes for loading the `text`
  column in addition to `separate_opinion_extracted`
- `preprocessing.py` — identical (UDPipe wrapper, Document/Sentence/Token
  dataclasses, clean_text)
- `features/function_words.py` — identical
- `features/surface.py` — identical
- `features/ngrams.py` — identical

### New or substantially modified modules

- **`classifiers.py`**: Completely new. Contains `PerAuthorClassifier` class that
  wraps 19 binary classifiers (LogisticRegression or XGBoost). Methods: `fit(X, y)`,
  `predict_proba(X) → np.ndarray (n_samples × 19)`, `get_author_names()`.
  Supports `classifier_type='logistic'` or `classifier_type='xgboost'`.
- **`evaluation.py`**: Rewritten for binary per-author evaluation. LOO-CV loop
  trains all 19 classifiers per fold. Computes ROC AUC, PR AUC, F1 per author,
  plus rank-1/rank-3/MRR across all authors.
- **`feature_importance.py`**: New. Extracts LR coefficients and XGBoost gain
  per author. Outputs per-author top-20 discriminative features.
- **`application.py`**: New. Loads trained classifiers, processes decision texts,
  outputs probability CSV (raw + softmax-normalized).
- **`features/morphology.py`**: New. Morphological feature distributions (Case,
  Number, Gender, Tense, Voice ratios) and XPOS bigrams.
- **`scripts/run_pipeline.py`**: Extended from exp 01 with 7 steps, decision
  text processing, ablation mode, and feature importance reporting.

---

## 7. Key Design Decisions

### D1: Independent probabilities, not softmax-normalized

Each binary classifier outputs its own P(author = J | text). We do **not**
normalize these to sum to 1 across the 19 judges. Rationale: the true author may
be outside the 19-judge set (an assistant). If we normalize, we lose the ability
to detect "no match."

For ranking purposes (who is the *most likely* author among the 19), the raw
probabilities suffice — we simply rank by probability.

### D2: LogisticRegression over SGDClassifier

Experiment 01 used `SGDClassifier(loss='log_loss')` which is an SGD
approximation of logistic regression. For experiment 02 we switch to
`LogisticRegression(solver='lbfgs')`:
- Exact solution (not stochastic approximation)
- Native `predict_proba()` without additional calibration step
- `class_weight='balanced'` support
- Deterministic results
- Suitable for our data size (277 × 563 — `lbfgs` handles this easily)

### D3: Vocabulary built from dissents only

Character n-gram and POS n-gram vocabularies are built from the 277 dissents. The
same vocabularies are applied when extracting features from decision texts. This
prevents information leakage from the application set into the training pipeline.

During LOO-CV, strictly speaking the vocabulary should be rebuilt per fold. However,
removing 1 out of 277 documents has negligible effect on a top-200 vocabulary, so
we build the vocabulary once from all dissents for efficiency. This is a standard
and defensible approximation.

### D4: Scaler fitted on training data

`StandardScaler` is fitted on the training portion:
- During LOO-CV: fitted on the 276-sample training fold
- For final application: fitted on all 277 dissents

### D5: No text chunking (yet)

Experiment 01 identified text chunking as a planned improvement. We do not
implement it in experiment 02's initial version to keep the comparison clean.
It can be added later as an option.

### D6: The `text` column may need preprocessing

The `text` column contains the **full decision text** which may include headers,
procedural parts, and ancillary materials (see README.md §3 on `subset`). For
this initial experiment, we apply the same `clean_text()` function and UDPipe
processing. If the full decision texts contain HTML or structured markup, we may
need additional cleaning. This is flagged as a question below.

---

## 8. Dependencies

Same as experiment 01 plus calibration (already in scikit-learn):

| Package | Version | Purpose |
|---------|---------|---------|
| `python` | ^3.11 | Runtime |
| `pandas` | ^3.0.0 | Data loading |
| `numpy` | ^2.4.2 | Feature matrix |
| `scikit-learn` | ^1.8.0 | LogisticRegression, CalibratedClassifierCV, metrics |
| `matplotlib` | ^3.10.8 | Plots (calibration curves, ROC curves) |
| `seaborn` | ^0.13.2 | Heatmaps (probability matrix visualization) |
| `ufal-udpipe` | ^1.4.0.1 | Tokenization, POS tagging, lemmatization |
| `xgboost` | ^2.1.0 | Gradient boosted trees (secondary classifier) |

---

## 9. Expected Outputs

| File | Description |
|------|-------------|
| `outputs/corpus.pkl` | Filtered DataFrame |
| `outputs/dissent_documents.pkl` | UDPipe-processed dissent documents |
| `outputs/dissent_features.pkl` | Feature matrix (277 × ~620–640) + labels + feature names |
| `outputs/loo_results.pkl` | LOO-CV results: per-author binary metrics + ranking metrics |
| `outputs/loo_probabilities.csv` | LOO-CV probability matrix: 277 rows × 19 probability columns |
| `outputs/results_summary.txt` | Human-readable evaluation report |
| `outputs/decision_documents.pkl` | UDPipe-processed decision texts |
| `outputs/decision_features.pkl` | Feature matrix for decisions |
| `outputs/authorship_probabilities.csv` | Final output: per-decision probability over 19 judges |
| `outputs/trained_classifiers.pkl` | Serialized classifiers + scaler + vocabulary for reuse |
| `outputs/feature_importance.csv` | Per-author top-20 discriminative features (LR + XGBoost) |
| `outputs/ablation_results.csv` | Feature ablation study results (per feature set, per classifier) |

---

## 10. Implementation Order

1. **Scaffold** — Create project structure, pyproject.toml, copy shared modules
2. **Classifiers** — Implement `PerAuthorClassifier` with `fit()` and
   `predict_proba()` for both LogisticRegression and XGBoost
3. **Morphological features** — Implement `features/morphology.py` (XPOS
   bigrams, morphological category distributions)
4. **Evaluation** — LOO-CV with binary and ranking metrics
   - 4b. **Feature ablation** — Run evaluation with each feature set alone and
     in combinations; compare LR vs XGBoost across feature sets
5. **Feature importance** — LR coefficients + XGBoost gain per author;
   per-author top-20 features report
6. **Pipeline script** — Steps 1–4 (load → udpipe → features → evaluate),
   verify on dissents
7. **Application** — Steps 5–7 (process decisions → extract features → score),
   produce `authorship_probabilities.csv` (raw + normalized)
8. **Analysis & visualization** — Calibration curves, probability heatmaps,
   comparison with exp 01, feature importance plots
9. **Documentation** — README.md, methodology_and_results.md

---

## 11. Assumptions

- **A1**: Dissenting opinions are single-authored by the attributed judge (same
  assumption as experiment 01).
- **A2**: The stylometric features that distinguish judges in dissents are
  transferable to full decision texts (same writing style regardless of document
  type).
- **A3**: The `text` column in `subset_disent2.csv` is suitable for analysis
  without major additional preprocessing beyond `clean_text()`.
- **A4**: 19 independent binary classifiers with `class_weight='balanced'` are
  sufficient to handle the imbalance; no need for SMOTE or other synthetic
  oversampling.

---

## 12. Open Questions

1. **What is the format of the `text` column?** Is it plain text, HTML, or mixed?
   Experiment 01 used `separate_opinion_extracted` which appears to be HTML-extracted.
   If `text` contains raw HTML, we need an HTML-stripping step before `clean_text()`.

2. **Should we also apply the classifiers to `subset.csv` (the majority opinions
   dataset)?** The `subset_disent2.csv` `text` column contains full decisions
   for cases *with* dissents. The broader `subset.csv` has all decisions. Applying
   to `subset.csv` would be the full ghostwriting analysis, but requires
   `git lfs pull` and potentially different preprocessing.

3. **Calibration feasibility**: For authors with only 5 dissents, calibration
   via `CalibratedClassifierCV` with 5-fold CV means some folds will have only
   1 positive sample. Should we skip calibration for small-sample authors, or
   use a different calibration strategy (e.g., Platt scaling which needs fewer
   samples)?

4. **Probability normalization**: We decided *not* to normalize probabilities
   across the 19 classifiers. Should we additionally provide a normalized
   version (softmax over the 19 raw probabilities) for convenience, even if
   the unnormalized version is the primary output?

5. **Temporal split consideration**: Some judges served different periods.
   Should we account for temporal drift in writing style, or is LOO-CV
   sufficient for this initial experiment?

6. **Feature set**: Should we stick to the exact same 563 features as exp 01,
   or is this also a good opportunity to run the ablation study from the
   sprint backlog? (Could help identify which features matter for the binary
   framing.)

---

## 13. Answers to Open Questions (from User)

1. **`text` column format**: Plain text (with linebreaks, unformatted headers,
   lists etc.). No HTML stripping needed — `clean_text()` as-is should suffice.
   **→ Decision D6 updated: no additional HTML preprocessing required.**

2. **`subset.csv`**: Contains only metadata, not usable text data. The only
   working dataset is `subset_disent2.csv`. **→ Application scope confirmed:
   we apply classifiers to the `text` column of `subset_disent2.csv` only.**

3. **Small-sample authors**: Don't worry about them for now. Build a solid
   foundation for authors with more dissents; deal with the rest later.
   **→ We still include all 19 authors in training (their dissents are useful
   as negative examples for other classifiers), but we focus evaluation and
   reporting on authors with ≥ ~10 dissents. Calibration is skipped for
   small-sample authors.**

4. **Probability normalization**: The user will be comparing probabilities across
   authors for the same document. **→ Decision: output both raw and normalized
   probabilities.** Raw probabilities (independent per-classifier) are the
   primary output — they preserve the "no match" signal. A softmax-normalized
   version is provided as a convenience column set for ranking comparison.
   The normalization is purely a post-processing step and does not affect
   training or evaluation.

5. **Temporal split**: LOO-CV is sufficient for now. No temporal split needed.

6. **Feature ablation**: Added to scope (see §10, step 4b).

---

## 14. User Questions and Answers

### Q1: Only logistic regression? How about XGBoost?

XGBoost is a strong candidate and was already on the sprint backlog. Key
considerations for our setting:

| | LogisticRegression | XGBoost |
|---|---|---|
| **Overfitting risk** | Low (linear, 1 hyperparameter C) | Higher (trees, many hyperparams) |
| **Feature interactions** | None (linear) | Yes (tree splits) |
| **Imbalance handling** | `class_weight='balanced'` | `scale_pos_weight` |
| **Probability quality** | Good natively | Needs calibration |
| **Interpretability** | Coefficients per feature | `feature_importances_` (gain) |
| **Speed** | Fast | Somewhat slower, 19× models |

**Decision**: Implement **both**. Logistic regression is the primary classifier
(simpler, more interpretable, better-calibrated probabilities). XGBoost is a
second classifier run in parallel for comparison. This also serves as a sanity
check — if XGBoost dramatically outperforms LR, feature interactions matter and
we should investigate.

XGBoost hyperparameters need care with 277 samples: shallow trees (`max_depth=3`),
strong regularization (`reg_alpha`, `reg_lambda`), early stopping. We use
`class_weight`-equivalent via `scale_pos_weight = n_neg / n_pos` per author.

**Added to dependencies**: `xgboost` package.

### Q2: UDPipe 2 instead of UDPipe 1?

UDPipe 2 (Straka, 2018) uses contextualized embeddings and achieves better
tagging/lemmatization quality than UDPipe 1. However, there are important
practical differences:

| | UDPipe 1 | UDPipe 2 |
|---|---|---|
| **Tokenization** | Yes (built-in) | **No** (requires pre-tokenized CoNLL-U input) |
| **Interface** | Local C library (`ufal.udpipe` Python bindings) | Python server or REST API |
| **Speed** | Fast, local | Slower, needs GPU for training |
| **Model availability** | Downloadable `.udpipe` files | REST service at LINDAT, or self-hosted |
| **Quality** | Good (UD 2.5 models) | Better (contextualized embeddings) |

The critical issue: UDPipe 2 **does not tokenize**. We would need a separate
tokenizer (e.g., UDPipe 1 just for tokenization, or MorphoDiTa, or a regex
tokenizer) and then pipe the CoNLL-U to UDPipe 2 for tagging/lemmatization.

**Decision for experiment 02**: Stick with **UDPipe 1** for the initial
implementation. The quality difference in POS tagging is real but unlikely to
be the bottleneck for authorship attribution (stylometric features are robust
to minor tagging errors). Switching to UDPipe 2 can be done later as an
enhancement — the preprocessing module is isolated, so swapping the backend
is straightforward.

**Future option**: Use UDPipe 2 via the LINDAT REST API
(`https://lindat.mff.cuni.cz/services/udpipe/api/`) for better quality without
local setup. Would require network access and rate limiting consideration for
277+ documents.

### Q3: More UDPipe features (POS, morphology)?

Currently we use from UDPipe:
- **Lemmas** → function word matching
- **UPOS tags** → POS bigrams (78 features)
- **Token forms** → surface features, char n-grams

Unused fields already available in our `Token` dataclass:
- **XPOS** (Czech-specific POS tags, much more fine-grained than UPOS)
- **Feats** (morphological features: Case, Number, Gender, Tense, Mood, etc.)

Potential new feature sets:

| Feature set | Description | Est. features |
|-------------|-------------|---------------|
| `xpos_ngrams` | XPOS tag bigrams (finer-grained POS patterns) | ~150–300 |
| `morph_features` | Morphological feature distributions (e.g., ratio of genitive case, passive voice markers) | ~30–50 |
| `morph_ngrams` | Sequences of morphological categories | ~100–200 |

Morphological features are particularly promising for Czech (highly inflected
language) — a judge who prefers passive constructions or certain case patterns
would show distinctive morphological distributions.

**Decision**: Add **morphological features** to the scope as a new feature set.
Implement as a separate module `features/morphology.py` extracting:
- Distribution of morphological categories (Case, Number, Gender, Tense, Voice)
- XPOS bigrams
- Passive/active voice ratio (derivable from morphological features)

This is included in the feature ablation study — we can measure whether
morphological features add signal beyond the existing 563 features.

### Q4: Feature pruning and feature importance?

Yes — this is critical for interpretability and added to scope.

**Feature importance methods:**

1. **Logistic regression coefficients**: For each author's binary classifier,
   the coefficient vector directly shows which features increase/decrease the
   probability of that author. E.g., "Jan Filip's classifier has high positive
   weight on `sent_len_mean` → he writes longer sentences."

2. **Permutation importance**: Model-agnostic. Shuffle one feature at a time,
   measure drop in ROC AUC. Works for both LR and XGBoost.

3. **XGBoost feature importance**: Built-in `feature_importances_` (gain-based).
   Shows which features are most used in tree splits.

**Feature pruning methods:**

1. **L1 regularization** (Lasso): Use `LogisticRegression(penalty='l1',
   solver='saga')` to automatically zero out irrelevant features. Compare
   with the L2 baseline.

2. **Recursive feature elimination (RFE)**: Iteratively remove least important
   features. Expensive but thorough.

3. **Variance threshold**: Remove near-zero-variance features as a preprocessing
   step.

**Decision**: Implement feature importance reporting (LR coefficients + XGBoost
importance) as part of the evaluation output. Feature pruning (L1 vs L2
comparison) is included in the ablation study. RFE is deferred to a future
experiment.

**Output**: Per-author top-20 most discriminative features in
`outputs/feature_importance.csv` and `outputs/results_summary.txt`.
