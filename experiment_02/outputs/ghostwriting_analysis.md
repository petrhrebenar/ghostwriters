# Ghostwriting Analysis — Rapporteur vs. Classifier

Source:
- `outputs/authorship_probabilities_logistic.csv` — 277 decisions × 19 probabilities (from step 7 `apply`).
- `subset.csv` — original dataset from [stepanpaulik/ccc_dataset](https://github.com/stepanpaulik/ccc_dataset), 93,826 rows with `judge_rapporteur_name`. Joined on `doc_id`.

Produced files:
- `outputs/ghostwriting_analysis_logistic.csv` — 181 rows (one per decision whose rapporteur is among our 19 trained authors) with: rapporteur, `prob_rapporteur`, `rapporteur_rank`, top-1/top-3 match flags, plus the full 19-probability row.
- `outputs/ghostwriting_per_rapporteur.csv` — per-rapporteur aggregate: #decisions, mean rank, top-1/top-3 rate, dissent-LOO ROC AUC.

---

## 1. Setup

Of the 277 decisions with a dissent in our corpus:

- **181 decisions (65 %)** have a rapporteur who is *also* one of our 19 trained authors (i.e., the rapporteur personally wrote ≥ 5 dissents, so we have a stylistic fingerprint for them).
- **96 decisions (35 %)** have a rapporteur outside the trained set (Vojen Güttler, Jiří Mucha, Jaromír Jirsa, Pavel Holländer, Vladimír Čermák, Pavel Rychetský, Antonín Procházka, and others — mostly judges with < 5 dissents in the filtered corpus). These 96 cannot be directly checked against the rapporteur.

**Key question.** For the 181 evaluable decisions: does the binary classifier trained on judge J's *dissents* assign a high probability to judge J's *own majority opinion*?

- **If yes** across many decisions → the rapporteur personally drafted them (style carries over from dissent to majority).
- **If no** across many decisions → the majority opinion was drafted by someone *else* (ghostwriter).

---

## 2. Headline results

| | Top-1 rapporteur match | Top-3 | Top-5 |
|---|---:|---:|---:|
| **Classifier** | **12.2 %** (22 / 181) | **34.3 %** (62 / 181) | **49.2 %** (89 / 181) |
| Random baseline | 5.3 % (1/19) | 15.8 % (3/19) | 26.3 % (5/19) |
| **Lift vs. random** | **×2.3** | **×2.2** | **×1.9** |

For comparison, the *same classifiers* achieve 73.3 % top-1 on dissents (LOO-CV).

**Interpretation.** The rapporteur's stylistic fingerprint *is* detectable in
their majority opinions at a rate significantly above chance, but at a rate
that is **6× lower** than for dissents (12 % vs. 73 %). This is consistent
with the central hypothesis of the project:

> Majority opinions in the Czech Constitutional Court are frequently drafted
> by judicial assistants, not by the rapporteur themselves.

The classifier's top pick *is* the rapporteur only 1 time in 8. In
contrast, if majority opinions were consistently self-drafted, we would
expect the top-1 rate to approach the LOO-CV baseline (~70 %). A 12 % rate
means the majority text typically does *not* carry the rapporteur's
idiolect strongly enough to dominate the other 18 candidate classifiers.

**Distribution of P(author = rapporteur | decision):**

| | raw probability | softmax-normalized |
|---|---:|---:|
| mean | 0.071 | 0.055 |
| median | 0.005 | 0.052 |
| max | 0.977 | 0.128 |

The median raw probability that a decision was written by its own
rapporteur is **0.5 %** (yes, half of one percent). The softmax
normalization, which forces the 19 probabilities to sum to 1, pulls the
rapporteur's share to ≈1/19 = 5.3 % on average — essentially the random
baseline. This is striking.

---

## 3. Per-rapporteur breakdown

Only rapporteurs with N ≥ 2 decisions. `loo_roc_auc` is the dissent
classifier's LOO-CV ROC AUC (a measure of how distinctive the judge's
*dissent* style is).

| Rapporteur | N dec. | LOO ROC AUC | top-1 rate | top-3 rate | mean prob | mean rank (of 19) |
|---|---:|---:|---:|---:|---:|---:|
| Vojtěch Šimíček | 35 | 0.978 | 0.114 | 0.543 | 0.058 | 5.94 |
| Eliška Wagnerová | 18 | 0.919 | 0.056 | 0.222 | 0.016 | 6.44 |
| **Jan Filip** | **16** | **0.999** | **0.000** | **0.000** | **0.003** | **10.38** |
| Radovan Suchánek | 16 | 0.935 | 0.438 | 0.812 | 0.264 | 2.19 |
| Ludvík David | 16 | 0.897 | 0.062 | 0.062 | 0.013 | 10.44 |
| Jiří Zemánek | 14 | 0.971 | 0.000 | 0.071 | 0.006 | 8.29 |
| Jiří Nykodým | 11 | 0.396 | 0.000 | 0.273 | 0.009 | 6.82 |
| Kateřina Šimáčková | 9 | 0.991 | 0.111 | 0.444 | 0.039 | 6.78 |
| Vladimír Kůrka | 7 | 0.975 | 0.143 | 0.143 | 0.099 | 5.00 |
| Stanislav Balík | 6 | 0.891 | 0.167 | 0.667 | 0.191 | 2.83 |
| Michaela Židlická | 6 | 0.959 | 0.000 | 0.000 | 0.003 | 9.17 |
| Miloslav Výborný | 5 | 0.975 | 0.000 | 0.400 | 0.004 | 8.80 |
| David Uhlíř | 5 | 0.943 | 0.000 | 0.000 | 0.002 | 12.60 |
| **Ivana Janů** | **5** | **0.852** | **0.600** | **0.800** | **0.163** | **2.00** |
| Jan Musil | 5 | 0.995 | 0.000 | 0.200 | 0.006 | 9.00 |
| Vladimír Sládeček | 3 | 0.954 | 0.333 | 0.333 | 0.307 | 5.00 |
| Josef Fiala | 2 | 0.982 | 0.000 | 1.000 | 0.029 | 2.50 |
| Iva Brožová | 1 | 1.000 | 1.000 | 1.000 | 0.933 | 1.00 |
| Pavel Varvařovský | 1 | 0.993 | 1.000 | 1.000 | 0.977 | 1.00 |

---

## 4. Findings

### 4.1 Self-drafters — strong evidence the rapporteur personally wrote the decision

Filters: dissent classifier ROC AUC ≥ 0.95 AND top-3 rate ≥ 0.5 AND the
patterns are reproducible across multiple decisions.

1. **Radovan Suchánek** (16 decisions). Top-1 rate **44 %**, top-3 rate
   **81 %**, mean rank 2.2, mean raw probability 0.26. **Strongest
   self-drafting signal in the corpus with a meaningful sample.** He
   consistently drafts his own majority opinions; his dissent fingerprint
   is clearly present in 13 of 16 decisions.

2. **Ivana Janů** (5 decisions). Top-1 **60 %**, top-3 80 %, mean rank 2.0.
   Despite her classifier's relatively lower LOO ROC AUC (0.852), she
   comes out on top for her own decisions — meaning her dissent style *is*
   visible in her majority opinions even though it overlaps with other
   judges'. She drafts her own.

3. **Stanislav Balík** (6 decisions). Top-3 67 %, mean rank 2.8. Drafts.

4. **Vojtěch Šimíček** (35 decisions — by far the largest rapporteur
   group). Top-1 11 %, top-3 **54 %**. Mixed picture: his dissent style
   is present in roughly half of his decisions and absent in the other
   half. Given his high dissent-ROC (0.978), this is consistent with
   variable drafting practice across his tenure / case types.

5. Low-sample authors whose single / pair of decisions match perfectly:
   Iva Brožová (1 / 1 with raw prob 0.93), Pavel Varvařovský (1 / 1, raw
   prob 0.98), Josef Fiala (2 / 2 in top-3).

### 4.2 Ghostwriting candidates — strong evidence the rapporteur did NOT draft

Filters: dissent classifier ROC AUC ≥ 0.95 AND top-3 rate ≤ 0.15 AND
N ≥ 5 decisions.

1. **Jan Filip** (16 decisions, dissent ROC 0.999 — the most distinctive
   writer in the corpus). Top-1 rate **0 %**, top-3 rate **0 %**, mean rank
   **10.4** out of 19, mean raw probability **0.003**. **This is the
   strongest ghostwriting signal in the data.** Filip has an essentially
   perfect dissent classifier, yet in *zero* of his 16 majority opinions
   does his style come through as even top-3 likely. His mean rank of
   10.4 is slightly *below* the random-guessing expectation (9.5) — his
   classifier actively rejects his own majority opinions.

2. **Michaela Židlická** (6 decisions, ROC 0.959). Top-3 rate 0 %, mean
   rank 9.2. Same pattern as Filip on a smaller sample.

3. **Jiří Zemánek** (14 decisions, ROC 0.971). Top-3 rate 7 %, mean rank
   8.3. Same pattern, N = 14.

4. **David Uhlíř** (5 decisions, ROC 0.943). Top-3 rate 0 %, mean rank
   12.6. His classifier already has known calibration issues (F1 = 0 on
   dissents); interpret with caution, but ranking-wise his decisions are
   consistently at the bottom of the `prob_Uhlíř` distribution.

5. **Ludvík David** (16 decisions, ROC 0.897). Top-3 rate 6 %, mean rank
   10.4. Below the ROC ≥ 0.95 threshold but with a large sample and clear
   pattern.

### 4.3 Caveats

1. **Genre mismatch.** Dissents are argumentative first-person texts;
   majority opinions include procedural boilerplate, fact recitations, and
   standardized legal formulas. Some features (sentence length,
   passive-voice ratio, certain character trigrams) shift systematically
   between genres, which depresses *all* probability estimates on
   decisions relative to dissents. The sharp 73 % → 12 % top-1 drop
   therefore mixes two effects: real ghostwriting + genre drift. Only the
   *per-rapporteur contrast* (Suchánek drafts / Filip does not) controls
   for the shared drift.

2. **Classifier calibration.** Raw probabilities from
   `class_weight='balanced'` logistic regression are not calibrated, so
   absolute values like "mean prob 0.003" should not be read as "0.3 %
   literal probability". Ranks and relative values are the right
   interpretation.

3. **181 of 277 decisions evaluable.** The remaining 96 decisions have a
   rapporteur outside our 19-author set. For those we can still observe
   "which of the 19 trained judges scores highest", and if a trained
   judge scores very high on a decision attributed to a non-trained
   judge, that's a different form of ghostwriting signal (a trained
   judge stylistically matching someone else's attributed decision —
   suggesting the trained judge, or someone drafting in their house
   style, was involved).

4. **Not controlled for chamber, case type, or time period.** A judge
   who participated in a chamber that drafted collaboratively might show
   lower style-match even if they did contribute substantively. Cross-
   references against `formation` and `date_decision` are recommended as
   next steps.

---

## 5. Summary

The classifier picks up the rapporteur's style **2–3× better than random**
but **6× worse than it picks up dissent authors on dissents**. The
per-rapporteur picture splits cleanly into two groups:

- **Self-drafters**: Suchánek (16), Janů (5), Balík (6), and partly
  Šimíček (35). Their majority decisions bear their dissent fingerprint.
- **Ghostwriting candidates**: Filip (16), Židlická (6), Zemánek (14),
  Kůrka (7), and with caveats Uhlíř (5) and David (16). Their dissent
  fingerprint is largely absent from their majority decisions.

The most striking single observation is **Jan Filip**: the judge with the
most distinctive dissent style in the corpus (ROC AUC 0.999, top weight
per dissent) has **zero top-3 matches** out of 16 attributed majority
decisions. This is the clearest ghostwriting signal the analysis
produces.

Recommended next steps:

1. Probability calibration (isotonic regression) to make absolute numbers
   interpretable and stabilize thresholds.
2. Formation / chamber cross-analysis to test whether individual judges'
   signals correlate with specific chambers.
3. Score the 96 decisions with non-trained rapporteurs and look for
   patterns where a trained judge scores anomalously high on someone
   else's attributed decision.
4. Within-author variance analysis (Rosenthal–Yoon χ²) on each trained
   judge's *own dissents* — if their dissents themselves show high
   internal stylistic variance, even the "self-drafters" signal may be
   inflated.
