# Feature Importance Analysis — Logistic Regression

Source: `outputs/feature_importance_logistic.csv` (380 rows = 19 authors × top-20 features).
Model: the same 19 binary classifiers trained on all 277 dissents in the apply
step; importance = L2-regularized logistic-regression coefficients on
StandardScaled features. Positive weight → feature increases P(author = J).

The feature-name prefixes encode the feature set:

| Prefix | Set |
|---|---|
| `fw_` | Function word (relative frequency over tokens) |
| `char3_` | Character 3-gram (relative frequency) |
| `pos2_` | UPOS bigram |
| `xpos2_` | Czech-specific XPOS bigram |
| `sent_`, `word_`, `punct_`, `ttr_` | Surface features |
| `morph_` | Morphological category distributions |

---

## Per-author top-5 features

### Tier 1 — highly distinctive writers

**Jan Filip** (ROC AUC 0.999) — discourse connectives and demonstratives:
`dokud`, `takže`, `u`, `ten`, plus an XPOS bigram for preposition+6th-case
pronoun. A highly idiosyncratic mix of subordinating conjunctions and
demonstrative pronouns.

**Jan Musil** (ROC AUC 0.995) — punctuation/structure-driven:
`word_len_std` (variance of word length), adjective+noun locative XPOS bigram,
semicolons per 1k tokens, and a *negative* weight on `ale` ("but" — he uses it
less often). Structural rather than lexical.

**Iva Brožová** (ROC AUC 1.000) — archaic / formal register:
`protože`, `dle` (archaic "according to"), plus XPOS bigrams of noun+clause
marker. `dle` is a known archaism — its use is a strong stylistic marker.

**Pavel Varvařovský** — `tedy`, `nežli`, `pak`, `vlastně` — discourse markers
building an argumentative chain.

**Kateřina Šimáčková** — `sent_count` (she writes *many* sentences per text),
`napříč` ("across"), and several feminine-ending character trigrams
(`ván`, `kov`, `ová` — note `ová` is the feminine adjective/surname suffix;
could indicate vocabulary or topic).

**Josef Fiala** — `pos2_ADP_PROPN` (preposition + proper noun) plus
adjective+noun XPOS bigrams in the 6th (locative) case. Legal-citation pattern.

**Vojtěch Šimíček** — the highest single weight in the whole table:
`sent_count` at +0.51, plus `totiž` ("namely"), `nikdo` ("no one"), `nyní`,
`právě`. Structural + discourse-marker driven.

**Vladimír Kůrka** — punctuation-heavy: semicolons per 1k, double-quote
character trigrams (`' "'`), punctuation rate `"`, plus POS bigrams
involving PUNCT. Extremely distinctive *typographic* style.

**Miloslav Výborný** — `buď`, semicolons per 1k, `ADJ_VERB` POS bigram,
`především`, `též` ("also", archaic).

**Jiří Zemánek** — `jakmile` ("as soon as"), `přestože` ("although"),
`sent_len_mean`, `sent_len_max` — long sentences with subordination.
Plus a negative weight on `char3_že` — he actually uses the "že" trigram
*less* than average (despite `přestože` containing "že", suggesting his
`že`-heavy words are different).

### Tier 2 — moderate signal

**Radovan Suchánek** (ROC 0.935, N=30) — huge *negative* weight on
semicolons (−0.44) — he avoids them. Plus XPOS bigrams of
masculine-animate-genitive + punctuation, and `přitom` / `ani` function
words.

**Ivana Janů** (ROC 0.852, N=32 — the anomaly) — top features are a mix
of XPOS bigrams (pronoun+locative-noun), archaic `potom`, character
trigram `char3_ ob`, and `bezpochyby`, `pouze`. No dominant single
signal — consistent with her moderate ROC AUC despite 32 training samples.
She doesn't have a single strong stylistic peak; the classifier relies on
many weak features that overlap with other authors.

**Stanislav Balík** — `fw_ke` (preposition variant), `pos2_AUX_PRON`
bigram, `ještě`, `asi`, char trigram `led` (possibly part of `následně`
or similar).

**Eliška Wagnerová** — `naopak`, `avšak`, `skrz`, `vedle`, `při` — all
function words, strongly argumentative.

**Ludvík David** — `přestože`, `též`, semicolons, `zjevně`, `však`.
Archaic connectives.

**David Uhlíř** (ROC 0.943 but F1 = 0.0 — ranking-only useful) — `vůči`,
`zda`, `také` plus two *negative* features (`char3_ně`, `fw_jenž`).
Very small sample (N=6); the classifier found signal but the 0.5
threshold is unreachable.

**Michaela Židlická** — `doposud`, `neboť`, `pos2_PRON_NOUN`, `dle`,
`nicméně`. Archaic/literary connectives.

**Vladimír Sládeček** — `také` (big: +0.41), `char3_ky`, preposition+noun
dative XPOS bigram, `char3_den`, `ostatně`.

**Jiří Nykodým** (ROC 0.396 — failed) — `toho`, `ovšem`, `čili`, `char3_pří`,
NUM_ADP POS bigram. Interpretable features but weights don't generalize:
with only 5 samples, the "top features" likely overfit to spurious
correlations.

---

## Cross-author patterns

### Shared discriminative features

The single feature most widely used as a top-20 discriminator is **`punct_;_per1k`**
(semicolons per 1,000 tokens), appearing in 7 of 19 authors' top-20 lists
— sometimes positive, sometimes negative:

| Author | Rank | Weight | Direction |
|---|---:|---:|---|
| Radovan Suchánek | 1 | −0.443 | avoids semicolons |
| Miloslav Výborný | 2 | +0.288 | uses them a lot |
| Jan Musil | 3 | +0.247 | uses them |
| Ludvík David | 3 | +0.250 | uses them |
| Vladimír Kůrka | 1 | +0.234 | uses them |
| Stanislav Balík | 15 | −0.173 | avoids |
| Ivana Janů | 20 | +0.247 | mild positive |

Semicolon usage is a strong *idiolectal* marker in Czech legal writing — the
sample is effectively split between "semicolon judges" and "no-semicolon
judges".

### Feature-set contribution (inferred from top-20 lists)

Across all 380 rows (19 × 20):

- Function words dominate: most top-20 entries carry the `fw_` prefix.
- XPOS bigrams appear prominently for the highest-ROC-AUC authors (Jan
  Musil, Iva Brožová, Josef Fiala) — confirming that the new
  **morphology** feature set added discriminative power beyond Exp 01's
  UPOS bigrams.
- Surface features (`sent_count`, `sent_len_mean`, `word_len_std`,
  `punct_;_per1k`, `ttr_form`) are a meaningful minority — crucial for
  authors like Vojtěch Šimíček, Jan Musil, and Kateřina Šimáčková whose
  signal is structural rather than lexical.
- Character trigrams appear but rarely at the very top; they seem to
  supplement rather than dominate.

### Interpretation re: the "Ivana Janů anomaly"

Janů's 32-dissent corpus produces the largest training set, yet her ROC AUC
(0.852) is lower than that of authors with 6–12 dissents. Looking at her
top-5 features:

1. XPOS bigram (pronoun+locative-noun)
2. `fw_potom` (archaic "then")
3. `char3_ ob` (three characters starting a word with "ob")
4. `fw_bezpochyby` ("without a doubt")
5. `fw_pouze` ("only")

None of these are strongly unique — `potom`, `pouze`, `bezpochyby` are all
in many other judges' vocabularies. The top-weighted features max out at
+0.33, compared to +0.51 (Šimíček) or +0.40 (Sládeček). **There is no
dominant stylistic signature**. This supports the hypothesis from §4.3 of
`methodology_and_results.md`: her dissents may genuinely exhibit high
within-author stylistic variance — a candidate for a Rosenthal-Yoon-style
within-author χ² analysis to check for heterogeneity in her own dissent
corpus.
