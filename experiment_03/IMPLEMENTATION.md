# Experiment 03 — Implementation (as-built) & deviations from the plan

This is the **as-built** record. It documents what was actually implemented and
where it diverged from [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). For the
scientific write-up and results see
[docs/methodology_and_results.md](docs/methodology_and_results.md).

## TL;DR of deviations

| # | Area | Plan | As built | Why |
|---|------|------|----------|-----|
| 1 | Scraper transport | **Option B** (pure-HTTP ASP.NET postback emulation) | **Option A** (Selenium + Firefox), the plan's documented fallback | Pure-HTTP POSTs never executed — server-side ASP.NET validators silently re-render the form. The reference `ccc_dataset` also drives a real browser. |
| 2 | Search driver | by date-range *or* spisová značka | by **ECLI** (one ECLI → one record card) | Targeted and exact; one request per decision, no pagination. |
| 3 | Corpus size | "~272" | **309** (236 senate + 73 plenary); **308 scraped** | Plenary-regex fix recovered more than estimated; `Pl-42-17_1` returned no ECLI hit. |
| 4 | Annotation scope | (re)annotate the corpus | **kept the 236 human-reviewed**, annotated only the **73 new plenary** | Annotation is metadata-independent (LLM only sees body text), so enriching metadata did not require re-annotating. |
| 5 | Metadata join | scrape produces records | **enriched the 236 in place** (kept exact body text), fetched bodies only for the 73 new | Preserves the line numbering the existing tags are tied to. |
| 6 | Dataset shape | one analysis dataframe (~experiment-02 columns) | **two/three tables**: `decisions`, `dissents`, `sections` | Dissents are now split per judge (training); RATIO is a separate scoring target; sections added for per-decision cards. |
| 7 | Scoring target | "RATIO only, or RATIO + full text both" | **RATIO + every section** (no full-text baseline) | Section-by-section scoring (added mid-project) subsumes and improves on full-text. |
| 8 | Layout | `src/` package sketch | **flat dirs** under `experiment_03/` | Resolved decision. |
| 9 | Classifier | logistic + xgboost | **logistic only** (xgboost deferred) | LOO×19 XGBoost CV was the wall-clock bottleneck; deferred by agreement. |
| 10 | Env | clean `experiment_03` poetry install | **reused experiment-02's env** for analysis | Pragmatic; clean install is a tracked loose end. |

## Stage-by-stage

### Stage 0 — corpus (self-containment)
- **As built:** `scrape/build_seed_list.py` bootstraps `corpus/seed.tsv`
  (file_id ↔ ECLI) **once** from the legacy CSV; the committed TSV is the corpus
  source of truth thereafter. Plenary ECLI regex widened (`Pl` accepted).
- **Deviation:** seed is **ECLI-keyed** (plan stored file ids only) because the
  record-card search keys on ECLI. The file id is recovered authoritatively from
  the card's "URL adresa" field.

### Stage 1 — scrape (the big one)
- **As built:** `scrape/scraper.py` (Selenium Firefox: ECLI search → ResultDetail
  id; HTTP GetText for body), `scrape/parse_card.py` (**transport-agnostic**
  `recordCardTable` → metadata: rapporteur, clean nominative `separate_opinion`,
  ISO dates, formation, type_verdict, file_id), `scrape/pipeline.py`
  (orchestration, resumable, raw-HTML cache, enrich-in-place vs fetch-new).
- **Deviations:** Option A not B (#1); ECLI search (#2); enrich-in-place (#5).
  The plan's transport-agnostic parser design **held** — only the fetch layer
  changed from the planned one.
- **Open item realized:** `judge_rapporteur_id` — we key on normalized names, no
  external judge table (as the plan permitted).

### Stage 2 — LLM annotation
- **As built:** existing `llm_annotate/` reused unchanged in logic; only annotated
  the 73 new plenary (resumable: `run_llm` skips done, `render` now **skips
  existing tagged files** so human-reviewed `.txt` are never clobbered — a guard
  added during this work).
- **Deviations:** scope (#4); wired the API to the UFAL endpoint/key/model via
  `.env` (`OPENROUTER_UFAL_*`), loaded silently. Module invocation path updated
  from `annotation.*` to `llm_annotate.*` (flat layout).

### Stage 3 — dataset
- **As built:** `extract_tags.py` (spans), `authorship.py` (DIS/CON → judge:
  span-heading name reconciled against scraped `separate_opinion`, genitive-
  tolerant surname match), `build_dataset.py` (decisions + dissents),
  `build_sections.py` (per-(decision,label) sections table).
- **Deviations:** the plan's "open question" on multi-author dissents was
  resolved by the annotation schema itself (one tag-pair per judge) + heading
  attribution — **0 unattributed**. Three tables instead of one (#6). Sections
  table is new (supports the per-decision cards).

### Stage 4 — analysis
- **As built:** ported `fingerprint/` from experiment-02; **rewrote
  `data_loader.py`** to read the new tables (parquet-or-CSV); factored shared
  feature/UDPipe code into `fingerprint/featureset.py`; `scripts/06_analyze.py`
  (train per-judge on DIS/CON, LOO-validate, score RATIO, attribute vs
  rapporteur); `score_sections.py` (score every section, reusing the saved
  model); `plots.py` (validation confusion, corpus matrix, section-type figure,
  308 per-decision cards).
- **Deviations:** RATIO + all sections, no full-text baseline (#7); logistic only
  (#9). **Bug fixed:** the ported ngram features returned `{}` for degenerate
  (too-short) docs, breaking on short RATIO/sections — now zero-fill against the
  vocab (`fingerprint/features/ngrams.py`).

## Biggest conceptual shift vs the plan

The plan framed RATIO as a **cleaner authorship signal** for the rapporteur. In
practice the dissent→RATIO application is **cross-genre** (dissent voice vs
majority "Ústavní soud" voice), and cross-genre transfer is weak (RATIO 15.2% vs
in-genre DIS 88.6%). The experiment therefore became **as much methodological as
substantive**: we cannot cleanly separate genre-transfer from ghostwriting from
this design alone. The dissent-trained anchor (trusted labels) is retained; a
considered **RATIO→RATIO** reframe was **rejected** (it would forfeit trusted
authorship labels). See the report §7 and the deferred clerk-data direction (§9).

## Self-containment checklist (plan §6)

- [x] No code path reads `subset_disent2.csv` except the one-time seed bootstrap.
- [x] `experiment_03/` has its own `pyproject.toml` + entrypoints (`scripts/01..06`).
- [x] Plenary included; rapporteur + clean dissent come from the record-card scrape.
- [x] Every artifact regenerable: scrape → annotate → dataset → analysis.
- [x] Module imports renamed from `annotation.*` to `llm_annotate.*`.
- [x] UDPipe model documented + gitignored (symlinked into `models/`).
- [ ] Clean `experiment_03` poetry env (currently reuses experiment-02's). *(loose end)*

## Deferred / loose ends

- **Clerk-assignment ground truth** (reference repo `ccc_clerks.R`) — the only
  clean way to decompose genre vs ghostwriting. Top follow-up.
- **XGBoost** pass (`06_analyze.py --classifier both`).
- **Human review** of the 72 machine-annotated plenary decisions.
- **`Pl-42-17_1`** refetch (no ECLI hit; 308/309).
- **Clean `experiment_03` poetry env**.
- `subset_disent2.csv` can be removed from the repo root now that the seed list
  is committed (parity reached).
