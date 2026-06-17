# Experiment 03 — Implementation Plan

LLM-annotated, self-contained authorship / ghostwriting analysis of Czech
Constitutional Court (ÚS) decisions.

## 0. Status & how to read this

This document is the **plan** for the next session. Nothing in the
scraping / analysis rewrite has been implemented yet. What already exists in
this directory (moved here from the old top-level `annotation/`) is the
**LLM annotation pipeline** (`llm_annotate/`) plus its local data
(`data/`, gitignored). The new work is: (a) a richer scraper that captures
the judge **rapporteur** and **dissenting judges**, and (b) rebuilding the
experiment-02 analysis so it runs **entirely on our own data** with no
dependency on `subset_disent2.csv`.

---

## 1. Goal & guiding principle

**Goal.** A single, self-contained `experiment_03/` that, from scratch:

1. scrapes ÚS decisions from NALUS **including** structured metadata
   (rapporteur, dissenting judges, formation, dates, výrok, …);
2. LLM-annotates each decision into the 9-category structural scheme
   (`HEAD REC PART PROC FACT RATIO DISP DIS CON`);
3. derives a clean stylometric dataset from those annotations
   (RATIO = court's reasoning authored under the rapporteur; DIS/CON =
   separate opinions authored by named judges);
4. reruns the experiment-02 per-writer classification / ghostwriting
   analysis on that dataset.

**Guiding principle — self-containment.** No reliance on
`subset_disent2.csv` or any externally packaged dataset. The whole point of
this exercise is to remove the dependency on someone else's pre-built data
and reproduce every input ourselves. Every artifact must be regenerable from
code + a NALUS scrape.

**Why this is better than experiment-02's input.** experiment-02 trains on
the `separate_opinion` field and scores the *entire* decision text. Full
decision text is contaminated: it contains recapitulation of the petition
(REC), other parties' statements (PART), procedural boilerplate (PROC) and
factual background (FACT) — **none of which is authored by the rapporteur**.
LLM annotation lets us isolate **RATIO** (the court's own reasoning, the real
authorship signal for the deciding judge) and **DIS/CON** (separate-opinion
prose, attributable to a named judge). Cleaner signal in, cleaner attribution
out.

---

## 2. What we discovered (facts that drive the design)

These are verified findings from the previous session; they are the rationale
for the scraper rewrite.

### 2.1 NALUS endpoints

- **`GetText.aspx?sz=<file-id>`** — what the current `nalus_v2` scraper uses.
  Returns **only the decision body HTML**. No structured metadata: no
  rapporteur, no clean dissent list. Confirmed by fetching `I.ÚS 1056/07`.
- **`ResultDetail.aspx?id=<N>`** — the record-card page. Contains a
  `table.recordCardTable` with **all** structured metadata. A bare GET of
  `ResultDetail.aspx?id=130000` bounced to the search form → the page is a
  **stateful ASP.NET WebForms** app and needs a live session
  (`ASP.NET_SessionId` cookie + `__VIEWSTATE` / `__EVENTVALIDATION`), and the
  `id` is discovered from search-result links, not guessable.

### 2.2 The record card contains everything we need (proven by reference)

The reference project `stepanpaulik/ccc_dataset` (`scripts/ccc_web_scraping.R`)
is the working blueprint. It drives a real browser (RSelenium/Firefox):

1. `Search.aspx` → set `decidedFrom`, submit, paginate results (20/page),
   collect `a.resultData` hrefs → `ResultDetail.aspx?id=<N>`;
2. for each detail page, parse `table.recordCardTable` → a label→value map.

Fields available on the card (Czech label → our field):

| NALUS label | field | notes |
|---|---|---|
| `Soudce zpravodaj` | `judge_rapporteur_name` | **critical**, missing today |
| `Odlišné stanovisko` | `separate_opinion` (dissenting judges) | **critical**, clean — no regex |
| `Identifikátor evropské judikatury` | `doc_id` / ECLI | |
| `Spisová značka` | `case_id` | |
| `Datum rozhodnutí` / `podání` / `vyhlášení` | dates | |
| `Forma rozhodnutí` / `Typ řízení` | type_decision / type_proceedings | |
| `Typ výroku` | type_verdict | granted/rejected, grounds |
| `Navrhovatel`, `Dotčený orgán`, `Napadený akt`, `Předmět řízení`, `Věcný rejstřík`, `Dotčené … předpisy` | misc | nice-to-have |

The R code reorders names (`word(.,2) word(.,1)`) and joins
`judge_rapporteur_name` against a `ccc_judges` table for `judge_rapporteur_id`.
We can replicate the reorder; the id-join is optional (we can build our own
judge table from observed names).

### 2.3 Current scraper limitations (root causes)

- **`judge_rapporteur_name` is always `None`** — `scrapers/nalus_v2/parse.py`
  declares the field but never assigns it (GetText has no such field anyway).
- **`separate_opinion` is buggy** — extracted by a fragile regex over body
  text; mis-captures non-names and misses diacritics.
- **Plenary decisions are silently dropped** — `scrapers/extract_spis_zn.py`
  `ecli_to_spis_zn()` regex `ECLI:CZ:US:(\d+):(\d+)\.US\.…` requires a *digit*
  senate code, so `Pl.US` (plenum) never matches. This is why we have 236
  decisions instead of the full 309 (73 plenary). **Trivial fix**: widen the senate group
  to accept `Pl`. Plenary decisions are full-court rulings and are **not**
  disqualified from the analysis — they should be included.

---

## 3. Scraping approach — options

We must scrape the **record card** (only source of rapporteur + clean
dissent). Three transports were considered.

### Option A — Browser automation (Selenium / Playwright)
Mirror the reference R code: drive a headless browser through `Search.aspx`,
paginate, collect `id`s, visit each `ResultDetail.aspx?id=`, parse the card.

- **Pros:** proven to work; robust to ASP.NET statefulness; the session,
  cookies, viewstate and pagination postbacks are handled by the browser.
- **Cons:** heavy dependency (browser + driver); slower. Irrelevant at our
  scale (~309 records).
- **Effort:** low (blueprint exists). **Risk:** low.

### Option B — `requests`/`httpx` session emulating the ASP.NET postbacks  ✅ CHOSEN
Pure-HTTP session: GET `Search.aspx` (capture session cookie + hidden
`__VIEWSTATE`/`__VIEWSTATEGENERATOR`/`__EVENTVALIDATION`), POST the search
form (by date range or spisová značka), parse result links for `id`s, GET each
`ResultDetail.aspx?id=` on the same session, parse `table.recordCardTable`
with BeautifulSoup / lxml.

- **Pros:** lightweight; matches the existing `nalus_v2` (requests-based)
  design; no browser infra; easy to run in CI / on a server.
- **Cons:** ASP.NET viewstate/eventvalidation handshake and paginated
  postbacks are fiddly and can break if the form changes.
- **Effort:** medium. **Risk:** medium.
- **Note:** the library doesn't have to be BeautifulSoup — `lxml` /
  `selectolax` are fine. Pick whatever parses the card table cleanly.

### Option C — keep GetText + regex (rejected)
Cannot yield the rapporteur at all; dissent extraction stays fragile. Rejected.

**Decision: implement Option B**, but structure the scraper so the
**parsing layer is transport-agnostic** (input = card HTML → output = metadata
dict). That way, if the ASP.NET handshake proves too brittle, we can drop in
Option A (Selenium/Playwright) as the fetch layer **without touching the
parser**. Build Option A as the documented fallback.

### Option B — concrete steps & risk notes
1. `session = requests.Session()`; GET `Search.aspx`; scrape hidden fields +
   cookie.
2. POST search. Two possible drivers:
   - **by date range** (`decidedFrom`, like the reference) → walk all results;
     or
   - **by spisová značka** one at a time (we already have `spis_zn_list.txt`)
     → smaller, more targeted requests. Preferred for incremental re-runs.
   - **Risk:** must discover the exact control names
     (`ctl00$MainContent$…`) and submit valid `__EVENTVALIDATION`. Inspect the
     live form first.
3. Parse the results page → `ResultDetail.aspx?id=<N>` links (and handle
   pagination postbacks if using date range).
4. GET each detail page on the session; parse `table.recordCardTable` into a
   dict; normalize (name reorder, date parsing, formation from ECLI prefix,
   split multi-value cells on newlines).
5. Persist one JSON per decision: `{ metadata…, full_text }`. `full_text` can
   continue to come from `GetText.aspx` (we already do this well) or from the
   detail page's `DocContent` node — keep the existing GetText fetch to avoid
   rework.
6. **Rate-limit & cache** (reuse `nalus_v2/fetch.py` delay/retry logic); cache
   raw HTML so re-parsing never re-hits the network.

---

## 4. Target directory layout (self-contained)

```
experiment_03/
  IMPLEMENTATION_PLAN.md        <- this file
  README.md                     <- (to update) experiment-03 overview
  scrape/                       <- NEW: record-card scraper (Option B)
    fetch.py                    <- session + GetText (transport)
    search.py                   <- ASP.NET search/pagination -> ids
    parse_card.py               <- recordCardTable HTML -> metadata dict (transport-agnostic)
    pipeline.py                 <- orchestrate scrape -> data/01_scraped/<id>.json
  llm_annotate/                 <- EXISTING 9-category annotation pipeline
    ingest.py clean.py build_prompts.py run_llm.py render.py
    common.py prompts.py run_all.py extract_tags? …
  extract_tags.py               <- EXISTING: tagged text -> spans
  build_dataset.py              <- NEW: annotations + metadata -> analysis dataframe
  fingerprint/                  <- analysis pkg (ported from experiment_02/src/fingerprint)
    data_loader.py              <- REWRITTEN: load our dataset, not subset_disent2.csv
    preprocessing.py features/ classifiers.py evaluation.py feature_importance.py
  scripts/
    run_scrape.py run_annotate.py run_pipeline.py   <- end-to-end entrypoints
  data/                         <- gitignored (local, ~92 MB now)
    01_scraped/  02_cleaned/  03_prompts/  04_responses/  05_tagged/  06_reviewed/
    dataset/                    <- NEW: derived analysis tables (parquet/csv)
  outputs/                      <- results, models, reports (pkls gitignored)
  pyproject.toml                <- self-contained deps (port from experiment_02)
  models/ or model path note    <- UDPipe czech-pdt-ud-2.5 (gitignored, download per README)
```

Notes:
- Code referencing `annotation.llm_annotate.*` module paths and the
  `run_all.py` docstring still say `annotation` — **update the package
  name/imports to `experiment_03`** (or make `experiment_03` an installable
  package). `common.py` derives `DATA_DIR` from `__file__`, so the data path
  already follows the move; only import strings need fixing.
- UDPipe model: experiment-02 used `czech-pdt-ud-2.5-191206.udpipe` in
  `models/`. Document the download; keep gitignored.

---

## 5. Stage-by-stage plan

### Stage 1 — Scrape (Option B)
- Fix the plenary regex in the ECLI→spis_zn converter; regenerate the
  `spis_zn` list so plenary (`Pl.ÚS`) decisions are included (309 total: 236 senate + 73 plenary).
- Implement `scrape/` (search → ids → card parse → JSON with metadata +
  full_text). Output to `data/01_scraped/<id>.json`.
- **Acceptance:** every record JSON has non-null `judge_rapporteur_name`
  (where NALUS provides it) and a structured `separate_opinion` list; plenary
  decisions present.

### Stage 2 — LLM annotation (existing)
- Run `llm_annotate` over all scraped decisions (9-category, boundary output,
  Claude Sonnet via OpenRouter). Already validated and human-spot-checked
  (see `data/06_reviewed/feedback.md`: usable as-is; RATIO consistently good).
- Output: `data/05_tagged/<id>.txt` (tags) + `.html` (review view).
- **Acceptance:** every decision tagged; `extract_tags.py` parses all 9 labels
  back to line spans without loss.

### Stage 3 — Build analysis dataset (replaces subset_disent2.csv)
- New `build_dataset.py`: join (a) extracted spans per decision with (b) the
  scraped metadata, to emit a self-contained dataframe roughly mirroring the
  columns `experiment_02/data_loader` expects, plus the cleaner span-based
  text:
  - `doc_id`, `case_id`, `date_decision`, `formation`, `type_decision`
  - `judge_rapporteur_name` (author of RATIO)
  - `separate_opinion` (dissenting/concurring judge names)
  - `ratio_text` — concatenated RATIO spans (court reasoning)
  - `dissent_text` — concatenated DIS spans, per separate-opinion author
  - `concur_text` — CON spans (optional)
- **Authorship mapping decision (open):** the metadata `separate_opinion` may
  list several judges while the LLM marks DIS/CON spans without an author. For
  multi-author dissents we need a rule to attribute a span to a judge (e.g.
  single-dissent decisions are unambiguous; multi-dissent ones may be dropped
  or split by in-text "Odlišné stanovisko soudce X" markers). Decide before
  feature extraction.
- **Acceptance:** dataset reproducible from `data/` with one command; no
  reference to `subset_disent2.csv`.

### Stage 4 — Rerun the experiment-02 analysis
- Port `experiment_02/src/fingerprint/` into `experiment_03/fingerprint/` and
  **rewrite `data_loader.py`** to read the Stage-3 dataset instead of
  `subset_disent2.csv` (currently hard-coded at
  `parents[3]/"subset_disent2.csv"`).
- Keep the rest of the pipeline conceptually identical (steps: load →
  udpipe → features → evaluate → udpipe_decisions → features_decisions →
  apply), but:
  - train per-judge dissent classifiers on `dissent_text` (DIS spans);
  - **score `ratio_text` instead of full decision text** — this is the key
    improvement enabled by annotation;
  - attribute predicted authorship to the `judge_rapporteur_name` to flag
    potential ghostwriting.
- Reuse the same feature sets (function_words, surface, char_ngrams,
  pos_ngrams, morphology) and classifiers (logistic, xgboost) for
  comparability with the experiment-02 baseline.
- **Acceptance:** `scripts/run_pipeline.py` runs end-to-end from `data/` to
  `outputs/` with `--from-step` resume support; results comparable to (and
  hopefully cleaner than) experiment-02.

---

## 6. Self-containment checklist
- [ ] No code path reads `subset_disent2.csv` (or any repo-root CSV).
- [ ] `experiment_03/` has its own `pyproject.toml` / deps and entrypoints.
- [ ] Plenary decisions included; rapporteur + dissent come from the scrape.
- [ ] Every artifact regenerable: scrape → annotate → dataset → analysis.
- [ ] Module imports renamed from `annotation.*` to the experiment-03 package.
- [ ] UDPipe model download documented in README; large/generated files
      gitignored (`data/`, `outputs/*.pkl`, `models/`).

---

## 7. Effort & risk summary
| Work item | Effort | Risk |
|---|---|---|
| Plenary regex fix + regenerate list | minutes | none |
| Option B scraper (search handshake + card parser) | ~0.5–1 day | medium (ASP.NET) |
| Selenium/Playwright fallback (if needed) | ~0.5 day | low |
| Wire annotation over the 73 new plenary decisions | hours | low |
| `build_dataset.py` + authorship mapping rule | ~0.5 day | medium (multi-author dissents) |
| Port + rewrite `data_loader` and rerun analysis | ~0.5–1 day | low–medium |

---

## 8. Open questions — RESOLVED (2026-06-17)
1. **Search driver / corpus source.** Self-containment achieved by committing
   the seed list as a tracked artifact: `scrape/build_seed_list.py` bootstraps
   `corpus/seed_spis_zn.txt` **once** from the legacy CSV (the only place the CSV
   is read), then the scraper reads the committed list. Result: **309** file ids
   = 236 senate + **73 plenary** (the plenary-regex fix; see §2.3). The CSV can
   be removed once parity is reached (§9). A NALUS date-range crawl remains the
   documented path to *regenerate* the list from scratch if ever needed.
2. **Multi-author separate opinions — span-heading attribution.** The annotation
   schema emits one tag-pair per judge, so multi-judge decisions already arrive
   as separate spans. `authorship.py` reads the judge from each span's heading
   ("Odlišné stanovisko soudce <Name>", in genitive) and, when scraped
   `separate_opinion` is present, matches it to the clean nominative name.
   Validated: 233/235 spans named, 13 judges with ≥5 opinions. No spans dropped.
3. **Score RATIO and full text — run both.** Train per-judge on DIS/CON spans;
   score `ratio_text` (the improvement) and, for a like-for-like comparison with
   the experiment-02 baseline, optionally the full decision text too.
4. **`judge_rapporteur_id`:** key on normalized names; build our own lookup from
   observed names only if needed (no external judge table).
5. **Parser library:** BeautifulSoup (already a dep) for the card table; `lxml`
   available as the parser backend.

**Other resolved choices:** keep the human-reviewed 236 annotations and run the
LLM only on the 73 new plenary decisions (annotation is metadata-independent);
**flat** directory layout under `experiment_03/` (no `src/` package).

## 9. Repo-level loose ends
- **`subset_disent2.csv` is still tracked at repo root** (~15 MB). Keep it
  until experiment-03 reaches parity with the experiment-02 baseline, then
  remove/relocate it to complete the self-containment goal.
- **Branch / dir name** is `experiment-03-llm-annotation`; rename if a
  different slug is preferred.
- **UDPipe model** (`czech-pdt-ud-2.5-191206.udpipe`) is not in the repo;
  document its download and keep `models/` gitignored.
