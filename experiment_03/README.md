# Experiment 03 — LLM-annotated authorship / ghostwriting analysis

Self-contained, end-to-end study of authorship in Czech Constitutional Court
(ÚS) decisions. From a NALUS scrape we LLM-segment each decision into a
9-category structural scheme, isolate the court's own reasoning (**RATIO**) and
the separate opinions (**DIS/CON**), and rerun the experiment-02 per-writer
classification on that **cleaner** signal — training per-judge classifiers on
dissents and scoring the RATIO authored under each rapporteur.

**Self-containment:** everything (spec, code, data, results, report) lives here.
No dependency on `subset_disent2.csv` or any externally packaged dataset; every
artifact is regenerable from code + a NALUS scrape. See
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for design and rationale.

## Pipeline

Code mirrors the numbered `data/` stages. Run everything from `experiment_03/`.

| Stage | Code | Input → Output |
|------|------|----------------|
| 0 corpus | `scrape/build_seed_list.py` | bootstrap `corpus/seed_spis_zn.txt` (309 file ids: 236 senate + 73 plenary) |
| 1 scrape | `scrape/` | NALUS record card + body → `data/01_scraped/<id>.json` (rapporteur, separate_opinion, full_text) |
| 2 annotate | `llm_annotate/` | body → 9-category spans → `data/05_tagged/<id>.txt` (+ human review) |
| 3a spans | `extract_tags.py` | `data/05_tagged/` → `data/07_spans/<id>.json` |
| 3b dataset | `build_dataset.py` + `authorship.py` | spans + metadata → `data/dataset/{decisions,dissents}.parquet` |
| 4 analyze | `fingerprint/` + `scripts/06_analyze.py` | dataset → `outputs/` (LOO-CV + authorship probabilities) |

The 9 categories (`HEAD REC PART PROC FACT RATIO DISP DIS CON`) are defined in
[docs/anotacni_schema_US.md](docs/anotacni_schema_US.md); annotator návod and a
worked example are in [docs/](docs/).

## Status (2026-06-17)

- **Done & validated:** seed list (309), the LLM annotation of the original 236
  decisions (human-reviewed, see `data/06_reviewed/feedback.md`), span
  extraction (236/236, 0 tag problems), dataset build + authorship attribution
  (235 separate opinions → **13 judges with ≥5 opinions**; 235/236 decisions
  carry RATIO, median ~1430 RATIO words), the ported analysis package, and the
  rewritten `data_loader`.
- **Pending (critical path):** the record-card scraper (`scrape/`, Option B —
  ASP.NET session) which supplies the **rapporteur** and clean **separate_opinion**
  names; annotating the 73 new plenary decisions; the metadata join; and running
  `scripts/06_analyze.py` end-to-end. Until the scraper lands, author labels fall
  back to the (consistent but genitive) span-heading names and rapporteur is null.

## Setup

```bash
cd experiment_03
poetry install
export OPENROUTER_API_KEY=sk-or-...     # for the annotation stage (Claude Sonnet)
```

UDPipe Czech model (analysis stage), gitignored — download once into `models/`:
https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131
→ `models/czech-pdt-ud-2.5-191206.udpipe`

## Notes

- `data/`, `outputs/*.pkl`, and `models/` are gitignored (regenerable / large);
  `corpus/seed_spis_zn.txt` is committed (the corpus's source of truth,
  bootstrapped once from the legacy CSV — see `scrape/build_seed_list.py`).
- The LLM only emits segment boundaries; DIS/CON author recovery is downstream
  (`authorship.py`): the name is in the span heading and cross-checked against
  scraped `separate_opinion`.
- `prepare_for_tagging.py` is the superseded manual-prep script (replaced by
  `llm_annotate/clean.py`); kept only for reference.
