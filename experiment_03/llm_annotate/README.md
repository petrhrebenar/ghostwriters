# LLM annotation — 9-category structural segmentation via line ranges

LLM-assisted replacement for manual annotation of Constitutional Court
decisions. An LLM (default: Claude Sonnet via OpenRouter) **segments the whole
document** into the 9 structural categories of `Poznámky/anotacni_schema_US.md`,
and a human reviews/corrects the result.

**Categories** (canonical order): `HEAD` header · `REC` applicant's petition ·
`PART` other parties' statements · `PROC` procedural prerequisites · `FACT`
factual background (court's neutral voice) · `RATIO` court's own reasoning
(**primary target**) · `DISP` disposition/verdict · `DIS` dissent · `CON`
concurrence.

## How it works (exhaustive partition via boundaries)

Each decision is split into numbered lines (one paragraph ≈ one line). The model
returns an **ordered list of segment boundaries** `{"start_line", "label"}`, each
running until the next. This is:

- **exhaustive** — every line gets exactly one label (a gap-free, non-overlapping
  partition); categories may repeat (e.g. two `DISP` spans) or be absent,
- **partition-safe by construction** — boundaries can't overlap or leave gaps,
- **robust** to messy/old formatting (no clean paragraph reconstruction needed),
- **cheap** (the model emits boundaries, not the whole document),
- **directly useful** for stylometry (we reconstruct exact per-label text).

Classification follows **predominant function and speaker, not section heading**
(e.g. an applicant's paragraph inside a "Statements" section is still `REC`).

## Data flow

```
data/
  01_scraped/<id>.json      scraper output (full_text + metadata)
  02_cleaned/<id>.json      canonical {id, meta, lines:[...]}
            /<id>.txt        metadata header + numbered lines (human)
            /<id>.html       numbered-line preview (browser)
  03_prompts/<id>.json      exact {model, params, system, user} sent
  04_responses/<id>.json    raw reply + expanded/validated spans + usage
  05_tagged/<id>.txt        9-category tags inserted (feeds extract_tags.py)
           /<id>.html       colour-coded review view (9-colour legend)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r annotation/llm_annotate/requirements.txt
export OPENROUTER_API_KEY=sk-or-...
```

Run everything from the **repo root** (modules use package imports).

## Usage

End to end (start small to sanity-check, then scale up):

```bash
# dry run: ingest+clean+prompts, estimate tokens, NO API calls
python -m annotation.llm_annotate.run_all --limit 5 --dry-run

# real run on 5 docs
python -m annotation.llm_annotate.run_all --limit 5

# full corpus
python -m annotation.llm_annotate.run_all
```

Or stage by stage:

```bash
python -m annotation.llm_annotate.ingest          # 01 (from scrapers/nalus_v2/data/decisions)
python -m annotation.llm_annotate.clean           # 02
python -m annotation.llm_annotate.build_prompts   # 03  (--model ...)
python -m annotation.llm_annotate.run_llm         # 04  (resumable; skips done)
python -m annotation.llm_annotate.render          # 05
```

`run_llm` is **resumable** — it skips decisions already in `04_responses/`
(use `--overwrite` to force) and reports per-doc segment counts, coverage, and
validation flags. It expands the model's boundaries into a verified partition
(merging adjacent same-label runs; flagging any anomalies).

## Human review

1. Open `data/05_tagged/<id>.html` to eyeball the colour-coded spans.
2. Fix anything wrong by editing `data/05_tagged/<id>.txt` (move a tag a line
   up/down, add/remove a span).
3. Run the existing `annotation/extract_tags.py` on `data/05_tagged/` to produce
   final structured JSON/JSONL.

## Notes

- Model is configurable (`--model`); default `anthropic/claude-sonnet-4.6`.
- The LLM's only job is segment boundaries + labels — it is **not** asked to
  name authors. DIS/CON author recovery is downstream: the judge's name is both
  inside the span (its heading line) and in scraped metadata `separate_opinion`.
- Reliability varies by category: `HEAD`/`DISP`/`DIS`/`CON` are formulaic and
  reliable; `FACT`↔`RATIO`/`PROC` borders and especially `REC`↔`PART`
  (speaker attribution) are the main error sources — hence the human review.
- `RATIO` is the primary stylometric target; exhaustive segmentation sharpens
  its boundaries by forcing explicit neighbours (FACT→RATIO→DISP).
- A previous 3-category (RATIO/DIS/CON) version is tagged `annotate-3cat-v1`
  with a data snapshot in `.archive/`.
- Everything under `data/` is generated and gitignored.
