# LLM annotation — RATIO / DIS / CON via line ranges

LLM-assisted replacement for manual span annotation. Instead of a human reading
every Constitutional Court decision and inserting `<RATIO>/<DIS>/<CON>` tags, an
LLM (default: Claude Sonnet via OpenRouter) marks the spans as **line ranges**,
and a human only reviews/corrects the result.

## Why line ranges (not reproduced text)

Each decision is split into numbered lines (one paragraph ≈ one line). The model
returns `{"label", "start_line", "end_line", "author?"}` rather than echoing
text. This is:

- **robust** to messy/old formatting (no clean paragraph reconstruction needed),
- **cheap** (the model emits a few integers, not the whole document),
- **trivial to validate** (in-bounds, non-overlapping),
- **directly useful** for stylometry (we reconstruct exact per-label text).

## Data flow

```
data/
  01_scraped/<id>.json      scraper output (full_text + metadata)
  02_cleaned/<id>.json      canonical {id, meta, lines:[...]}
            /<id>.txt        metadata header + numbered lines (human)
            /<id>.html       numbered-line preview (browser)
  03_prompts/<id>.json      exact {model, params, system, user} sent
  04_responses/<id>.json    raw reply + parsed/validated spans + usage
  05_tagged/<id>.txt        <RATIO>/<DIS>/<CON> tags inserted (feeds extract_tags.py)
           /<id>.html       colour-coded review view
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
(use `--overwrite` to force) and reports per-doc span counts + validation flags.

## Human review

1. Open `data/05_tagged/<id>.html` to eyeball the colour-coded spans.
2. Fix anything wrong by editing `data/05_tagged/<id>.txt` (move a tag a line
   up/down, add/remove a span).
3. Run the existing `annotation/extract_tags.py` on `data/05_tagged/` to produce
   final structured JSON/JSONL.

## Notes

- Model is configurable (`--model`); default `anthropic/claude-sonnet-4.6`.
- The LLM's only job is span boundaries — it is **not** asked to name authors
  (that's a second task that risks degrading boundary quality).
- DIS/CON detection is structurally explicit and expected to be reliable;
  RATIO boundaries are fuzzier but tolerant for stylometry.
- Author recovery is downstream: the dissenting judge's name is both inside the
  span (its heading line) and in the scraped metadata `separate_opinion`.
- Everything under `data/` is generated and gitignored.
