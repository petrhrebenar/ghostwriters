# Annotation — RATIO / DIS / CON span tagging

Prepares Constitutional Court decisions for **tag-based annotation** of the spans
relevant to authorship analysis, and parses the tagged results back into
structured data.

Only three span types are annotated (see `ANOTACE_navod.md` for the rationale):

- `RATIO` — the court's own legal reasoning (majority opinion).
- `DIS` — dissenting opinion (disagrees with the verdict).
- `CON` — concurring opinion (agrees with verdict, differs on reasoning).

This intentionally drops the full structural schema: segmentation of the older,
inconsistently formatted decisions proved brittle, and only `RATIO`/`DIS`/`CON`
are needed for the authorship work (`RATIO` → majority-author stylometry,
`DIS`/`CON` → per-writer labelled text).

## Workflow

1. **Prepare clean text** (no tags inserted, to avoid biasing the annotator):
   ```bash
   # run inside the nalus_v2 poetry env (provides beautifulsoup4)
   cd scrapers/nalus_v2
   poetry run python ../../annotation/prepare_for_tagging.py \
       --input  data/decisions \
       --output ../../annotation/decisions_to_annotate
   ```
   Produces one readable `.txt` per decision (metadata header + paragraph body).

2. **Annotate** — the annotator wraps spans with `<RATIO>…</RATIO>`,
   `<DIS>…</DIS>`, `<CON>…</CON>`. Instructions: `ANOTACE_navod.md`;
   worked example: `priklad_anotace.txt`.

3. **Extract spans** from the returned files:
   ```bash
   python annotation/extract_tags.py \
       --input  annotation/decisions_to_annotate \
       --output annotation/annotated
   ```
   Writes per-decision JSON + `annotated.jsonl`, and reports unbalanced/nested tags.

## Files

| File | Purpose |
|------|---------|
| `prepare_for_tagging.py` | Raw decision JSON → clean taggable text. |
| `extract_tags.py` | Tagged text → structured JSON/JSONL (with validation). |
| `ANOTACE_navod.md` | Annotator instructions (Czech). |
| `priklad_anotace.txt` | Example of a completed annotation. |
| `decisions_to_annotate/` | Generated clean text, the annotation base (236 files, gitignored). |

## Notes

- `decisions_to_annotate/` (and `annotated/`) are **gitignored** — they are
  generated from the scraped data (`scrapers/nalus_v2/data/`, also gitignored)
  via `prepare_for_tagging.py`. Regenerate with the command above.
- The annotator-facing bundle (clean text + návod + schema + example) is shared
  via `/srv/fileshare/tp/anotace_us/`.
