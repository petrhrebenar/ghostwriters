# Assessment: Scraping rozhodnuti.justice.cz

## Overview

https://rozhodnuti.justice.cz/ is the Ministry of Justice's public database of decisions from district, regional, and appellate courts. Unlike NALUS, this database provides a proper REST API for accessing decisions.

## API Structure

**Hierarchical endpoints:**
- `/api/opendata` - Summary of available years
- `/api/opendata/<year>` - Summary of months in that year
- `/api/opendata/<year>/<month>` - Summary of days in that month
- `/api/opendata/<year>/<month>/<day>` - Decisions published on that day
- `/api/finaldoc/{uuid}` - Full decision text (structured JSON)

## Data Volume

**~580,000 decisions total (2020-2026):**
- 2020: 797
- 2021: 150,940
- 2022: 181,864
- 2023: 85,464
- 2024: 61,209
- 2025: 71,537
- 2026: 29,246

Results are paginated at 100 decisions per page.

## Decision Data Format

**Metadata (from list endpoint):**
- Case number, court, author, ECLI
- Decision date, publication date
- Subject, keywords, cited provisions
- Link to full document

**Full decision (from finaldoc endpoint):**
Structured JSON with:
- `header` - Parties and case information
- `verdict` - Decision outcome
- `justification` - Full reasoning text
- `information` - Appeal information
- `metadata` - Structured metadata (ECLI, judge, regulations, etc.)

## Implementation Complexity

**Low complexity** - much simpler than NALUS:
- No HTML parsing needed (pure JSON API)
- No authentication required
- Clear pagination (100 per page)
- Well-documented API
- Structured data format

**Main challenges:**
- Volume: 580K decisions (vs 236 for NALUS)
- Rate limiting may be needed
- Storage: Decision texts are substantial
- Need to handle pagination across ~5,800 pages

## Comparison to NALUS

| Aspect | NALUS | rozhodnuti.justice.cz |
|--------|-------|----------------------|
| API | HTML endpoint (GetText.aspx) | REST API with JSON |
| Format | HTML (requires parsing) | Structured JSON |
| Authentication | None | None |
| Pagination | N/A | Yes (100/page) |
| Volume | 236 decisions | ~580,000 decisions |
| Complexity | Medium (format conversion) | Low (direct API) |

## Implementation Estimate

**Time:** 1-2 days for basic scraper
**Effort:** Low (much simpler than NALUS)
**Risk:** Minimal (stable, documented API)

The scraper would be straightforward:
1. Iterate through years/months/days
2. Fetch decision lists with pagination
3. Extract UUIDs
4. Fetch full decision texts
5. Save as JSON (no HTML conversion needed)

## Recommendation

This is a low-risk, high-value scraping target due to:
- Official Ministry of Justice API
- Structured, machine-readable data
- Large volume of decisions (580K vs 236 for NALUS)
- Simple implementation compared to NALUS
