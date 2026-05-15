# NALUS Constitutional Court Decision Scraper

## Executive Summary

This scraper extracts decision texts and metadata from the NALUS database (nalus.usoud.cz), the official database of the Czech Constitutional Court. It was developed to recreate the ghostwriting dataset from the source rather than converting existing CSV data, enabling full transparency and reproducibility of the research.

## Data Format

The scraper produces a new data representation optimized for research use:

- **Decision Text**: Individual HTML files (`decisions/*.html`) containing the full decision text with original formatting
- **Metadata**: Single JSON file (`metadata.json`) with structured metadata for all decisions

This separation enables:
- Easy text extraction and NLP processing from HTML files
- Efficient metadata queries from JSON
- Git-friendly version control (HTML and JSON are diffable)
- Clear separation between content and metadata

## Technical Implementation

### Scraper Details

- **Language**: Python 3.10+
- **Dependencies**: `requests`, `beautifulsoup4` (managed via Poetry)
- **Endpoint**: Direct NALUS API (`GetText.aspx?sz={spis_zn}`)
- **Identifier Format**: `SENAT-NUMBER-VERSION_SUBVERSION` (e.g., `3-2329-21_1`)

### Key Features

- **Format Conversion**: Converts ECLI identifiers to NALUS spisová značka format
- **Batch Processing**: Scrapes all decisions from a list file
- **Metadata Extraction**: Automatically extracts date, panel composition, and dissent status
- **Error Handling**: Graceful handling of timeouts and connection errors

## Results

- **Total Decisions**: 236 unique decisions identified from the dataset
- **Successfully Scraped**: 235 decisions (99.6% success rate)
- **Failed**: 1 decision (timeout: `4-2884-22_2`)
- **Output Size**: ~9 MB (HTML files + metadata)

## Usage

### Setup

```bash
cd scrapers
poetry install
```

### Extract Identifiers

```bash
poetry run python extract_spis_zn.py
```

Extracts spisová značka from ECLI identifiers in the CSV dataset.

### Run Scraper

```bash
poetry run python nalus_scraper.py
```

Scrapes all decisions from `spis_zn_list.txt` and saves to `data/nalus/`.

## Example Files

See the `example/` directory for a sample decision:
- `3-2329-21_1.html` - Full decision text (ECLI:CZ:US:2023:3.US.2329.21.1)
- `metadata_example.json` - Corresponding metadata

## File Structure

```
scrapers/
├── data/
│   └── nalus/
│       ├── decisions/        # 235 HTML decision files
│       └── metadata.json      # Metadata for all decisions
├── example/                  # Sample files for reference
│   ├── 3-2329-21_1.html
│   └── metadata_example.json
├── nalus_scraper.py          # Main scraper script
├── extract_spis_zn.py        # ECLI to spis_zn converter
├── spis_zn_list.txt          # List of identifiers to scrape
├── pyproject.toml            # Poetry configuration
└── README.md                 # This file
```

## Advantages Over CSV Format

1. **Source Transparency**: Data scraped directly from official database
2. **Rich Formatting**: HTML preserves original document structure
3. **Metadata Separation**: JSON enables efficient querying
4. **Version Control**: Git-friendly diffing of HTML and JSON
5. **Reproducibility**: Full scraping pipeline is documented and versioned

## Contact

For questions or issues, please refer to the project repository.
