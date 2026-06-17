#!/usr/bin/env python3
"""Export CCC decision JSONs into factgenie's data format.

Reads the scraped decision JSONs (scrapers/data/rozhodnuti/decisions/*.json)
and writes, into the factgenie-symlinked `annotations/data/` tree:

  - inputs:  data/inputs/<dataset_id>/<split>.json
             A JSON array of "data" objects (metadata shown as side context),
             consumed by factgenie's `basic.JSONDataset` class.
  - outputs: data/outputs/<dataset_id>/<split>-<setup_id>.jsonl
             One line per example: the annotatable text (justificationText).
  - registers the dataset in data/datasets.yml.

The output `output` field is the text that gets span-annotated in factgenie;
the input `data` object is rendered as a table next to it for context.

Run with a Python that has PyYAML available, e.g. the factgenie venv:
  ~/ufal/factgenie/.venv/bin/python annotations/export_to_factgenie.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("export_to_factgenie")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECISIONS_DIR = REPO_ROOT / "scrapers" / "data" / "rozhodnuti" / "decisions"
DEFAULT_DATA_ROOT = REPO_ROOT / "annotations" / "data"


def format_case_number(case: dict | None) -> str:
    if not case:
        return ""
    senate = case.get("senate", "")
    registry = case.get("registry", "")
    index = case.get("index", "")
    year = case.get("year", "")
    return f"{senate} {registry} {index}/{year}".strip()


def format_judge(solver: dict | None) -> str:
    if not solver:
        return ""
    parts = [
        solver.get("titlesBefore", ""),
        solver.get("firstName", ""),
        solver.get("lastName", ""),
        solver.get("titlesAfter", ""),
    ]
    return " ".join(p for p in parts if p).strip()


def build_example(decision: dict) -> tuple[dict, str] | None:
    """Return (input_data, output_text) or None if the decision is unusable."""
    text = (decision.get("justificationText") or "").strip()
    if not text:
        return None

    meta = decision.get("metadata", {}) or {}
    data = {
        "uuid": decision.get("uuid", ""),
        "ecli": meta.get("ecli", ""),
        "court": meta.get("courtCode", ""),
        "judge": format_judge(meta.get("solver")),
        "case_number": format_case_number(meta.get("caseNumber")),
        "case_subject": meta.get("caseSubject", ""),
        "type": meta.get("type", ""),
        "decision_at": meta.get("decisionAt", ""),
        "verdict": (decision.get("verdictText") or "").strip(),
    }
    return data, text


def export(
    decisions_dir: Path,
    data_root: Path,
    dataset_id: str,
    split: str,
    setup_id: str,
) -> None:
    files = sorted(decisions_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No decision JSONs found in {decisions_dir}")

    inputs: list[dict] = []
    outputs: list[str] = []
    skipped: list[str] = []

    for path in files:
        with open(path, encoding="utf-8") as f:
            decision = json.load(f)
        built = build_example(decision)
        if built is None:
            skipped.append(path.name)
            continue
        data, text = built
        inputs.append(data)
        outputs.append(text)

    if not inputs:
        raise SystemExit("No usable decisions (all missing justificationText).")

    # Write inputs: data/inputs/<dataset_id>/<split>.json
    input_dir = data_root / "inputs" / dataset_id
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / f"{split}.json"
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(inputs, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %d inputs -> %s", len(inputs), input_path)

    # Write outputs: data/outputs/<dataset_id>/<split>-<setup_id>.jsonl
    output_dir = data_root / "outputs" / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}-{setup_id}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, text in enumerate(outputs):
            record = {
                "dataset": dataset_id,
                "split": split,
                "setup_id": setup_id,
                "example_idx": idx,
                "output": text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d outputs -> %s", len(outputs), output_path)

    # Register in datasets.yml (merge with any existing entries)
    datasets_path = data_root / "datasets.yml"
    registry: dict = {}
    if datasets_path.exists():
        with open(datasets_path, encoding="utf-8") as f:
            registry = yaml.safe_load(f) or {}

    registry[dataset_id] = {
        "class": "basic.JSONDataset",
        "description": "CCC court decisions: justification text for span annotation.",
        "enabled": True,
        "splits": [split],
    }
    with open(datasets_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(registry, f, allow_unicode=True, sort_keys=False)
    logger.info("Registered dataset '%s' -> %s", dataset_id, datasets_path)

    if skipped:
        logger.warning("Skipped %d file(s) with no justificationText: %s", len(skipped), ", ".join(skipped))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions-dir", type=Path, default=DEFAULT_DECISIONS_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset-id", default="ccc-decisions")
    parser.add_argument("--split", default="test")
    parser.add_argument("--setup-id", default="original")
    args = parser.parse_args()

    export(
        decisions_dir=args.decisions_dir,
        data_root=args.data_root,
        dataset_id=args.dataset_id,
        split=args.split,
        setup_id=args.setup_id,
    )


if __name__ == "__main__":
    main()
