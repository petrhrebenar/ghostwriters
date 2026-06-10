#!/usr/bin/env python3
"""
Pipeline for scraping NALUS decisions with regex-based metadata extraction.
"""

import json
import time
from pathlib import Path
from typing import Dict, List
from fetch import NALUSFetcher
from parse import parse_decision, to_dict


class NALUSPipeline:
    """Pipeline for scraping and parsing NALUS decisions."""
    
    def __init__(self, spis_zn_list: List[str], output_dir: str = "data", delay: float = 0.5):
        """
        Initialize pipeline.
        
        Args:
            spis_zn_list: List of spisová značka to scrape
            output_dir: Directory to save outputs
            delay: Delay between requests in seconds
        """
        self.spis_zn_list = spis_zn_list
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.decisions_dir = self.output_dir / "decisions"
        self.decisions_dir.mkdir(exist_ok=True)
        
        self.inputs_dir = self.output_dir / "inputs"
        self.inputs_dir.mkdir(exist_ok=True)
        
        self.fetcher = NALUSFetcher(delay=delay)
        
        # Progress tracking
        self.progress_file = self.output_dir / "progress.json"
        self.progress = self._load_progress()
        
        # Statistics
        self.stats = {
            "total": len(spis_zn_list),
            "completed": 0,
            "failed": 0,
            "warnings": []
        }
    
    def _load_progress(self) -> Dict:
        """Load progress from file if exists."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data["completed"] = set(data["completed"])
                data["failed"] = set(data.get("failed", []))
                return data
        return {"completed": set(), "failed": set()}
    
    def _save_progress(self):
        """Save progress to file."""
        progress_to_save = {
            "completed": list(self.progress["completed"]),
            "failed": list(self.progress.get("failed", set()))
        }
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_to_save, f, indent=2, ensure_ascii=False)
    
    def _is_complete(self, spis_zn: str) -> bool:
        """Check if a decision has been scraped."""
        return spis_zn in self.progress["completed"]
    
    def _mark_complete(self, spis_zn: str):
        """Mark a decision as completed."""
        self.progress["completed"].add(spis_zn)
        self._save_progress()
    
    def _mark_failed(self, spis_zn: str):
        """Mark a decision as failed."""
        self.progress["failed"].add(spis_zn)
        self._save_progress()
    
    def run(self):
        """Run the pipeline for all spis_zn."""
        print(f"Starting pipeline for {len(self.spis_zn_list)} decisions")
        print(f"Output directory: {self.output_dir}")
        print(f"Rate limiting: {self.fetcher.delay}s delay between requests")
        print()
        
        for i, spis_zn in enumerate(self.spis_zn_list, 1):
            if self._is_complete(spis_zn):
                print(f"[{i}/{len(self.spis_zn_list)}] Skipping {spis_zn} (already complete)")
                continue
            
            print(f"[{i}/{len(self.spis_zn_list)}] Processing {spis_zn}...")
            
            # Fetch
            text = self.fetcher.fetch_decision(spis_zn)
            if not text:
                print(f"  FAILED to fetch {spis_zn}")
                self._mark_failed(spis_zn)
                self.stats["failed"] += 1
                continue
            
            # Parse
            metadata = parse_decision(text, spis_zn)
            
            # Save per-decision JSON
            safe_spis = spis_zn.replace('/', '_').replace('\\', '_')
            json_path = self.decisions_dir / f"{safe_spis}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(to_dict(metadata), f, indent=2, ensure_ascii=False)
            
            # Track warnings
            if metadata._warnings:
                self.stats["warnings"].append({
                    "spis_zn": spis_zn,
                    "warnings": metadata._warnings
                })
                print(f"  Warnings: {', '.join(metadata._warnings)}")
            
            self._mark_complete(spis_zn)
            self.stats["completed"] += 1
            print(f"  Saved to {json_path}")
            
            # Progress report every 50 decisions
            if i % 50 == 0:
                print(f"\n=== Progress: {i}/{len(self.spis_zn_list)} ===")
                print(f"Completed: {self.stats['completed']}, Failed: {self.stats['failed']}")
                print(f"Fetcher stats: {self.fetcher.stats}")
                print()
        
        # Generate JSONL for factgenie
        self._generate_jsonl()
        
        # Save summary
        self._save_summary()
    
    def _generate_jsonl(self):
        """Generate split.jsonl for factgenie consumption."""
        jsonl_path = self.inputs_dir / "split.jsonl"
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for json_file in sorted(self.decisions_dir.glob("*.json")):
                with open(json_file, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                    # Write as JSONL line
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        print(f"\nGenerated JSONL: {jsonl_path}")
    
    def _save_summary(self):
        """Save summary statistics."""
        summary = {
            "total_decisions": self.stats["total"],
            "completed": self.stats["completed"],
            "failed": self.stats["failed"],
            "fetcher_stats": self.fetcher.stats,
            "warnings_count": len(self.stats["warnings"]),
            "warnings": self.stats["warnings"][:10]  # First 10 warnings
        }
        
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"Pipeline complete!")
        print(f"{'='*60}")
        print(f"Total: {self.stats['total']}")
        print(f"Completed: {self.stats['completed']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Fetcher stats: {self.fetcher.stats}")
        print(f"Warnings: {len(self.stats['warnings'])}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*60}")


def load_spis_zn_list(filepath: str) -> List[str]:
    """Load spis_zn list from text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    """Run the pipeline."""
    # Load spis_zn list
    spis_zn_list = load_spis_zn_list("../spis_zn_list.txt")
    print(f"Loaded {len(spis_zn_list)} spis_zn to process")
    
    # Run pipeline
    pipeline = NALUSPipeline(
        spis_zn_list=spis_zn_list,
        output_dir="data",
        delay=0.5  # 0.5s delay = ~2 requests/sec
    )
    pipeline.run()


if __name__ == "__main__":
    main()
