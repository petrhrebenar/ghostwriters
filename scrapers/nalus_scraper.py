#!/usr/bin/env python3
"""
NALUS Constitutional Court Decision Scraper

Scrapes decisions from the NALUS database (nalus.usoud.cz).
Uses the direct GetText.aspx endpoint which doesn't require ViewState.
"""

import requests
from bs4 import BeautifulSoup
import json
import re
from pathlib import Path
from typing import Dict, Optional, List
from urllib.parse import urljoin


class NALUSScraper:
    """Scraper for NALUS Constitutional Court database."""
    
    BASE_URL = "https://nalus.usoud.cz/Search/"
    GETTEXT_ENDPOINT = "GetText.aspx"
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_dir = self.output_dir / "decisions"
        self.decisions_dir.mkdir(exist_ok=True)
        
    def get_decision_text(self, spis_zn: str) -> Optional[str]:
        """
        Fetch decision text by spisová značka (file number).
        
        Args:
            spis_zn: File number in format "1-1056-07_2"
        
        Returns:
            HTML content as string, or None if failed
        """
        # NALUS format: SENAT-NUMBER-VERSION_SUBVERSION (no slashes)
        url = urljoin(self.BASE_URL, f"{self.GETTEXT_ENDPOINT}?sz={spis_zn}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Check if we got actual content (not empty template)
            if len(response.text) < 5000:  # Empty templates are usually very short
                return None
            
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {spis_zn}: {e}")
            return None
    
    def parse_decision_metadata(self, html: str) -> Dict:
        """
        Extract metadata from decision HTML.
        
        Args:
            html: HTML content string
        
        Returns:
            Dictionary with extracted metadata
        """
        soup = BeautifulSoup(html, 'html.parser')
        metadata = {}
        
        # Try to extract ECLI if present
        ecli_match = re.search(r'ECLI:CZ:US:[\d]+:[\w.]+', html)
        if ecli_match:
            metadata['ECLI'] = ecli_match.group(0)
        
        # Try to extract decision date
        date_match = re.search(r'(\d{1,2})\.?\s*(\w+)\s*(\d{4})', html)
        if date_match:
            metadata['date_raw'] = date_match.group(0)
        
        # Try to extract panel members
        panel_pattern = re.search(r'senát.*?(JUDr\.|prof\.|Ing\.|Mgr\.).*?(\s+[\w\s]+?)(?=,|a|soudce)', html, re.IGNORECASE | re.DOTALL)
        if panel_pattern:
            metadata['panel_raw'] = panel_pattern.group(0)
        
        # Check if it's a dissenting opinion
        if 'odlišné stanovisko' in html.lower() or 'disent' in html.lower():
            metadata['is_dissent'] = True
        
        return metadata
    
    def save_decision(self, spis_zn: str, html: str, metadata: Dict) -> str:
        """
        Save decision as HTML file.
        
        Args:
            spis_zn: File number
            html: HTML content
            metadata: Metadata dictionary
        
        Returns:
            Path to saved file
        """
        # Sanitize filename
        safe_spis = spis_zn.replace('/', '_').replace('\\', '_')
        filename = f"{safe_spis}.html"
        filepath = self.decisions_dir / filename
        
        # Save HTML
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return str(filepath)
    
    def scrape_decision(self, spis_zn: str) -> Optional[Dict]:
        """
        Scrape a single decision.
        
        Args:
            spis_zn: File number (e.g., "3-2329-21")
        
        Returns:
            Dictionary with 'html_file', 'metadata', 'spis_zn', or None if failed
        """
        print(f"Scraping {spis_zn}...")
        
        html = self.get_decision_text(spis_zn)
        if not html:
            print(f"Failed to fetch {spis_zn}")
            return None
        
        metadata = self.parse_decision_metadata(html)
        metadata['spis_zn'] = spis_zn
        
        html_file = self.save_decision(spis_zn, html, metadata)
        
        return {
            'spis_zn': spis_zn,
            'html_file': html_file,
            'metadata': metadata
        }
    
    def scrape_batch(self, spis_zn_list: List[str]) -> Dict:
        """
        Scrape multiple decisions.
        
        Args:
            spis_zn_list: List of file numbers
        
        Returns:
            Dictionary mapping spis_zn to scrape results
        """
        results = {}
        
        for spis_zn in spis_zn_list:
            result = self.scrape_decision(spis_zn)
            if result:
                results[spis_zn] = result
        
        return results
    
    def save_metadata_json(self, results: Dict, filename: str = "metadata.json"):
        """
        Save all metadata to JSON file.
        
        Args:
            results: Dictionary of scrape results
            filename: Output filename
        """
        metadata_dict = {}
        
        for spis_zn, result in results.items():
            metadata_dict[spis_zn] = result['metadata']
            metadata_dict[spis_zn]['html_file'] = result['html_file']
        
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Saved metadata to {filepath}")


def main():
    """Scrape all decisions from spis_zn list."""
    scraper = NALUSScraper(output_dir="data/nalus")
    
    # Load spis_zn list from file
    spis_zn_file = "spis_zn_list.txt"
    with open(spis_zn_file, 'r', encoding='utf-8') as f:
        spis_zn_list = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(spis_zn_list)} spis_zn to scrape")
    
    results = scraper.scrape_batch(spis_zn_list)
    scraper.save_metadata_json(results)
    
    print(f"\nScraped {len(results)} decisions successfully.")
    print(f"Failed: {len(spis_zn_list) - len(results)}")


if __name__ == "__main__":
    main()
