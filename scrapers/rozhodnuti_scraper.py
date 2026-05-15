#!/usr/bin/env python3
"""
Rozhodnuti.justice.cz Scraper

Scrapes decisions from the Ministry of Justice database using their REST API.
Implements rate limiting to avoid overwhelming the server.
"""

import requests
import json
import time
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime


class RozhodnutiScraper:
    """Scraper for rozhodnuti.justice.cz REST API."""
    
    BASE_URL = "https://rozhodnuti.justice.cz/api"
    OPENDATA_ENDPOINT = "/opendata"
    FINALDOC_ENDPOINT = "/finaldoc"
    
    def __init__(self, output_dir: str = "data/rozhodnuti", delay: float = 1.0):
        """
        Initialize scraper.
        
        Args:
            output_dir: Directory to save scraped data
            delay: Delay between requests in seconds (rate limiting)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_dir = self.output_dir / "decisions"
        self.decisions_dir.mkdir(exist_ok=True)
        self.delay = delay
        
        # Track statistics
        self.stats = {
            "total_requests": 0,
            "successful_fetches": 0,
            "failed_fetches": 0,
            "data_losses": []
        }
    
    def _request_with_delay(self, url: str) -> Optional[requests.Response]:
        """
        Make HTTP request with rate limiting delay.
        
        Args:
            url: URL to request
            
        Returns:
            Response object or None if failed
        """
        # Add delay before request (except first request)
        if self.stats["total_requests"] > 0:
            time.sleep(self.delay)
        
        try:
            self.stats["total_requests"] += 1
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            self.stats["successful_fetches"] += 1
            return response
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            self.stats["failed_fetches"] += 1
            return None
    
    def get_years(self) -> Optional[List[Dict]]:
        """
        Get list of available years with decision counts.
        
        Returns:
            List of year information or None if failed
        """
        url = f"{self.BASE_URL}{self.OPENDATA_ENDPOINT}"
        response = self._request_with_delay(url)
        
        if response:
            return response.json()
        return None
    
    def get_months(self, year: int) -> Optional[List[Dict]]:
        """
        Get list of months for a given year.
        
        Args:
            year: Year to query
            
        Returns:
            List of month information or None if failed
        """
        url = f"{self.BASE_URL}{self.OPENDATA_ENDPOINT}/{year}"
        response = self._request_with_delay(url)
        
        if response:
            return response.json()
        return None
    
    def get_days(self, year: int, month: int) -> Optional[List[Dict]]:
        """
        Get list of days for a given year/month.
        
        Args:
            year: Year to query
            month: Month to query
            
        Returns:
            List of day information or None if failed
        """
        url = f"{self.BASE_URL}{self.OPENDATA_ENDPOINT}/{year}/{month}"
        response = self._request_with_delay(url)
        
        if response:
            return response.json()
        return None
    
    def get_decision_list(self, year: int, month: int, day: int, page: int = 0) -> Optional[Dict]:
        """
        Get list of decisions for a specific date.
        
        Args:
            year: Year to query
            month: Month to query
            day: Day to query
            page: Page number (0-indexed, 100 items per page)
            
        Returns:
            Dictionary with decision list or None if failed
        """
        url = f"{self.BASE_URL}{self.OPENDATA_ENDPOINT}/{year}/{month}/{day}?page={page}"
        response = self._request_with_delay(url)
        
        if response:
            return response.json()
        return None
    
    def get_decision_full(self, uuid: str) -> Optional[Dict]:
        """
        Get full decision text and metadata by UUID.
        
        Args:
            uuid: Decision UUID
            
        Returns:
            Dictionary with full decision data or None if failed
        """
        url = f"{self.BASE_URL}{self.FINALDOC_ENDPOINT}/{uuid}"
        response = self._request_with_delay(url)
        
        if response:
            return response.json()
        return None
    
    def save_decision(self, uuid: str, data: Dict) -> str:
        """
        Save decision data to file.
        
        Args:
            uuid: Decision UUID
            data: Decision data dictionary
            
        Returns:
            Path to saved file
        """
        filename = f"{uuid}.json"
        filepath = self.decisions_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def check_data_completeness(self, data: Dict) -> List[str]:
        """
        Check if decision data is complete and flag any losses.
        
        Args:
            data: Decision data dictionary
            
        Returns:
            List of data loss warnings
        """
        warnings = []
        
        # Check required fields
        required_fields = ['uuid', 'header', 'verdict', 'justification', 'metadata']
        for field in required_fields:
            if field not in data:
                warnings.append(f"Missing required field: {field}")
        
        # Check for empty critical sections
        if 'header' in data and not data['header']:
            warnings.append("Empty header section")
        
        if 'verdict' in data and not data['verdict']:
            warnings.append("Empty verdict section")
        
        if 'justification' in data and not data['justification']:
            warnings.append("Empty justification section")
        
        # Check for anonymization markers
        if 'header' in data:
            header_text = json.dumps(data['header'])
            if 'ANON' not in header_text:
                warnings.append("No anonymization markers found in header")
        
        # Check metadata
        if 'metadata' in data:
            metadata = data['metadata']
            if 'ecli' not in metadata:
                warnings.append("Missing ECLI in metadata")
            if 'solver' not in metadata:
                warnings.append("Missing judge information in metadata")
        
        return warnings
    
    def scrape_sample(self, num_decisions: int = 10) -> Dict:
        """
        Scrape a small sample of decisions for testing.
        
        Args:
            num_decisions: Number of decisions to scrape
            
        Returns:
            Dictionary with scrape results and statistics
        """
        print(f"Starting sample scrape of {num_decisions} decisions...")
        print(f"Rate limiting: {self.delay}s delay between requests")
        
        results = {}
        scraped_count = 0
        
        # Get recent year with most decisions (2022 has 181,864)
        year = 2022
        months = self.get_months(year)
        
        if not months:
            print("Failed to get months")
            return {"error": "Failed to get months"}
        
        # Find first month with decisions
        for month_info in months:
            month = month_info['mesic']
            print(f"\nChecking {year}-{month:02d} ({month_info['pocet']} decisions)")
            
            days = self.get_days(year, month)
            if not days:
                continue
            
            # Try to scrape from first few days
            for day_info in days:
                if scraped_count >= num_decisions:
                    break
                
                day = day_info['datum'].split('-')[2]
                print(f"  Checking {year}-{month:02d}-{day} ({day_info['pocet']} decisions)")
                
                # Get first page
                decision_list = self.get_decision_list(year, month, day, page=0)
                if not decision_list:
                    continue
                
                # Scrape decisions from this day
                for item in decision_list.get('items', []):
                    if scraped_count >= num_decisions:
                        break
                    
                    uuid = item['odkaz'].split('/')[-1]
                    ecli = item.get('ecli', 'unknown')
                    
                    print(f"    Scraping {ecli} ({uuid})...")
                    
                    # Get full decision
                    full_data = self.get_decision_full(uuid)
                    if not full_data:
                        print(f"      FAILED to fetch full decision")
                        self.stats["data_losses"].append({
                            "uuid": uuid,
                            "ecli": ecli,
                            "error": "Failed to fetch full decision"
                        })
                        continue
                    
                    # Check data completeness
                    warnings = self.check_data_completeness(full_data)
                    if warnings:
                        print(f"      WARNINGS: {', '.join(warnings)}")
                        self.stats["data_losses"].append({
                            "uuid": uuid,
                            "ecli": ecli,
                            "warnings": warnings
                        })
                    
                    # Save decision
                    filepath = self.save_decision(uuid, full_data)
                    print(f"      Saved to {filepath}")
                    
                    results[uuid] = {
                        "ecli": ecli,
                        "filepath": filepath,
                        "metadata": item,
                        "warnings": warnings if warnings else None
                    }
                    
                    scraped_count += 1
                
                if scraped_count >= num_decisions:
                    break
            
            if scraped_count >= num_decisions:
                break
        
        # Save results summary
        summary = {
            "scrape_date": datetime.now().isoformat(),
            "num_decisions_scraped": scraped_count,
            "target_decisions": num_decisions,
            "statistics": self.stats,
            "results": results
        }
        
        summary_path = self.output_dir / "sample_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"Sample scrape complete!")
        print(f"Scraped: {scraped_count}/{num_decisions} decisions")
        print(f"Total requests: {self.stats['total_requests']}")
        print(f"Successful: {self.stats['successful_fetches']}")
        print(f"Failed: {self.stats['failed_fetches']}")
        print(f"Data losses/warnings: {len(self.stats['data_losses'])}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*60}")
        
        return summary


def main():
    """Run sample scrape for testing."""
    scraper = RozhodnutiScraper(output_dir="data/rozhodnuti", delay=1.0)
    scraper.scrape_sample(num_decisions=10)


if __name__ == "__main__":
    main()
