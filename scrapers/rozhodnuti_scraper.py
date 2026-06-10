#!/usr/bin/env python3
"""
Rozhodnuti.justice.cz Scraper

Scrapes decisions from the Ministry of Justice database using their REST API.
Implements rate limiting to avoid overwhelming the server.
"""

import requests
import json
import time
import random
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime


class RozhodnutiScraper:
    """Scraper for rozhodnuti.justice.cz REST API."""
    
    BASE_URL = "https://rozhodnuti.justice.cz/api"
    OPENDATA_ENDPOINT = "/opendata"
    FINALDOC_ENDPOINT = "/finaldoc"
    
    def __init__(self, output_dir: str = "data/rozhodnuti", delay: float = 0.3, 
                 phase_duration_hours: float = 2.0, break_duration_hours: float = 0.5):
        """
        Initialize scraper.
        
        Args:
            output_dir: Directory to save scraped data
            delay: Base delay between requests in seconds (rate limiting)
            phase_duration_hours: How long to scrape before taking a break
            break_duration_hours: How long to pause between phases
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_dir = self.output_dir / "decisions"
        self.decisions_dir.mkdir(exist_ok=True)
        self.delay = delay
        self.phase_duration = phase_duration_hours * 3600  # Convert to seconds
        self.break_duration = break_duration_hours * 3600  # Convert to seconds
        self.phase_start_time = time.time()
        
        # Track statistics
        self.stats = {
            "total_requests": 0,
            "successful_fetches": 0,
            "failed_fetches": 0,
            "data_losses": []
        }
        
        # Progress tracking
        self.progress_file = self.output_dir / "progress.json"
        self.progress = self._load_progress()
    
    def _load_progress(self) -> Dict:
        """Load progress from file if exists."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert list back to set
                data["completed"] = set(data["completed"])
                data["failed"] = set(data.get("failed", []))
                return data
        return {"completed": set(), "failed": set(), "current_year": None, "current_month": None, "current_day": None}
    
    def _save_progress(self):
        """Save progress to file."""
        # Convert sets to lists for JSON serialization
        progress_to_save = {
            "completed": list(self.progress["completed"]),
            "failed": list(self.progress.get("failed", set())),
            "current_year": self.progress["current_year"],
            "current_month": self.progress["current_month"],
            "current_day": self.progress["current_day"]
        }
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_to_save, f, indent=2, ensure_ascii=False)
    
    def _mark_complete(self, uuid: str):
        """Mark a decision as completed in progress."""
        self.progress["completed"].add(uuid)
        self._save_progress()
    
    def _is_complete(self, uuid: str) -> bool:
        """Check if a decision has been scraped."""
        return uuid in self.progress["completed"]
    
    def _request_with_delay(self, url: str) -> Optional[requests.Response]:
        """
        Make HTTP request with rate limiting delay.
        
        Args:
            url: URL to request
            
        Returns:
            Response object or None if failed
        """
        # Check if we need to take a break (phase-based scraping)
        if self.stats["total_requests"] > 0:
            elapsed = time.time() - self.phase_start_time
            if elapsed >= self.phase_duration:
                print(f"\n{'='*60}")
                print(f"Phase complete. Taking {self.break_duration/3600:.1f} hour break...")
                print(f"{'='*60}")
                time.sleep(self.break_duration)
                self.phase_start_time = time.time()
                print(f"Resuming scraping...")
            
            # Add randomized delay to look more natural
            randomized_delay = self.delay * random.uniform(0.8, 1.2)
            time.sleep(randomized_delay)
        
        # Retry with exponential backoff on transient errors
        max_retries = 4
        for attempt in range(max_retries):
            try:
                self.stats["total_requests"] += 1
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                # Validate JSON response is complete (catches IncompleteRead silently swallowed)
                _ = response.content
                self.stats["successful_fetches"] += 1
                return response
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    backoff = (2 ** attempt) * 5 + random.uniform(0, 3)  # 5s, 10s, 20s, 40s
                    print(f"  Retry {attempt+1}/{max_retries-1} for {url} after {backoff:.1f}s: {e}")
                    time.sleep(backoff)
                else:
                    print(f"Error fetching {url} (gave up after {max_retries} attempts): {e}")
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
    
    def save_decision(self, uuid: str, data: Dict, year: int, month: int) -> str:
        """
        Save decision data to file with hierarchical structure.
        
        Args:
            uuid: Decision UUID
            data: Decision data dictionary
            year: Year of publication
            month: Month of publication
            
        Returns:
            Path to saved file
        """
        # Hierarchical structure: court_code/year/uuid.json
        court_code = data.get('metadata', {}).get('courtCode', 'UNKNOWN')
        court_dir = self.decisions_dir / court_code
        court_dir.mkdir(exist_ok=True)
        year_dir = court_dir / str(year)
        year_dir.mkdir(exist_ok=True)
        
        filename = f"{uuid}.json"
        filepath = year_dir / filename
        
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
                    
                    # Save decision with hierarchical structure
                    filepath = self.save_decision(uuid, full_data, year, month)
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
    
    def scrape_all(self, start_year: int = 2020, end_year: int = 2026):
        """
        Scrape all decisions from the database.
        
        Args:
            start_year: First year to scrape
            end_year: Last year to scrape
        """
        print(f"Starting full scrape from {start_year} to {end_year}...")
        print(f"Rate limiting: {self.delay}s delay between requests")
        print(f"Progress tracking enabled - can resume if interrupted")
        
        total_scraped = 0
        total_decisions = 0
        
        # Get all years
        years_data = self.get_years()
        if not years_data:
            print("Failed to get years")
            return
        
        # Filter years based on start/end
        years_to_scrape = [y for y in years_data if start_year <= y['rok'] <= end_year]
        
        for year_info in years_to_scrape:
            year = year_info['rok']
            year_count = year_info['pocet']
            total_decisions += year_count
            
            print(f"\n{'='*60}")
            print(f"Year {year}: {year_count} decisions")
            print(f"{'='*60}")
            
            # Get months for this year
            months = self.get_months(year)
            if not months:
                print(f"  Failed to get months for {year}")
                continue
            
            for month_info in months:
                month = month_info['mesic']
                month_count = month_info['pocet']
                
                print(f"\n  {year}-{month:02d}: {month_count} decisions")
                
                # Update progress
                self.progress["current_year"] = year
                self.progress["current_month"] = month
                self._save_progress()
                
                # Get days for this month
                days = self.get_days(year, month)
                if not days:
                    print(f"    Failed to get days for {year}-{month:02d}")
                    continue
                
                for day_info in days:
                    day = int(day_info['datum'].split('-')[2])
                    day_count = day_info['pocet']
                    
                    print(f"    {year}-{month:02d}-{day:02d}: {day_count} decisions")
                    
                    # Update progress
                    self.progress["current_day"] = day
                    self._save_progress()
                    
                    # Handle pagination
                    page = 0
                    while True:
                        decision_list = self.get_decision_list(year, month, day, page)
                        if not decision_list:
                            break
                        
                        items = decision_list.get('items', [])
                        if not items:
                            break
                        
                        # Scrape decisions from this page
                        for item in items:
                            uuid = item['odkaz'].split('/')[-1]
                            ecli = item.get('ecli', 'unknown')
                            
                            # Skip if already scraped
                            if self._is_complete(uuid):
                                continue
                            
                            # Get full decision
                            full_data = self.get_decision_full(uuid)
                            if not full_data:
                                print(f"      FAILED: {ecli} ({uuid})")
                                self.progress["failed"].add(uuid)
                                self.stats["data_losses"].append({
                                    "uuid": uuid,
                                    "ecli": ecli,
                                    "error": "Failed to fetch full decision"
                                })
                                continue
                            
                            # Recovered from a previous failure
                            if uuid in self.progress["failed"]:
                                self.progress["failed"].discard(uuid)
                            
                            # Check data completeness
                            warnings = self.check_data_completeness(full_data)
                            if warnings:
                                self.stats["data_losses"].append({
                                    "uuid": uuid,
                                    "ecli": ecli,
                                    "warnings": warnings
                                })
                            
                            # Save decision with hierarchical structure
                            filepath = self.save_decision(uuid, full_data, year, month)
                            self._mark_complete(uuid)
                            total_scraped += 1
                            
                            if total_scraped % 100 == 0:
                                print(f"      Progress: {total_scraped}/{total_decisions} decisions scraped")
                        
                        # Check if there are more pages
                        if len(items) < 100:
                            break
                        
                        page += 1
        
        # Save final summary
        summary = {
            "scrape_date": datetime.now().isoformat(),
            "years_scraped": f"{start_year}-{end_year}",
            "total_decisions_available": total_decisions,
            "total_decisions_scraped": total_scraped,
            "statistics": self.stats
        }
        
        summary_path = self.output_dir / "full_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"Full scrape complete!")
        print(f"Scraped: {total_scraped}/{total_decisions} decisions")
        print(f"Total requests: {self.stats['total_requests']}")
        print(f"Successful: {self.stats['successful_fetches']}")
        print(f"Failed: {self.stats['failed_fetches']}")
        print(f"Data losses/warnings: {len(self.stats['data_losses'])}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*60}")


def main():
    """Run full scrape with phased approach."""
    # 0.3s delay = ~3.3 requests/sec = ~1.16M requests / 345,600s = 4 days
    # Phase: 2 hours scraping, 0.5 hours break (reduces continuous load)
    scraper = RozhodnutiScraper(
        output_dir="data/rozhodnuti", 
        delay=0.3,
        phase_duration_hours=2.0,
        break_duration_hours=0.5
    )
    scraper.scrape_all(start_year=2020, end_year=2026)


if __name__ == "__main__":
    main()
