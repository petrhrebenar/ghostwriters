#!/usr/bin/env python3
"""
Fetch decision text from NALUS GetText.aspx endpoint.
"""

import requests
import time
import random
from typing import Optional
from urllib.parse import quote


class NALUSFetcher:
    """Fetch decision text from NALUS GetText.aspx endpoint."""
    
    BASE_URL = "https://nalus.usoud.cz/Search/GetText.aspx"
    
    def __init__(self, delay: float = 0.5):
        """
        Initialize fetcher.
        
        Args:
            delay: Base delay between requests in seconds (rate limiting)
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_fetches": 0,
            "failed_fetches": 0
        }
    
    def _request_with_delay(self, url: str) -> Optional[requests.Response]:
        """
        Make HTTP request with rate limiting delay.
        
        Args:
            url: URL to request
            
        Returns:
            Response object or None if failed
        """
        # Add randomized delay
        randomized_delay = self.delay * random.uniform(0.8, 1.2)
        time.sleep(randomized_delay)
        
        # Retry with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.stats["total_requests"] += 1
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Check if we got actual content (not empty template)
                if len(response.text) < 5000:
                    return None
                
                self.stats["successful_fetches"] += 1
                return response
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    backoff = (2 ** attempt) * 2 + random.uniform(0, 1)
                    print(f"  Retry {attempt+1}/{max_retries-1} after {backoff:.1f}s: {e}")
                    time.sleep(backoff)
                else:
                    print(f"Error fetching {url} (gave up after {max_retries} attempts): {e}")
                    self.stats["failed_fetches"] += 1
                    return None
    
    def fetch_decision(self, spis_zn: str) -> Optional[str]:
        """
        Fetch decision text by spisová značka (file number).
        
        Args:
            spis_zn: File number in format "1-1056-07_2"
        
        Returns:
            HTML content as string, or None if failed
        """
        # URL-encode the spis_zn
        encoded_spis = quote(spis_zn, safe='')
        url = f"{self.BASE_URL}?sz={encoded_spis}"
        
        response = self._request_with_delay(url)
        if response:
            return response.text
        return None
