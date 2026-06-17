#!/usr/bin/env python3
"""Selenium (Firefox) fetch layer for NALUS — the proven transport.

The search is a stateful ASP.NET WebForms app; pure-HTTP postbacks re-render the
form without executing (server-side validators), so we drive a real browser like
the reference ccc_dataset. We search by **ECLI** (one decision -> one hit ->
its ``ResultDetail.aspx?id=``), which is far cheaper than the reference's full
date-range crawl since we already know our 309 ECLIs.

Body text is plain HTTP (GetText), exactly as the reference (rvest) does.

Parsing lives in ``parse_card`` (transport-agnostic); this module only fetches.
"""

from __future__ import annotations

import random
import re
import time
from typing import List, Optional

import requests

SEARCH_URL = "https://nalus.usoud.cz/Search/Search.aspx"
DETAIL_URL = "https://nalus.usoud.cz/Search/ResultDetail.aspx?id={}"
GETTEXT_URL = "https://nalus.usoud.cz/Search/GetText.aspx?sz={}"
GECKODRIVER = "/snap/bin/geckodriver"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

_RE_DETAIL_ID = re.compile(r"ResultDetail\.aspx\?id=(\d+)")
_RE_TOTAL = re.compile(r"z celkem\s+(\d+)")


class NalusScraper:
    """Browser-driven record-card scraper + HTTP body fetcher."""

    def __init__(self, headless: bool = True, geckodriver: str = GECKODRIVER,
                 delay: float = 1.0, page_wait: float = 2.5):
        self.geckodriver = geckodriver
        self.headless = headless
        self.delay = delay
        self.page_wait = page_wait
        self._drv = None
        self._http = requests.Session()
        self._http.headers.update({"User-Agent": UA})

    # ── lifecycle ──────────────────────────────────────────────
    def start(self) -> None:
        from selenium import webdriver
        from selenium.webdriver.firefox.options import Options
        from selenium.webdriver.firefox.service import Service

        opts = Options()
        if self.headless:
            opts.add_argument("-headless")
        self._drv = webdriver.Firefox(service=Service(self.geckodriver), options=opts)
        self._drv.set_page_load_timeout(90)

    def quit(self) -> None:
        if self._drv is not None:
            self._drv.quit()
            self._drv = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.quit()

    # ── search by ECLI -> ResultDetail id ──────────────────────
    def search_ecli(self, ecli: str) -> tuple[Optional[str], int]:
        """Return (detail_id, total_hits). detail_id is None unless exactly 1 hit."""
        from selenium.webdriver.common.by import By

        self._drv.get(SEARCH_URL)
        box = self._drv.find_element(By.ID, "ctl00_MainContent_ecli")
        box.clear()
        box.send_keys(ecli)
        self._drv.find_element(By.ID, "ctl00_MainContent_but_search").click()
        time.sleep(self.page_wait)
        html = self._drv.page_source

        tot_m = _RE_TOTAL.search(re.sub(r"<[^>]+>", " ", html))
        total = int(tot_m.group(1)) if tot_m else len(set(_RE_DETAIL_ID.findall(html)))
        ids = sorted(set(_RE_DETAIL_ID.findall(html)))
        detail_id = ids[0] if total == 1 and ids else None
        return detail_id, total

    def card_html(self, detail_id: str) -> str:
        """Fetch a ResultDetail record-card page (HTML)."""
        time.sleep(self.delay * random.uniform(0.8, 1.2))
        self._drv.get(DETAIL_URL.format(detail_id))
        time.sleep(self.page_wait)
        return self._drv.page_source

    # ── body text (plain HTTP) ─────────────────────────────────
    def fetch_body(self, file_id: str, max_retries: int = 3) -> Optional[str]:
        """Raw GetText HTML for a decision (same format as the legacy scrape)."""
        url = GETTEXT_URL.format(requests.utils.quote(file_id, safe=""))
        for attempt in range(max_retries):
            try:
                time.sleep(self.delay * random.uniform(0.8, 1.2))
                r = self._http.get(url, timeout=30)
                r.raise_for_status()
                if len(r.text) < 5000:
                    return None
                return r.text
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"    body fetch failed for {file_id}: {e}")
                    return None
                time.sleep((2 ** attempt) * 2 + random.uniform(0, 1))
        return None
