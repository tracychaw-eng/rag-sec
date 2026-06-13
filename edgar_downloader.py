"""
edgar_downloader.py — Fetch latest 10-K filings from SEC EDGAR API.
Part of the llamaindex-sec project.

Usage:
    python edgar_downloader.py                  # downloads MSFT, NVDA, JPM
    python edgar_downloader.py AAPL GOOG TSLA   # custom tickers

SEC EDGAR requires:
    - User-Agent header with name + email
    - Max 10 requests/second
"""

import os
import sys
import json
import time
import re
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# UPDATE these with your real name/email — SEC blocks requests without it
USER_AGENT = "LlamaIndexSEC-Project tracychaw@gmail.com"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

DATA_DIR = Path("data/filings")
DEFAULT_TICKERS = ["MSFT", "NVDA", "JPM"]

# Rate limiting: SEC allows max 10 req/sec → sleep 0.12s between requests
RATE_LIMIT_DELAY = 0.12
_last_request_time = 0.0


def _rate_limited_get(url: str) -> requests.Response:
    """GET with rate limiting and retry on 429."""
    global _last_request_time

    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)

    for attempt in range(3):
        resp = requests.get(url, headers=HEADERS, timeout=30)
        _last_request_time = time.time()

        if resp.status_code == 200:
            return resp
        elif resp.status_code == 429:
            wait = 2 ** attempt
            print(f"  ⚠ Rate limited (429). Waiting {wait}s before retry...")
            time.sleep(wait)
        else:
            resp.raise_for_status()

    raise RuntimeError(f"Failed after 3 retries: {url}")


# ---------------------------------------------------------------------------
# STEP 1: Ticker → CIK lookup
# ---------------------------------------------------------------------------
_cik_cache: dict = {}


def load_cik_map() -> dict:
    """Download SEC's ticker→CIK mapping (cached in memory)."""
    global _cik_cache
    if _cik_cache:
        return _cik_cache

    print("📥 Loading SEC company tickers map...")
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = _rate_limited_get(url)
    data = resp.json()

    # data is {0: {cik_str, ticker, title}, 1: {...}, ...}
    for entry in data.values():
        ticker = entry["ticker"].upper()
        cik = str(entry["cik_str"])
        _cik_cache[ticker] = cik

    print(f"  ✓ Loaded {len(_cik_cache)} tickers")
    return _cik_cache


def get_cik(ticker: str) -> str:
    """Return CIK for a ticker, zero-padded to 10 digits."""
    cik_map = load_cik_map()
    ticker = ticker.upper()
    if ticker not in cik_map:
        raise ValueError(f"Ticker '{ticker}' not found in SEC database")
    return cik_map[ticker].zfill(10)


# ---------------------------------------------------------------------------
# STEP 2: Find latest 10-K filing metadata
# ---------------------------------------------------------------------------
def get_latest_10k_meta(cik: str) -> dict:
    """Query submissions endpoint, return metadata for the most recent 10-K."""
    print(f"  📋 Fetching filing history for CIK {cik}...")
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = _rate_limited_get(url)
    company_data = resp.json()

    company_name = company_data.get("name", "Unknown")
    recent = company_data.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form == "10-K":
            return {
                "company": company_name,
                "form": form,
                "date": dates[i],
                "accession": accessions[i],
                "primary_doc": primary_docs[i],
            }

    raise ValueError(f"No 10-K filing found for CIK {cik} ({company_name})")


# ---------------------------------------------------------------------------
# STEP 3: Download the actual filing document
# ---------------------------------------------------------------------------
def download_filing(cik: str, meta: dict) -> str:
    """Download the primary document of a 10-K filing. Returns the text/HTML."""
    accession_no_dashes = meta["accession"].replace("-", "")
    doc_name = meta["primary_doc"]

    url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_no_dashes}/{doc_name}"
    print(f"  ⬇ Downloading: {url}")
    resp = _rate_limited_get(url)
    return resp.text


# ---------------------------------------------------------------------------
# STEP 4: Save to disk
# ---------------------------------------------------------------------------
def save_filing(ticker: str, meta: dict, content: str) -> Path:
    """Save filing content to data/filings/{TICKER}_10k_{date}.html"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ext = ".html" if "<html" in content[:500].lower() else ".txt"
    filename = f"{ticker.upper()}_10K_{meta['date']}{ext}"
    filepath = DATA_DIR / filename

    filepath.write_text(content, encoding="utf-8")
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  💾 Saved: {filepath} ({size_mb:.1f} MB)")
    return filepath


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def download_10k(ticker: str) -> Path:
    """Full pipeline: ticker → CIK → find 10-K → download → save."""
    print(f"\n{'='*60}")
    print(f"📄 Processing: {ticker}")
    print(f"{'='*60}")

    cik = get_cik(ticker)
    print(f"  🔑 CIK: {cik}")

    meta = get_latest_10k_meta(cik)
    print(f"  📌 Found: {meta['company']}")
    print(f"     Form: {meta['form']} | Filed: {meta['date']}")

    content = download_filing(cik, meta)
    filepath = save_filing(ticker, meta, content)

    return filepath


def main():
    tickers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TICKERS

    print("=" * 60)
    print("SEC EDGAR 10-K Downloader")
    print(f"Tickers: {', '.join(tickers)}")
    print("=" * 60)

    results = {}
    for ticker in tickers:
        try:
            path = download_10k(ticker)
            results[ticker] = {"status": "success", "path": str(path)}
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[ticker] = {"status": "error", "error": str(e)}

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for ticker, result in results.items():
        if result["status"] == "success":
            print(f"  ✅ {ticker}: {result['path']}")
        else:
            print(f"  ❌ {ticker}: {result['error']}")


if __name__ == "__main__":
    main()