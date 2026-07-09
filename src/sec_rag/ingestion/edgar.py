"""SEC EDGAR client — ticker → CIK → 10-K filings → HTML on disk.

Port of the legacy edgar_downloader.py into the package, extended to
fetch the latest N annual filings per ticker (multi-year corpus).

SEC requirements: User-Agent with contact info, max 10 req/s.
"""

import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger("sec_rag.edgar")

USER_AGENT = "sec-rag-project tracy.chaw@gmail.com"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
RATE_LIMIT_DELAY = 0.12

_last_request = 0.0
_cik_cache: dict[str, str] = {}


def _get(url: str) -> requests.Response:
    global _last_request
    elapsed = time.time() - _last_request
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)
    for attempt in range(3):
        resp = requests.get(url, headers=HEADERS, timeout=30)
        _last_request = time.time()
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Failed after retries: {url}")


def get_cik(ticker: str) -> str:
    """CIK for a ticker, zero-padded to 10 digits."""
    global _cik_cache
    if not _cik_cache:
        data = _get("https://www.sec.gov/files/company_tickers.json").json()
        _cik_cache = {e["ticker"].upper(): str(e["cik_str"])
                      for e in data.values()}
    ticker = ticker.upper()
    if ticker not in _cik_cache:
        raise ValueError(f"Ticker {ticker!r} not found in SEC database")
    return _cik_cache[ticker].zfill(10)


def _scan_batch(batch: dict, ticker: str, cik: str, out: list[dict],
                count: int) -> None:
    for i, form in enumerate(batch.get("form", [])):
        if form == "10-K":
            out.append({
                "ticker": ticker.upper(),
                "cik": cik,
                "date": batch["filingDate"][i],
                "accession": batch["accessionNumber"][i],
                "primary_doc": batch["primaryDocument"][i],
            })
            if len(out) >= count:
                return


def list_10k_filings(ticker: str, count: int = 1) -> list[dict]:
    """Metadata for the latest `count` 10-K filings, newest first.

    High-volume filers (e.g. JPM files prospectuses daily) push older
    10-Ks out of the "recent" window, so this pages through the older
    submission batches until enough 10-Ks are found.
    """
    cik = get_cik(ticker)
    data = _get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()

    out: list[dict] = []
    _scan_batch(data.get("filings", {}).get("recent", {}),
                ticker, cik, out, count)

    for older in data.get("filings", {}).get("files", []):
        if len(out) >= count:
            break
        batch = _get(f"https://data.sec.gov/submissions/{older['name']}").json()
        _scan_batch(batch, ticker, cik, out, count)

    return out[:count]


def download_filing(meta: dict, dest_dir: Path) -> Path:
    """Download one filing's primary document; skips if already on disk."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{meta['ticker']}_10K_{meta['date']}.html"
    if path.exists():
        logger.info("filing already on disk", extra={"file": path.name})
        return path

    accession = meta["accession"].replace("-", "")
    url = (f"https://www.sec.gov/Archives/edgar/data/"
           f"{meta['cik'].lstrip('0')}/{accession}/{meta['primary_doc']}")
    logger.info("downloading filing", extra={"url": url})
    resp = _get(url)
    path.write_text(resp.text, encoding="utf-8")
    return path


def download_10ks(ticker: str, dest_dir: Path, count: int = 1) -> list[Path]:
    """Latest `count` 10-Ks for a ticker, newest first."""
    return [download_filing(m, dest_dir)
            for m in list_10k_filings(ticker, count)]
