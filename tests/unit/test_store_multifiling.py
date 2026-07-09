from sec_rag.ingestion.store import ChunkStore
from sec_rag.models import ChildChunk, ParentChunk


def _chunks(ticker, date, n=2):
    parents = [ParentChunk(id=f"{ticker}_{date}_p{i}", ticker=ticker,
                           filing_date=date, item="1A", text=f"parent {i}")
               for i in range(n)]
    children = [ChildChunk(id=f"{ticker}_{date}_c{i}", parent_id=parents[0].id,
                           ticker=ticker, filing_date=date, item="1A",
                           text=f"child {i}")
                for i in range(n)]
    return parents, children


def test_two_years_coexist(tmp_path):
    store = ChunkStore(tmp_path)
    store.save("NVDA", "2026-02-25", *_chunks("NVDA", "2026-02-25"))
    store.save("NVDA", "2025-02-26", *_chunks("NVDA", "2025-02-26"))

    assert store.available_filings() == [("NVDA", "2025-02-26"),
                                         ("NVDA", "2026-02-25")]
    assert store.available_tickers() == ["NVDA"]
    children = store.load_children()
    assert len(children) == 4
    assert {c.filing_date for c in children} == {"2025-02-26", "2026-02-25"}


def test_reingest_one_filing_preserves_other(tmp_path):
    store = ChunkStore(tmp_path)
    store.save("NVDA", "2026-02-25", *_chunks("NVDA", "2026-02-25", n=2))
    store.save("NVDA", "2025-02-26", *_chunks("NVDA", "2025-02-26", n=3))
    # re-save the newer filing with different content
    store.save("NVDA", "2026-02-25", *_chunks("NVDA", "2026-02-25", n=5))

    by_date = {}
    for c in store.load_children():
        by_date.setdefault(c.filing_date, 0)
        by_date[c.filing_date] += 1
    assert by_date == {"2026-02-25": 5, "2025-02-26": 3}


def test_has_filing(tmp_path):
    store = ChunkStore(tmp_path)
    assert not store.has_filing("MSFT", "2025-07-30")
    store.save("MSFT", "2025-07-30", *_chunks("MSFT", "2025-07-30"))
    assert store.has_filing("MSFT", "2025-07-30")


def test_ticker_filter(tmp_path):
    store = ChunkStore(tmp_path)
    store.save("MSFT", "2025-07-30", *_chunks("MSFT", "2025-07-30"))
    store.save("JPM", "2026-02-13", *_chunks("JPM", "2026-02-13"))
    assert all(c.ticker == "JPM" for c in store.load_children(["JPM"]))
