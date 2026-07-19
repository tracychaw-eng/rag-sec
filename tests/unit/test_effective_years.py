from sec_rag.retrieval.retriever import HybridRetriever


class _StubStore:
    """JPM files in Feb (FY2024 report filed 2025); MSFT files in July
    (FY2025 report filed 2025)."""

    def available_filings(self):
        return [("JPM", "2025-02-14"), ("JPM", "2026-02-13"),
                ("MSFT", "2024-07-30"), ("MSFT", "2025-07-30")]


def _retriever():
    r = HybridRetriever.__new__(HybridRetriever)
    r.store = _StubStore()
    return r


def test_no_years_passthrough():
    assert _retriever()._effective_years(None, ["JPM"]) is None
    assert _retriever()._effective_years([], None) is None


def test_exact_filing_year_kept():
    # "the 2025 10-K" and the corpus has a 2025 filing -> exact match only
    assert _retriever()._effective_years([2025], ["JPM"]) == [2025]


def test_fiscal_year_shifts_to_next_filing_year():
    # "as of December 31, 2024" -> JPM's FY2024 report is filed 2025
    assert _retriever()._effective_years([2024], ["JPM"]) == [2025]


def test_uncovered_year_drops_filter():
    # events from 2022 aren't a filing year here and 2023 isn't either ->
    # drop the filter instead of retrieving nothing
    assert _retriever()._effective_years([2022], ["JPM"]) is None


def test_mapping_is_per_ticker():
    # 2024 is a real MSFT filing year but not a JPM one
    assert _retriever()._effective_years([2024], ["MSFT"]) == [2024]
    assert _retriever()._effective_years([2024], ["JPM"]) == [2025]


def test_multi_year_collapses_to_covering_filings():
    # FY2023+FY2024 for JPM: FY2023 has no filing (2023 or 2024), FY2024
    # maps to the 2025-filed report (which carries comparatives)
    assert _retriever()._effective_years([2023, 2024], ["JPM"]) == [2025]
