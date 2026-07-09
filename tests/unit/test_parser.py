from sec_rag.ingestion.parser import split_items

FILLER = "This is disclosure prose that goes on at some length. " * 40  # ~2.2k chars


def _toc() -> str:
    return "\n".join([
        "Item 1. Business",
        "Item 1A. Risk Factors",
        "Item 7. Management's Discussion",
    ])


def test_toc_entries_are_dropped_real_sections_kept():
    text = "\n".join([
        _toc(),
        "Item 1. Business", FILLER,
        "Item 1A. Risk Factors", FILLER + FILLER,
        "Item 7. Management's Discussion", FILLER,
    ])
    sections = split_items(text, min_section_chars=1500)
    items = [s.item for s in sections]
    assert items == ["1", "1A", "7"]
    ra = next(s for s in sections if s.item == "1A")
    assert len(ra.text) > 4000  # got the real section, not the TOC line


def test_running_header_repeats_are_merged():
    # MSFT-style: "Item 1A" repeated as page headers fragments the section
    text = "\n".join([
        _toc(),
        "Item 1. Business", FILLER,
        "Item 1A. Risk Factors", FILLER,
        "Item 1A", FILLER,          # running header, page 2
        "Item 1A", FILLER,          # running header, page 3
        "Item 7. MD&A", FILLER,
    ])
    sections = split_items(text, min_section_chars=1500)
    ra = next(s for s in sections if s.item == "1A")
    # All three fragments merged into one section
    assert len(ra.text) > 3 * len(FILLER)


def test_fallback_when_no_headings():
    sections = split_items("no items here " * 200)
    assert len(sections) == 1
    assert sections[0].item == "FULL"


def test_canonical_titles():
    text = "\n".join([
        "Item 1. Business", FILLER,
        "Item 1A. junk-title-from-header", FILLER,
        "Item 7A. whatever", FILLER,
    ])
    sections = split_items(text, min_section_chars=1500)
    ra = next(s for s in sections if s.item == "1A")
    assert ra.title == "Risk Factors"
