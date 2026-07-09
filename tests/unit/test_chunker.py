from sec_rag.ingestion.chunker import chunk_section
from sec_rag.models import ItemSection

PROSE = ("The Company depends on third-party suppliers for fabrication. "
         "Supply constraints could adversely affect revenue. " * 120)


def _section():
    return ItemSection(item="1A", title="Risk Factors", text=PROSE)


def test_every_chunk_carries_context_header():
    parents, children = chunk_section(_section(), "NVDA", "2026-02-25")
    assert parents and children
    for c in parents + children:
        assert c.text.startswith("[NVDA 10-K filed 2026-02-25, Item 1A — Risk Factors]")


def test_children_reference_existing_parents():
    parents, children = chunk_section(_section(), "NVDA", "2026-02-25")
    parent_ids = {p.id for p in parents}
    assert all(c.parent_id in parent_ids for c in children)


def test_token_budgets_respected():
    parents, children = chunk_section(
        _section(), "NVDA", "2026-02-25",
        parent_tokens=500, child_tokens=100)
    # generous tolerance: one oversized unit is allowed to overflow
    assert all(p.token_count <= 600 for p in parents)
    assert all(c.token_count <= 150 for c in children)


def test_children_cover_parent_content():
    parents, children = chunk_section(_section(), "NVDA", "2026-02-25")
    # every parent has at least one child
    covered = {c.parent_id for c in children}
    assert covered == {p.id for p in parents}
