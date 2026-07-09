"""10-K HTML → Item-level sections.

EDGAR primary documents are messy iXBRL HTML. Strategy:
  1. Strip to plain text with BeautifulSoup (like the legacy pipeline).
  2. Find lines that look like Item headings ("Item 1A. Risk Factors").
  3. Drop table-of-contents artifacts: a real section owns a long span of
     text before the next heading; TOC entries cluster with tiny gaps.
  4. Fall back to a single "FULL" section if detection fails, so ingestion
     never hard-fails on an unusual filing.
"""

import re
from pathlib import Path

from bs4 import BeautifulSoup

from ..models import ItemSection

# Canonical 10-K item titles (used when the heading line doesn't carry one)
ITEM_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis (MD&A)",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

# Heading at the start of a line: "Item 1A." / "ITEM 1A — Risk Factors" etc.
ITEM_HEADING_RE = re.compile(
    r"(?im)^\s*item\s+(\d{1,2}[abc]?)\s*[.:–—-]?\s*(.{0,100})$"
)


def _table_to_rows(table) -> str:
    """Render an HTML table as pipe-delimited rows.

    Raw get_text() scatters each cell onto its own line, destroying the
    row structure that makes financial-statement numbers answerable
    ("Revenue | 130,497 | 60,922"). iXBRL tables are full of empty
    spacer cells — those are dropped per row.
    """
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True)
                 for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if len(cells) >= 2:
            rows.append("| " + " | ".join(cells) + " |")
        elif len(cells) == 1:
            rows.append(cells[0])       # section header row inside a table
    return "\n".join(rows)


def html_to_text(filepath: Path) -> str:
    """Extract clean text from a 10-K HTML filing, preserving table rows."""
    raw = filepath.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "meta", "link", "header", "footer"]):
        tag.decompose()

    # Replace each table with its pipe-row rendering BEFORE get_text, so
    # numeric rows stay intact. Nested tables: innermost first.
    for table in reversed(soup.find_all("table")):
        rendered = _table_to_rows(table)
        table.replace_with("\n" + rendered + "\n" if rendered else "\n")

    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if len(ln) > 3)


def split_items(text: str, min_section_chars: int = 1500) -> list[ItemSection]:
    """Split filing text into Item sections, filtering TOC noise.

    Some filings (e.g. MSFT) repeat "Item N" as a running header on every
    page, fragmenting each real section into many consecutive same-item
    segments. Consecutive same-item candidates are therefore merged before
    any filtering. TOC entries never merge — they alternate item ids —
    and are then removed by the length filter.
    """
    matches = list(ITEM_HEADING_RE.finditer(text))
    if not matches:
        return [ItemSection(item="FULL", title="Full filing", text=text)]

    # Candidate segments: heading -> next heading (or EOF)
    candidates = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        item = m.group(1).upper()
        candidates.append((item, text[start:end]))

    # Merge consecutive runs of the same item id (running-header repeats)
    merged: list[tuple[str, str]] = []
    for item, body in candidates:
        if merged and merged[-1][0] == item:
            merged[-1] = (item, merged[-1][1] + body)
        else:
            merged.append((item, body))

    # Real sections own substantial text; TOC entries do not.
    sections: dict[str, ItemSection] = {}
    order: dict[str, int] = {}
    for pos, (item, body) in enumerate(merged):
        if len(body) < min_section_chars:
            continue
        # Keep the LONGEST occurrence per item id (TOC/cross-reference
        # copies are short).
        if item not in sections or len(body) > len(sections[item].text):
            sections[item] = ItemSection(
                item=item, title=ITEM_TITLES.get(item, ""), text=body)
            order[item] = pos

    if len(sections) < 3:
        # Detection failed (unusual layout) — never lose content
        return [ItemSection(item="FULL", title="Full filing", text=text)]

    # Preserve document order. Content before the first real section
    # (cover page) is intentionally dropped; everything else is retained.
    return sorted(sections.values(), key=lambda s: order.get(s.item, 999))


def parse_filing(filepath: Path, min_section_chars: int = 1500) -> list[ItemSection]:
    return split_items(html_to_text(filepath), min_section_chars)
