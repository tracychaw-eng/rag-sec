from pathlib import Path

from sec_rag.ingestion.chunker import _join_units
from sec_rag.ingestion.parser import html_to_text

TABLE_HTML = """
<html><body>
<p>Revenue was strong this year, as shown in the following table of results.</p>
<table>
  <tr><th>Line item</th><th>2026</th><th>2025</th></tr>
  <tr><td>Revenue</td><td></td><td>130,497</td><td>60,922</td></tr>
  <tr><td>Net income</td><td>72,880</td><td>29,760</td></tr>
  <tr><td></td><td></td></tr>
</table>
<p>Cost of revenue also increased year over year for the company overall.</p>
</body></html>
"""


def test_table_rows_preserved(tmp_path: Path):
    fp = tmp_path / "t.html"
    fp.write_text(TABLE_HTML, encoding="utf-8")
    text = html_to_text(fp)

    assert "| Revenue | 130,497 | 60,922 |" in text
    assert "| Net income | 72,880 | 29,760 |" in text
    assert "| Line item | 2026 | 2025 |" in text
    # prose around the table survives
    assert "Revenue was strong this year" in text
    # empty spacer rows dropped
    assert "|  |" not in text


def test_join_units_keeps_table_rows_on_own_lines():
    joined = _join_units([
        "Revenue grew strongly.",
        "| Revenue | 130,497 |",
        "| Net income | 72,880 |",
        "Costs also rose.",
    ])
    assert "Revenue grew strongly.\n| Revenue | 130,497 |" in joined
    assert "| Revenue | 130,497 |\n| Net income | 72,880 |" in joined
    assert "| Net income | 72,880 |\nCosts also rose." in joined
