"""Section-aware parent/child chunking.

Parents (~1500 tokens) are what the LLM reads — large enough that a fact
like "$28.9 billion plus penalties and interest" survives with its
surrounding sentence intact (the legacy pipeline's 512-token chunks cut
exactly this fact in half; F004 recall = 0.0).

Children (~300 tokens) are what gets embedded and BM25-indexed — small
enough that a single vector represents one idea.

Every chunk carries a context header naming the company, filing date, and
Item, because 10-K prose says "we/our" and never names the company —
the root cause of the legacy faithfulness failures.
"""

import re

import tiktoken

from ..models import ChildChunk, ItemSection, ParentChunk

_ENC = tiktoken.get_encoding("cl100k_base")


def _ntokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'(])")


def _split_sentences(text: str) -> list[str]:
    parts = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        parts.extend(s for s in _SENTENCE_RE.split(para) if s.strip())
    return parts


def _join_units(units: list[str]) -> str:
    """Prose sentences join with spaces; table rows (pipe lines) keep
    their own lines so numeric rows stay readable."""
    out = []
    for u in units:
        if out:
            out.append("\n" if (u.startswith("|") or
                                out[-1].startswith("|")) else " ")
        out.append(u)
    return "".join(out)


def _pack(units: list[str], budget: int) -> list[str]:
    """Greedy packing of text units into chunks of <= budget tokens."""
    chunks, cur, cur_tok = [], [], 0
    for u in units:
        t = _ntokens(u)
        if cur and cur_tok + t > budget:
            chunks.append(_join_units(cur))
            cur, cur_tok = [], 0
        # A single unit larger than budget becomes its own chunk (rare:
        # tables flattened to one line)
        cur.append(u)
        cur_tok += t
    if cur:
        chunks.append(_join_units(cur))
    return chunks


def _header(ticker: str, filing_date: str, item: str, title: str) -> str:
    label = f"Item {item}" if item != "FULL" else "full filing"
    title_part = f" — {title}" if title else ""
    return f"[{ticker} 10-K filed {filing_date}, {label}{title_part}] "


def chunk_section(
    section: ItemSection,
    ticker: str,
    filing_date: str,
    parent_tokens: int = 1500,
    child_tokens: int = 300,
    child_overlap_tokens: int = 60,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    header = _header(ticker, filing_date, section.item, section.title)
    sentences = _split_sentences(section.text)
    if not sentences:
        return [], []

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    for pi, parent_text in enumerate(_pack(sentences, parent_tokens)):
        pid = f"{ticker}_{filing_date}_it{section.item}_p{pi:03d}"
        parents.append(ParentChunk(
            id=pid, ticker=ticker, filing_date=filing_date,
            item=section.item, item_title=section.title,
            text=header + parent_text,
            token_count=_ntokens(parent_text),
        ))

        # Children: sentence-packed windows with sentence-level overlap
        p_sentences = _split_sentences(parent_text)
        ci = 0
        start = 0
        while start < len(p_sentences):
            cur, cur_tok, end = [], 0, start
            while end < len(p_sentences) and (
                    not cur or cur_tok + _ntokens(p_sentences[end]) <= child_tokens):
                cur_tok += _ntokens(p_sentences[end])
                cur.append(p_sentences[end])
                end += 1
            child_text = _join_units(cur)
            children.append(ChildChunk(
                id=f"{pid}_c{ci:03d}", parent_id=pid,
                ticker=ticker, filing_date=filing_date,
                item=section.item, item_title=section.title,
                text=header + child_text,
                token_count=_ntokens(child_text),
            ))
            ci += 1
            if end >= len(p_sentences):
                break
            # step back for overlap
            overlap_tok, back = 0, end
            while back > start + 1 and overlap_tok < child_overlap_tokens:
                back -= 1
                overlap_tok += _ntokens(p_sentences[back])
            start = max(back, start + 1)

    return parents, children
