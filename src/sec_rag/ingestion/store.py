"""Chunk store — parents and children as JSONL per (ticker, filing_date).

Parents are the generation payload and live only here (not in the vector
DB). Children are duplicated: text+metadata here (BM25 + inspection),
vectors in Qdrant with the chunk id in the payload.

Keying by filing keeps multiple fiscal years per ticker side by side;
re-ingesting one filing never clobbers another year's chunks.
"""

from pathlib import Path

from ..models import ChildChunk, ParentChunk


class ChunkStore:
    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str, filing_date: str, kind: str) -> Path:
        return self.store_dir / f"{ticker.upper()}_{filing_date}.{kind}.jsonl"

    def save(self, ticker: str, filing_date: str,
             parents: list[ParentChunk], children: list[ChildChunk]) -> None:
        for kind, chunks in (("parents", parents), ("children", children)):
            with open(self._path(ticker, filing_date, kind), "w",
                      encoding="utf-8") as f:
                for c in chunks:
                    f.write(c.model_dump_json() + "\n")

    def has_filing(self, ticker: str, filing_date: str) -> bool:
        return self._path(ticker, filing_date, "children").exists()

    def available_filings(self) -> list[tuple[str, str]]:
        """[(ticker, filing_date), ...] sorted."""
        out = []
        for fp in sorted(self.store_dir.glob("*.children.jsonl")):
            stem = fp.name.removesuffix(".children.jsonl")
            ticker, _, date = stem.partition("_")
            out.append((ticker, date))
        return out

    def available_tickers(self) -> list[str]:
        return sorted({t for t, _ in self.available_filings()})

    def load_children(self, tickers: list[str] | None = None) -> list[ChildChunk]:
        return self._load("children", ChildChunk, tickers)

    def load_parents(self, tickers: list[str] | None = None) -> list[ParentChunk]:
        return self._load("parents", ParentChunk, tickers)

    def _load(self, kind: str, cls, tickers):
        wanted = {t.upper() for t in tickers} if tickers else None
        files = [
            fp for fp in sorted(self.store_dir.glob(f"*.{kind}.jsonl"))
            if wanted is None or fp.name.split("_")[0] in wanted
        ]
        if not files:
            raise FileNotFoundError(
                f"no {kind} JSONL in {self.store_dir} — run ingestion first "
                f"(python -m sec_rag.ingestion.pipeline)")
        out = []
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                out.extend(cls.model_validate_json(line)
                           for line in f if line.strip())
        return out

    def parents_by_id(self, tickers: list[str] | None = None) -> dict:
        return {p.id: p for p in self.load_parents(tickers)}
