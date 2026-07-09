"""Chunk store — parents and children as JSONL per ticker.

Parents are the generation payload and live only here (not in the vector
DB). Children are duplicated: text+metadata here (BM25 + inspection),
vectors in Qdrant with the chunk id in the payload.
"""

from pathlib import Path

from ..models import ChildChunk, ParentChunk


class ChunkStore:
    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str, kind: str) -> Path:
        return self.store_dir / f"{ticker.upper()}.{kind}.jsonl"

    def save(self, ticker: str, parents: list[ParentChunk],
             children: list[ChildChunk]) -> None:
        for kind, chunks in (("parents", parents), ("children", children)):
            with open(self._path(ticker, kind), "w", encoding="utf-8") as f:
                for c in chunks:
                    f.write(c.model_dump_json() + "\n")

    def load_children(self, tickers: list[str] | None = None) -> list[ChildChunk]:
        return self._load("children", ChildChunk, tickers)

    def load_parents(self, tickers: list[str] | None = None) -> list[ParentChunk]:
        return self._load("parents", ParentChunk, tickers)

    def _load(self, kind: str, cls, tickers):
        files = (
            [self._path(t, kind) for t in tickers] if tickers
            else sorted(self.store_dir.glob(f"*.{kind}.jsonl"))
        )
        out = []
        for fp in files:
            if not fp.exists():
                raise FileNotFoundError(
                    f"{fp} not found — run ingestion first "
                    f"(python -m sec_rag.ingestion.pipeline)")
            with open(fp, encoding="utf-8") as f:
                out.extend(cls.model_validate_json(line) for line in f if line.strip())
        return out

    def parents_by_id(self, tickers: list[str] | None = None) -> dict:
        return {p.id: p for p in self.load_parents(tickers)}

    def available_tickers(self) -> list[str]:
        return sorted(fp.name.split(".")[0]
                      for fp in self.store_dir.glob("*.children.jsonl"))
