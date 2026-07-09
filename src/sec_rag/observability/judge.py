"""Online faithfulness judging — samples production answers and scores
whether their claims are supported by the retrieved context.

Fire-and-forget: judging never adds latency to the user's request. Results
append to logs/judge.jsonl and feed a rolling in-process window exposed on
/health, so a hallucination-rate drift shows up between nightly evals.
"""

import json
import logging
import random
import threading
from collections import deque
from pathlib import Path

from openai import AsyncOpenAI

logger = logging.getLogger("sec_rag.judge")

JUDGE_PROMPT = """You are auditing a RAG system for hallucinations.

Given a question, the retrieved context, and the system's answer:
1. List each distinct factual claim the answer makes (ignore hedges and
   meta-statements like "the context does not mention X").
2. For each claim, decide whether the context supports it.

Return JSON: {"total_claims": int, "supported_claims": int,
"unsupported": [short description of each unsupported claim]}"""


class OnlineJudge:
    def __init__(self, openai_client: AsyncOpenAI, model: str,
                 sample_rate: float, log_path: Path,
                 window: int = 200):
        self.openai = openai_client
        self.model = model
        self.sample_rate = sample_rate
        self.log_path = Path(log_path)
        self._scores: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()
        self.sampled = 0
        self.failed = 0

    def should_sample(self) -> bool:
        return self.sample_rate > 0 and random.random() < self.sample_rate

    def stats(self) -> dict:
        with self._lock:
            scores = list(self._scores)
        return {
            "sample_rate": self.sample_rate,
            "sampled": self.sampled,
            "failed": self.failed,
            "window_n": len(scores),
            "faithfulness_avg": (round(sum(scores) / len(scores), 3)
                                 if scores else None),
        }

    async def judge(self, trace_id: str, question: str, answer: str,
                    contexts: list[str]) -> None:
        """Runs as a background task — must never raise into the caller."""
        try:
            resp = await self.openai.chat.completions.create(
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content":
                        f"Question: {question}\n\n"
                        f"Context:\n{chr(10).join(contexts)[:24000]}\n\n"
                        f"Answer: {answer}"},
                ],
            )
            data = json.loads(resp.choices[0].message.content)
            total = max(1, int(data.get("total_claims", 1)))
            supported = min(total, int(data.get("supported_claims", 0)))
            score = supported / total

            with self._lock:
                self._scores.append(score)
                self.sampled += 1

            record = {
                "trace_id": trace_id, "score": round(score, 3),
                "total_claims": total, "supported_claims": supported,
                "unsupported": data.get("unsupported", [])[:5],
            }
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            if score < 0.5:
                logger.warning("low faithfulness on sampled answer",
                               extra=record)
        except Exception:
            with self._lock:
                self.failed += 1
            logger.warning("online judge call failed", exc_info=True,
                           extra={"trace_id": trace_id})
