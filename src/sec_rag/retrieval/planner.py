"""Query planner — one cheap LLM call that replaces the legacy pipeline's
hard-coded question-ID routing (COMPARISON_SOURCES["R005"] = [...]) with a
runtime decision: intent classification, ticker extraction, and rewriting.

The rewrite also subsumes chat-history condensation: pass `history` and
follow-ups become standalone queries.
"""

import json

from openai import AsyncOpenAI

from ..config import Settings
from ..models import QueryPlan
from ..observability import tracing

PLANNER_PROMPT = """You are a query planner for a retrieval system over SEC 10-K filings.

Available companies (ticker: name):
{tickers}

Analyze the user's question and return JSON with these fields:
- "intent": one of
    "factual"    — asks for specific facts/figures/disclosures, answerable from one place
    "comparison" — asks to compare/contrast two or more companies
    "reasoning"  — requires inference or synthesis beyond what is directly stated
- "tickers": list of tickers the question is about. Use [] if the question
  does not name or clearly imply specific companies (means: search all).
  Comparative questions like "which company is most X" imply ALL companies.
- "years": list of filing years (integers) ONLY if the question explicitly
  restricts to specific years or asks to compare across years (e.g.
  "in the 2025 10-K", "how did X change year over year"). Use [] otherwise —
  most questions want the latest information and should not be restricted.
- "rewritten_query": the question rewritten as a standalone, retrieval-friendly
  query (resolve pronouns using the conversation history if provided).
- "paraphrases": 2 alternative phrasings using different vocabulary — think of
  the words a 10-K filing itself would use (e.g. "hacking" -> "cybersecurity
  incidents, unauthorized access"; "outsourcing" -> "third parties, vendors,
  service providers, central counterparties").
- "reason": one short sentence explaining the classification.

Return ONLY the JSON object."""


class QueryPlanner:
    def __init__(self, cfg: Settings, openai_client: AsyncOpenAI | None = None,
                 known_tickers: dict[str, str] | None = None):
        self.cfg = cfg
        self.openai = openai_client or AsyncOpenAI(api_key=cfg.openai_api_key)
        self.known_tickers = known_tickers or {
            "MSFT": "Microsoft", "NVDA": "NVIDIA", "JPM": "JPMorgan Chase",
        }

    async def plan(self, question: str,
                   history: list[dict] | None = None,
                   user_context: str | None = None) -> QueryPlan:
        ticker_list = "\n".join(f"  {t}: {n}" for t, n in self.known_tickers.items())
        messages = [
            {"role": "system",
             "content": PLANNER_PROMPT.format(tickers=ticker_list)},
        ]
        if user_context:
            messages.append({"role": "user", "content": user_context})
        if history:
            convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
            messages.append({"role": "user",
                             "content": f"Conversation so far:\n{convo}"})
        messages.append({"role": "user", "content": f"Question: {question}"})

        resp = await self.openai.chat.completions.create(
            model=self.cfg.planner_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        tracing.record_llm(self.cfg.planner_model, resp.usage, kind="planner")
        try:
            data = json.loads(resp.choices[0].message.content)
            tickers = [t.upper() for t in data.get("tickers", [])
                       if t.upper() in self.known_tickers]
            intent = data.get("intent", "factual")
            if intent not in ("factual", "comparison", "reasoning"):
                intent = "factual"
            return QueryPlan(
                intent=intent,
                tickers=tickers,
                years=[int(y) for y in data.get("years", [])
                       if str(y).isdigit()][:4],
                rewritten_query=data.get("rewritten_query") or question,
                paraphrases=[p for p in data.get("paraphrases", [])[:2]
                             if isinstance(p, str)],
                reason=data.get("reason", ""),
            )
        except (json.JSONDecodeError, TypeError, KeyError):
            # Planner failure must never block answering — degrade to a
            # plain factual search over all tickers.
            return QueryPlan(intent="factual", tickers=[],
                             rewritten_query=question,
                             reason="planner-parse-failure fallback")
