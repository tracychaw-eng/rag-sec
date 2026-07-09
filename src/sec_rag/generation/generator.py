"""Answer generation with mandatory citations and intent-specific formats.

Design targets from the baseline eval:
  - answer_relevancy 0.574 with 0.0 on hedging answers → answer the question
    directly first, no adjacent-topic padding, hedge only when truly absent
  - RAGAS's noncommittal classifier zeroes conditional risk-factor prose
    (even human reference answers) → open with an affirmative enumerating
    sentence
  - faithfulness gaps on reasoning → separate Evidence from Inference
  - citation discipline: every claim cites [TICKER, Item X] so grounding is
    checkable by humans, judges, and (later) an automated citation validator
"""

from typing import AsyncIterator

from openai import AsyncOpenAI

from ..config import Settings
from ..models import ContextBlock, QueryPlan
from ..observability import tracing

# Bump when any generation prompt changes — part of the answer-cache key,
# so a prompt edit can never serve answers produced by the old prompt.
PROMPT_VERSION = "2"

FACTUAL_SYSTEM = """You are a financial document analyst answering from SEC 10-K excerpts.

Rules:
1. Answer ONLY from the provided context. Never use outside knowledge.
2. Lead with the direct answer to the question in the first sentence, stated
   affirmatively. When the question asks what a company discloses about a
   topic, open with one sentence that enumerates the key points ("X discloses
   three risks: A, B, and C."), then elaborate. Do not open with conditional
   or hedged phrasing, and do not add background or adjacent topics unless
   essential to the answer.
3. Cite every factual claim inline as [TICKER, Item N] using the source
   labels shown in the context.
4. If the answer is genuinely absent from the context, say exactly:
   "This information is not available in the provided documents." — but only
   when NOTHING in the context answers it. If partial, answer what is there
   and state precisely what is missing in one sentence."""

REASONING_SYSTEM = """You are a financial analyst reasoning over SEC 10-K excerpts.

The question requires inference — the answer may not be stated directly.

Structure your answer:
1. Direct answer first (one or two sentences naming the conclusion).
2. "Evidence:" — the specific facts from the context you rely on, each cited
   inline as [TICKER, Item N].
3. "Inference:" — how the evidence supports the conclusion. Keep inference
   clearly separated from cited evidence.
Never use outside knowledge. If the evidence is insufficient, state the best
supported conclusion and what evidence is missing — do not just abstain."""

COMPARISON_SYSTEM = """You are a financial analyst comparing SEC 10-K disclosures across companies.

The context below contains excerpts grouped per company. Each company's
excerpts are delimited and labeled — never attribute one company's
disclosure to another.

Structure your answer:
1. One or two sentences of direct comparison verdict.
2. Per company: what its filing discloses, cited inline as [TICKER, Item N].
3. Key differences and similarities.
Use only the provided context. Answer only what is asked."""

ABSTAIN_TEXT = "This information is not available in the provided documents."


def _format_context(contexts: list[ContextBlock], group_by_ticker: bool) -> str:
    if not group_by_ticker:
        return "\n\n---\n\n".join(c.parent.text for c in contexts)
    by_ticker: dict[str, list[str]] = {}
    for c in contexts:
        by_ticker.setdefault(c.parent.ticker, []).append(c.parent.text)
    blocks = []
    for ticker, texts in by_ticker.items():
        inner = "\n\n".join(texts)
        blocks.append(f"===== EXCERPTS FROM {ticker} 10-K =====\n{inner}\n"
                      f"===== END OF {ticker} EXCERPTS =====")
    return "\n\n".join(blocks)


class Generator:
    def __init__(self, cfg: Settings, openai_client: AsyncOpenAI | None = None):
        self.cfg = cfg
        self.openai = openai_client or AsyncOpenAI(api_key=cfg.openai_api_key)

    def _messages(self, question: str, plan: QueryPlan,
                  contexts: list[ContextBlock],
                  user_context: str | None = None) -> list[dict]:
        system = {
            "factual": FACTUAL_SYSTEM,
            "reasoning": REASONING_SYSTEM,
            "comparison": COMPARISON_SYSTEM,
        }[plan.intent]
        if user_context:
            system = f"{system}\n\n{user_context}"
        context_str = _format_context(
            contexts, group_by_ticker=(plan.intent == "comparison"))
        return [
            {"role": "system", "content": system},
            {"role": "user",
             "content": f"Context:\n{context_str}\n\nQuestion: {question}"},
        ]

    async def generate(self, question: str, plan: QueryPlan,
                       contexts: list[ContextBlock],
                       user_context: str | None = None) -> str:
        if not contexts:
            return ABSTAIN_TEXT
        resp = await self.openai.chat.completions.create(
            model=self.cfg.llm_model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_answer_tokens,
            messages=self._messages(question, plan, contexts, user_context),
        )
        tracing.record_llm(self.cfg.llm_model, resp.usage, kind="generate")
        return resp.choices[0].message.content.strip()

    async def stream(self, question: str, plan: QueryPlan,
                     contexts: list[ContextBlock],
                     user_context: str | None = None) -> AsyncIterator[str]:
        """Yields text deltas. Usage is reported from the final stream chunk
        (stream_options.include_usage), so cost metering matches generate()."""
        if not contexts:
            yield ABSTAIN_TEXT
            return
        stream = await self.openai.chat.completions.create(
            model=self.cfg.llm_model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_answer_tokens,
            messages=self._messages(question, plan, contexts, user_context),
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.usage is not None:
                tracing.record_llm(self.cfg.llm_model, chunk.usage,
                                   kind="generate")
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
