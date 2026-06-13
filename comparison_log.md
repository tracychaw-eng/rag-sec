# Embedding Model Comparison Log

## Run 1 — text-embedding-ada-002 (previous sessions, in-memory)
Approx cost per index: $0.001487
Score range: 0.830 – 0.871

| Q# | Question | Top Score | Source |
|----|----------|-----------|--------|
| 1  | Main risk factors | 0.830 | jpm_10k.txt, nvda_10k.txt |
| 2  | Cybersecurity risks | 0.867 | msft_10k.txt, jpm_10k.txt, nvda_10k.txt |
| 3  | Competition | 0.838 | msft_10k.txt, jpm_10k.txt |
| 4  | Regulatory risks | 0.850 | msft_10k.txt, jpm_10k.txt, nvda_10k.txt |
| 5  | Macroeconomic risks | 0.871 | msft_10k.txt, jpm_10k.txt, nvda_10k.txt |

## Run 2 — text-embedding-3-small (today)
Approx cost per index: $$0.001487
Score range: 0.539 – 0.720

| Q# | Question | Top Score | Source |
|----|----------|-----------|--------|
| 1  | Main risk factors | 0.539 | jpm_10k.txt, nvda_10k.txt |
| 2  | Cybersecurity risks | 0.720 | nvda_10k.txt, msft_10k.txt |
| 3  | Competition | 0.627 | nvda_10k.txt, msft_10k.txt |
| 4  | Regulatory risks | 0.644 | msft_10k.txt |
| 5  | Macroeconomic risks | 0.649 | jpm_10k.txt, msft_10k.txt |

## Observations
- ada-002 produces higher absolute cosine scores (0.83–0.87 vs 0.54–0.72)
- Score difference is likely due to different vector space distributions,
  not necessarily better retrieval quality
- Answer content from 3-small was comparable or more detailed on Q2 and Q5
- ada-002 costs 5x more ($0.100 vs $0.020 per 1M tokens)
- Conclusion: 3-small is the better choice — equal or better answer quality
  at a fraction of the cost. Never use raw cosine scores to compare across
  different embedding models