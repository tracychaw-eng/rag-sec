# cost_logger.py
import json
import os
from datetime import datetime

COST_FILE = "embedding_costs.json"

# text-embedding-3-small pricing (per OpenAI as of 2024)
COST_PER_TOKEN = 0.020 / 1_000_000   # $0.020 per 1M tokens

def log_embedding_run(model, num_chunks, tokens_used, cost_usd):
    """Append a run entry to the cost log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "num_chunks": num_chunks,
        "tokens_used": tokens_used,
        "cost_usd": round(cost_usd, 6)
    }

    # Load existing log or start fresh
    if os.path.exists(COST_FILE):
        with open(COST_FILE, "r") as f:
            log = json.load(f)
    else:
        log = []

    log.append(entry)

    with open(COST_FILE, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n--- COST LOG ---")
    print(f"Model:      {model}")
    print(f"Chunks:     {num_chunks}")
    print(f"Tokens:     {tokens_used:,}")
    print(f"Cost:       ${cost_usd:.6f}")

def estimate_tokens(text):
    """Rough estimate: 1 token ≈ 4 characters in English."""
    return len(text) // 4