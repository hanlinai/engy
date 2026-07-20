"""Synthetic prompts of a controlled length.

Every prompt starts with a unique nonce so the server's prefix cache cannot
serve a warm KV block: a cached prefix would report a TTFT the real workload
never sees.
"""
from __future__ import annotations

import random

# A small closed vocabulary keeps the token/word ratio near 1:1 across
# tokenizers — every word here is common enough to be a single token.
VOCAB = (
    "the of and to in that is for it with as was on be at by this have from or "
    "one had not but what all were when we there can an your which their said "
    "if do will each about how up out them then she many some so these would "
    "into has more her two like him time see no could my than first been call"
).split()


def synth_prompt(*, seed: int, index: int, tokens: int) -> str:
    """Deterministic filler of roughly ``tokens`` words, unique per index."""
    rng = random.Random(f"{seed}:{index}")
    nonce = f"{seed:x}-{index:x}-{rng.getrandbits(32):08x}"
    body = [rng.choice(VOCAB) for _ in range(max(0, tokens - 1))]
    return " ".join([nonce, *body])
