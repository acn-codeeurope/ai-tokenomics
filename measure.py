#!/usr/bin/env python3
"""
measure.py — counts tokens, not opinions.

THREE THINGS THIS MEASURES
  1. tokens        exact counts from a real tokenizer, per prompt
  2. prefix        how many leading tokens are identical across the prompts a
                   run actually sends — this is what decides whether a cache
                   can help you at all, and it is a property of the
                   architecture, not of the price list
  3. money         verified prices applied to measured tokens

WHY THE TOKENIZER IS NAMED EVERYWHERE
  Token counts are not portable. The same rulebook measured with two encodings
  gives two different numbers, and quoting one vendor's count against another
  vendor's price is the exact mistake this talk is about. Every figure this
  module returns carries the encoding that produced it.
"""

from dataclasses import dataclass
import tiktoken

# Encodings we can measure offline. o200k_base is the current GPT-family
# encoding; cl100k_base is the previous one and is kept precisely so the
# spread between them is visible rather than assumed.
ENCODINGS = ("o200k_base", "cl100k_base")
_enc_cache = {}


def enc(name: str):
    if name not in _enc_cache:
        _enc_cache[name] = tiktoken.get_encoding(name)
    return _enc_cache[name]


def tokens(text: str, encoding: str = "o200k_base"):
    return enc(encoding).encode(text)


def count(text: str, encoding: str = "o200k_base") -> int:
    return len(tokens(text, encoding))


def spread(text: str) -> dict:
    """The same text under every encoding we can measure. The point of this
    function is that the answers differ."""
    return {e: count(text, e) for e in ENCODINGS}


@dataclass
class PrefixReport:
    prompts: int
    shared: int          # leading tokens identical across ALL prompts
    shortest: int        # tokens in the shortest prompt
    total: int           # tokens across all prompts

    @property
    def share(self) -> float:
        """Fraction of the average prompt that is a stable, cacheable prefix."""
        avg = self.total / self.prompts if self.prompts else 0
        return self.shared / avg if avg else 0.0


def prefix_report(prompt_texts, encoding: str = "o200k_base") -> PrefixReport:
    """How much of these prompts is a prefix that never changes.

    A cache can only help from the first differing token onward, so this is the
    ceiling on what caching can ever save — before any price is considered."""
    seqs = [tokens(p, encoding) for p in prompt_texts]
    if not seqs:
        return PrefixReport(0, 0, 0, 0)
    shared = 0
    for i in range(min(len(s) for s in seqs)):
        first = seqs[0][i]
        if all(s[i] == first for s in seqs[1:]):
            shared += 1
        else:
            break
    return PrefixReport(len(seqs), shared,
                        min(len(s) for s in seqs),
                        sum(len(s) for s in seqs))


# --- money ------------------------------------------------------------------
# Prices are not measured here; they come from the verified table in
# cost_model.py so there is exactly one place to correct if a vendor moves.
try:
    from cost_model import PRICE_IN, PRICE_OUT, CACHE_READ
except ImportError:                                   # running from repo root
    from notebook.cost_model import PRICE_IN, PRICE_OUT, CACHE_READ


def usd(tokens_in: int, tokens_out: int,
        cached_in: int = 0,
        price_in: float = PRICE_IN, price_out: float = PRICE_OUT) -> float:
    """Cost of one call. `cached_in` is the part of tokens_in served from cache."""
    fresh = max(tokens_in - cached_in, 0)
    return (fresh * price_in + cached_in * price_in * CACHE_READ
            + tokens_out * price_out) / 1e6


def fmt(n) -> str:
    return f"{n:,}"
