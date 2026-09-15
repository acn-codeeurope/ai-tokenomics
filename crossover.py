#!/usr/bin/env python3
"""crossover.py — at what volume does the graph version start paying off?

The single-pass measurement says Attempt 2 sends MORE input tokens than
Attempt 1, because splitting one call into several repeats the instructions
and the item text every time. That is true and it is the honest headline.

What it hides is that Attempt 2's prefixes are shared across markets, so they
are reused far more often as volume grows. Reuse is what a cache converts into
money. This script finds the point where that conversion overtakes the
overhead — and reports it as a volume, not as a slogan."""

from collections import Counter
import measure, rulebook, feedback, prompts
from cost_model import PRICE_IN, PRICE_OUT, CACHE_READ

ENC = "o200k_base"
CACHE_WRITE = 1.25          # first write costs more than a plain input token


def expected_output(item, mode):
    """The answer the architecture is supposed to produce, so output tokens are
    measured rather than guessed."""
    if mode == 1:
        return "[" + ", ".join(f'"{c}"' for c in item.truth) + "]"
    return ""                      # per-category worker answers with an id or nothing


def profile(calls_for, mode, repeats):
    pref = Counter()
    prefix_tok, suffix_tok, out_tok = {}, 0, 0
    for item in feedback.workload(repeats):
        scope = rulebook.scope_for(item.market)
        for prefix, full in calls_for(item, rulebook.scope_for):
            pref[prefix] += 1
            if prefix not in prefix_tok:
                prefix_tok[prefix] = measure.count(prefix, ENC)
            suffix_tok += measure.count(full, ENC) - prefix_tok[prefix]
        if mode == 1:
            out_tok += measure.count(expected_output(item, 1), ENC)
        else:
            for c in scope:
                out_tok += measure.count(c.cid if c.cid in item.truth else "", ENC)

    calls = sum(pref.values())
    prefix_total = sum(prefix_tok[p] * n for p, n in pref.items())
    first_writes = sum(prefix_tok[p] for p in pref)
    repeat_reads = prefix_total - first_writes

    plain = (prefix_total + suffix_tok) * PRICE_IN / 1e6 + out_tok * PRICE_OUT / 1e6
    cached = ((first_writes * CACHE_WRITE + repeat_reads * CACHE_READ + suffix_tok)
              * PRICE_IN / 1e6 + out_tok * PRICE_OUT / 1e6)
    return dict(calls=calls, tin=prefix_total + suffix_tok, out=out_tok,
                distinct=len(pref), plain=plain, cached=cached)


def table(repeats=(1, 10, 50, 200, 1000)) -> None:
    """The projection table. A function, not a `__main__` block, so importing
    this module from a notebook actually prints something."""
    print(f"\nencoding {ENC} · ${PRICE_IN:.2f}/1M in · ${PRICE_OUT:.2f}/1M out · "
          f"cache read {CACHE_READ}x · cache write {CACHE_WRITE}x\n")

    hdr = (f"{'runs':>6} {'calls inc':>9} {'calls fan':>9} | {'INCIDENT':>9} "
           f"{'scoped':>9} {'fan-out':>9} | {'inc cache':>10} {'scp cache':>9} "
           f"{'fan cache':>9} | savings vs incident")
    print(hdr)
    print("-" * len(hdr))

    seen = {}
    for n in repeats:
        inc = profile(prompts.incident_calls, 1, n)
        a1 = profile(prompts.attempt1_calls, 1, n)
        a2 = profile(prompts.attempt2_calls, 2, n)
        seen[n] = (inc, a1, a2)
        b0, b1, b2 = (min(x["plain"], x["cached"]) for x in (inc, a1, a2))
        print(f"{n:>6} {inc['calls']:>9,} {a2['calls']:>9,} | "
              f"{inc['plain']:>9.4f} {a1['plain']:>9.4f} {a2['plain']:>9.4f} | "
              f"{inc['cached']:>10.4f} {a1['cached']:>9.4f} {a2['cached']:>9.4f} | "
              f"scope {b0/b1:5.2f}x · fan-out {b0/b2:5.2f}x")

    print("\ndistinct prefixes stay flat while volume grows — that is the whole trick:")
    for n in (repeats[0], repeats[-1]):
        inc, a1, a2 = seen[n]          # reuse, do not re-tokenise 15,000 prompts
        print(f"  {n:>5} runs   incident {inc['distinct']:>3} prefixes / "
              f"{inc['calls']:>7,} calls   scoped {a1['distinct']:>3} / "
              f"{a1['calls']:>7,}   fan-out {a2['distinct']:>3} / {a2['calls']:>7,}")


if __name__ == "__main__":
    table()
