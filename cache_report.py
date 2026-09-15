#!/usr/bin/env python3
"""cache_report.py — how many distinct prefixes a run produces, and how often
each one is reused. This is the ceiling on what any cache can do for you, and
it is decided by the architecture before anyone looks at a price list."""

from collections import Counter
import measure, rulebook, feedback, prompts

ENC = "o200k_base"


def run(calls_for, label):
    pref_counts, tok_in_total, calls = Counter(), 0, 0
    prefix_tokens = {}
    for item in feedback.workload():
        for prefix, full in calls_for(item, rulebook.scope_for):
            calls += 1
            pref_counts[prefix] += 1
            if prefix not in prefix_tokens:
                prefix_tokens[prefix] = measure.count(prefix, ENC)
            tok_in_total += measure.count(full, ENC)

    distinct = len(pref_counts)
    reuse = sum(pref_counts.values()) / distinct
    # tokens that a cache could serve on repeat hits: every occurrence of a
    # prefix after the first one
    cacheable = sum(prefix_tokens[p] * (n - 1) for p, n in pref_counts.items())

    print(f"\n{label}")
    print(f"  calls in the run          {calls:>10,}")
    print(f"  input tokens, uncached    {tok_in_total:>10,}")
    print(f"  distinct prefixes         {distinct:>10,}")
    print(f"  average reuse per prefix  {reuse:>10.1f}x")
    print(f"  tokens a cache could serve{cacheable:>10,}"
          f"   ({cacheable/tok_in_total:5.1%} of input)")
    return dict(calls=calls, tok_in=tok_in_total, distinct=distinct,
                reuse=reuse, cacheable=cacheable)


if __name__ == "__main__":
    print(f"encoding: {ENC}   items: {len(feedback.ITEMS)}   "
          f"categories: {len(rulebook.CATEGORIES)}")
    a1 = run(prompts.attempt1_calls, "ATTEMPT 1 — one call, whole in-scope rulebook")
    a2 = run(prompts.attempt2_calls, "ATTEMPT 2 — one call per in-scope category")

    print("\n--- what actually changed ---")
    print(f"  calls                     {a1['calls']:>10,}  ->{a2['calls']:>9,}"
          f"   ({a2['calls']/a1['calls']:.1f}x)")
    print(f"  input tokens, uncached    {a1['tok_in']:>10,}  ->{a2['tok_in']:>9,}"
          f"   ({a2['tok_in']/a1['tok_in']:.2f}x)")
    print(f"  distinct prefixes         {a1['distinct']:>10,}  ->{a2['distinct']:>9,}")
    print(f"  reuse per prefix          {a1['reuse']:>9.1f}x  ->{a2['reuse']:>8.1f}x")
