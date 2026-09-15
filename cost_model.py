#!/usr/bin/env python3
"""
cost_model.py — the arithmetic behind the story, so no number on a slide
is asserted rather than derived.

THE MECHANISM (corrected by Grzegorz, 12 Sept 2026)

  One model call per piece of feedback — NOT one per category.
  The prompt is assembled dynamically: base instructions + the
  qualification rules of every category in scope + the feedback text.
  The model returns a LIST of matching categories.

  So the fan-out that cost the money lives INSIDE one prompt.
  Going from 5 categories in scope to 50 multiplies the rules block,
  not the call count.

  DEV volume is driven by people, not by a schedule: several developers,
  each running their own local prompt variant, many times a day, before
  anything reaches the repo.

WHAT THIS SCRIPT IS FOR
  Plug in what you know, see what falls out. Anything you do not know,
  solve for it with solve_for_items().

Run:  python3 notebook/cost_model.py
"""

from dataclasses import dataclass, replace


# --- prices: verified 12 Sept 2026 against the vendors' own pricing pages,
#     one URL per figure ---------------------------------------------------
# Vendor-neutral figures derived from three current workhorse models whose
# input price is identical to the cent ($2.00): Claude Sonnet 5, gpt-5.6-terra,
# gemini-3.1-pro-preview. Output median $12. Output is ~5x input across the board.
PRICE_IN = 2.00    # $ per 1M input tokens  (typical mid-tier)
PRICE_OUT = 12.00  # $ per 1M output tokens (typical mid-tier)
FRONTIER_IN, FRONTIER_OUT = 10.00, 50.00
BATCH_DISCOUNT = 0.50   # -50% on IN and OUT, all four major providers
CACHE_READ = 0.10       # cached input costs 0.1x normal input
PRICES_ARE_VERIFIED = True


@dataclass
class Prompt:
    base: int          # system instructions, output format
    rule: int          # tokens for ONE category's qualification rules
    feedback: int      # the customer reply itself
    out: int           # the returned list of categories

    def tokens_in(self, categories: int) -> int:
        return self.base + categories * self.rule + self.feedback


@dataclass
class DevLoad:
    """DEV is people iterating, not a cron.

    prompt_variants is the multiplier everyone forgets: a developer does not
    run one prompt, they run several competing versions of it before pushing
    the winner. Every variant re-sends the whole rules block."""
    developers: int
    prompt_variants: int
    runs_per_variant_day: int
    items_per_run: int
    working_days: int = 21

    @property
    def runs_per_month(self) -> int:
        return (self.developers * self.prompt_variants
                * self.runs_per_variant_day * self.working_days)

    @property
    def calls_per_month(self) -> int:
        return self.runs_per_month * self.items_per_run


def monthly_cost(load: DevLoad, p: Prompt, categories: int,
                 retry_factor: float = 1.0,
                 price_in: float = PRICE_IN, price_out: float = PRICE_OUT) -> float:
    per_call = (p.tokens_in(categories) * price_in / 1e6
                + p.out * price_out / 1e6)
    return load.calls_per_month * per_call * retry_factor


def rules_bill(load: DevLoad, p: Prompt, categories: int,
               price_in: float = PRICE_IN) -> float:
    """What you pay purely to re-send the rulebook, per month."""
    return load.calls_per_month * categories * p.rule * price_in / 1e6


def rules_bill_cached(load: DevLoad, p: Prompt, categories: int,
                      price_in: float = PRICE_IN,
                      write_mult: float = 1.25, read_mult: float = 0.10) -> float:
    """Same rulebook, cached once per run and read by every item in it."""
    block = categories * p.rule
    per_run = block * price_in * (write_mult + (load.items_per_run - 1) * read_mult) / 1e6
    return load.runs_per_month * per_run


def solve_for_items(target_usd: float, load: DevLoad, p: Prompt,
                    categories: int, retry_factor: float = 1.0) -> float:
    """How many items per run would produce `target_usd` a month?"""
    one = monthly_cost(replace(load, items_per_run=1), p, categories, retry_factor)
    return target_usd / one if one else float("nan")


def report(load: DevLoad, p: Prompt, agreed: int, actual: int,
           retries_before: float, retries_after: float) -> None:
    before = monthly_cost(load, p, agreed, retries_before)
    after = monthly_cost(load, p, actual, retries_after)
    tin_b, tin_a = p.tokens_in(agreed), p.tokens_in(actual)

    print(f"  calls/month           {load.calls_per_month:>12,}")
    print(f"  input tokens/call     {tin_b:>12,}  ->{tin_a:>10,}"
          f"   ({tin_a/tin_b:.2f}x)")
    print(f"  retry factor          {retries_before:>12.2f}  ->{retries_after:>10.2f}")
    # NOTE: the token ratio and the cost ratio are NOT the same number.
    # Output tokens do not grow with categories and are priced higher, so
    # they dilute the effect. Quote the cost ratio, never the token ratio.
    cost_ratio_prompt = (monthly_cost(load, p, actual, retries_before)
                         / monthly_cost(load, p, agreed, retries_before))
    print(f"  MONTHLY COST          {before:>11,.0f}$  ->{after:>9,.0f}$"
          f"   ({after/before:.2f}x)")
    print(f"    input tokens grew       {tin_a/tin_b:.2f}x")
    print(f"    but COST from prompt    {cost_ratio_prompt:.2f}x"
          f"   <- output tokens dilute it")
    print(f"    x retries               {retries_after/retries_before:.2f}x")
    print(f"    = total                 {cost_ratio_prompt*retries_after/retries_before:.2f}x")


if __name__ == "__main__":
    print(__doc__.split("Run:")[0].strip())
    print("\n" + "=" * 74)
    if not PRICES_ARE_VERIFIED:
        print("!! PRICES ARE PLACEHOLDERS — do not put these figures on a slide !!")
    print("=" * 74)

    # half a page of rules per category ~= 250 words ~= 350 tokens
    prompt = Prompt(base=400, rule=350, feedback=300, out=200)

    load = DevLoad(developers=4, prompt_variants=3,
                   runs_per_variant_day=4, items_per_run=1)
    need = solve_for_items(500, load, prompt, categories=5, retry_factor=1.0)
    print(f"\n4 devs x 3 prompt variants x 4 runs/day x 21 days"
          f" = {load.runs_per_month:,} runs/month.")
    print(f"To reach $500/month, each run processes ~{need:,.0f} items.\n")

    load = replace(load, items_per_run=round(need))
    print("AGREED (5 categories, no retries)  ->  AFTER (50 categories, retries)")
    report(load, prompt, agreed=5, actual=50,
           retries_before=1.0, retries_after=1.65)

    print("\n--- sensitivity: how much of the 10x is prompt vs retries ---")
    print("  (cost ratios, not token ratios)")
    for rule_tokens in (150, 250, 400, 600):
        p2 = replace(prompt, rule=rule_tokens)
        cost_growth = (monthly_cost(load, p2, 50) / monthly_cost(load, p2, 5))
        needed_retry = 10 / cost_growth
        print(f"  rules {rule_tokens:>4} tok/category -> cost from prompt {cost_growth:5.2f}x,"
              f"  retries must supply {needed_retry:4.2f}x to reach 10x")

    print("\n--- what the rulebook alone costs, after the override ---")
    raw = rules_bill(load, prompt, 50)
    cached = rules_bill_cached(load, prompt, 50)
    print(f"  rules block in every prompt   {50*prompt.rule:>10,} tokens")
    print(f"  re-sent, uncached             {raw:>10,.0f} $/month")
    print(f"  same block, cached per run    {cached:>10,.0f} $/month"
          f"   ({raw/cached:.1f}x cheaper)")
    print(f"  -> avoidable                  {raw-cached:>10,.0f} $/month")

    print("\n--- configured max retry attempts, per environment ---")
    for env, r in (("DEV", 50), ("TST", 30), ("PRD", 10)):
        worst = r * prompt.tokens_in(50) * PRICE_IN / 1e6
        print(f"  {env}: {r:>3} attempts -> one permanently failing item can burn"
              f" {r*prompt.tokens_in(50):>9,} tokens = {worst:5.2f} $")
    print("  The environment with the least protection was configured to try hardest.")
