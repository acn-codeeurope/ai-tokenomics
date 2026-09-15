#!/usr/bin/env python3
"""
prompts.py — how each architecture lays out a call.

THE ONE RULE THAT DECIDES EVERYTHING
  A prefix cache matches from token zero and stops at the first token that
  differs. So the only layout that can ever be cached is:

      [ stable: instructions + rules ]  [ variable: the item ]

  Both attempts below put the item last. That is deliberate: it means the
  difference we measure between them is the difference in ARCHITECTURE, not a
  layout mistake in one of them. If the original system interleaved the item
  with the rules, its cache hit rate was zero no matter what anyone paid for.

ATTEMPT 1  one call per item; the rules of every in-scope category inline.
           The prefix is the market's scope, so there are as many distinct
           prefixes as there are distinct market scopes.

ATTEMPT 2  one call per category in scope; each worker carries exactly one
           category's rules. The prefix is that category, so there are as many
           distinct prefixes as there are categories in use — and every market
           that shares a category shares its prefix.
"""

BASE_1 = (
    "You classify customer feedback against a fixed taxonomy.\n"
    "Return every category the message satisfies, as a JSON list of category "
    "ids. Return an empty list if none apply. Never return prose, never return "
    "a single best guess in place of the full list.\n"
    "The qualification rules follow. They are binding and complete.\n\n"
)

BASE_2 = (
    "You decide whether one customer message satisfies one category.\n"
    "Answer with the category id if it does, or with an empty string if it "
    "does not. Never return prose and never explain.\n"
    "The qualification rules for that one category follow. They are binding.\n\n"
)

ITEM_HEADER = "\n\n--- customer message ---\n"


def _rules(categories):
    return "\n\n".join(c.rules_text() for c in categories)


def attempt1(item, categories):
    """Returns (prefix, full_prompt). The prefix is everything a cache could
    ever match on."""
    prefix = BASE_1 + _rules(categories) + ITEM_HEADER
    return prefix, prefix + item.text


def attempt2(item, category):
    prefix = BASE_2 + category.rules_text() + ITEM_HEADER
    return prefix, prefix + item.text


def attempt1_calls(item, scope_for):
    """One call carrying the whole in-scope rulebook."""
    return [attempt1(item, scope_for(item.market))]


def attempt2_calls(item, scope_for):
    """One call per in-scope category — the fan-out, one worker each."""
    return [attempt2(item, c) for c in scope_for(item.market)]


def incident_calls(item, scope_for):
    """The state the incident actually ran in: the scope guardrail no longer
    bound, so every call carried the whole taxonomy, not the market's two to
    five categories. Kept separate so the comparison has three points, not
    two — otherwise we would be measuring two fixes against each other and
    calling the smaller one a regression."""
    import rulebook
    return [attempt1(item, rulebook.all_categories())]
