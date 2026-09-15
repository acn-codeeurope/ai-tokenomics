#!/usr/bin/env python3
"""
rulebook.py — the category taxonomy the two architectures are measured on.

WHY THIS FILE EXISTS
  Every number in this notebook is measured on real text run through a real
  tokenizer. That requires real text. The incident's taxonomy cannot leave the
  client, so this is a stand-in built to the same shape: 50 categories, a
  qualification block of roughly half a page each, and a per-market scope of
  two to five categories — the rule the original team agreed on and enforced
  with a guardrail in code.

  Synthetic content, production shape. Nothing here is client data.

  The numbers that matter are not invented: how many tokens a rules block
  costs, how much of a prompt is stable between calls, and how much smaller
  the scoped prompt is. Those are measured downstream in measure.py.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Category:
    cid: str
    name: str
    theme: str
    includes: tuple
    excludes: tuple
    edge: tuple
    examples: tuple

    def rules_text(self) -> str:
        """Render the qualification block exactly as it would be pasted into a
        prompt. Wording is deliberately repetitive across categories — that is
        what a business-owned taxonomy looks like, and it is also what makes
        the block compressible and cacheable."""
        lines = [
            f"### {self.cid} — {self.name}",
            f"Theme: {self.theme}.",
            "Assign this category when the customer's message does any of the following:",
        ]
        lines += [f"  - {x}" for x in self.includes]
        lines.append("Do not assign this category when:")
        lines += [f"  - {x}" for x in self.excludes]
        lines.append("Edge cases, decided previously and binding:")
        lines += [f"  - {x}" for x in self.edge]
        lines.append("Reference wordings that qualify:")
        lines += [f'  - "{x}"' for x in self.examples]
        lines.append(
            "If the message satisfies this category and others, return all of them. "
            "Never return a best guess in place of the full list. If none of the "
            "conditions above are met, this category must not appear in the answer."
        )
        return "\n".join(lines)


def _cat(cid, name, theme, inc, exc, edge, ex):
    return Category(cid, name, theme, tuple(inc), tuple(exc), tuple(edge), tuple(ex))


# --- the taxonomy -----------------------------------------------------------
# Fifty categories across nine themes. Deliberately overlapping at the edges,
# because that overlap is why the original prompt was told to return a list
# rather than a single label.

CATEGORIES = [
    _cat("CAT-001", "Late delivery", "logistics",
         ["the parcel arrived after the promised date",
          "the customer reports waiting longer than the quoted window",
          "a delivery date was moved more than once",
          "the customer asks where the order currently is"],
         ["the parcel arrived on time but damaged",
          "the delay is caused by the customer being unavailable",
          "the order was never dispatched at all"],
         ["a delay of under 24h still qualifies if the customer raises it",
          "pre-orders are out of scope until the release date passes"],
         ["it was supposed to be here on Tuesday and it is Friday",
          "third time they moved the date on me"]),
    _cat("CAT-002", "Damaged on arrival", "logistics",
         ["the product is broken, cracked, dented or leaking on opening",
          "the outer packaging shows crush or water damage",
          "parts are bent or detached inside a sealed box",
          "the customer sends photographs of physical damage"],
         ["the product is intact but the wrong item",
          "damage appeared after weeks of normal use",
          "the customer damaged it during installation"],
         ["cosmetic scratches count only if the customer calls them a defect",
          "damage to a free gift counts under this category, not under returns"],
         ["arrived with a hole in the side of the box",
          "screen was already cracked when I opened it"]),
    _cat("CAT-003", "Wrong item shipped", "logistics",
         ["the delivered product differs from the one ordered",
          "the size, colour or variant does not match the confirmation",
          "someone else's order arrived instead",
          "a multi-item order contains a substitution nobody agreed to"],
         ["the item matches the order but not the customer's expectation",
          "the customer ordered the wrong thing themselves"],
         ["a substitution flagged at checkout is out of scope",
          "missing items belong to CAT-004, not here"],
         ["ordered the blue one and got the grey one",
          "this is not what is on my invoice"]),
    _cat("CAT-004", "Missing items", "logistics",
         ["part of a multi-item order is absent",
          "accessories or cables listed on the box are not inside",
          "the parcel weight suggests a partial shipment",
          "the customer reports an empty or half-filled package"],
         ["the missing part is sold separately and was never ordered",
          "the item is visible in tracking as a separate parcel in transit"],
         ["missing manuals count; missing promotional inserts do not",
          "if nothing at all arrived, use CAT-001 instead"],
         ["box came with no power adapter",
          "two of the four filters are missing"]),
    _cat("CAT-005", "Courier conduct", "logistics",
         ["the customer describes rude or unprofessional behaviour by the driver",
          "the parcel was left in an unsafe or unagreed place",
          "a delivery was marked attempted when nobody called",
          "the driver refused to wait or to bring the item inside"],
         ["complaints about the delivery time rather than the person",
          "automated notification problems"],
         ["a parcel left with a neighbour counts only if unagreed",
          "threats or abuse escalate immediately and still get this category"],
         ["he marked it delivered and I was standing at the door",
          "left it in the rain by the bins"]),
]

# Categories 6-50 are generated from the same nine themes with the same block
# shape. They exist so the measured token volume matches production: fifty
# blocks, not five.
_THEMES = [
    ("billing", "Billing and payment"),
    ("product", "Product quality"),
    ("app", "App and website"),
    ("support", "Customer support"),
    ("returns", "Returns and refunds"),
    ("packaging", "Packaging and waste"),
    ("pricing", "Pricing and promotions"),
    ("stock", "Availability and stock"),
    ("comms", "Communication and notifications"),
]

_INC = [
    "the customer states the problem explicitly in their own words",
    "the message describes a repeated occurrence rather than a one-off",
    "the customer names the channel or touchpoint where it happened",
    "a reference number, order id or invoice number is quoted",
    "the customer compares the outcome against what was promised",
]
_EXC = [
    "the message only expresses general dissatisfaction with no specifics",
    "the issue was already resolved and the customer says so",
    "the complaint concerns a third party outside our control",
]
_EDGE = [
    "a message in a language other than the market default still qualifies",
    "sarcasm counts as a complaint when the underlying fact is verifiable",
    "forwarded messages count only if the customer adds their own statement",
]

for _i in range(6, 51):
    _theme_key, _theme_name = _THEMES[(_i - 6) % len(_THEMES)]
    _n = (_i - 6) // len(_THEMES) + 1
    CATEGORIES.append(_cat(
        f"CAT-{_i:03d}", f"{_theme_name} — aspect {_n}", _theme_key,
        _INC, _EXC, _EDGE,
        (f"this is the {_n}. time this month and nobody has fixed it",
         f"I have written about this {_theme_key} problem twice already"),
    ))

BY_ID = {c.cid: c for c in CATEGORIES}

# --- market scope -----------------------------------------------------------
# The agreement the original team worked to: two to five categories live per
# country, out of fifty-plus available. A guardrail in the code enforced it.
# That guardrail is the thing this notebook turns back into a visible node.

MARKET_SCOPE = {
    "PL": ("CAT-001", "CAT-002", "CAT-006", "CAT-014"),
    "DE": ("CAT-001", "CAT-003", "CAT-007", "CAT-015", "CAT-022"),
    "FR": ("CAT-002", "CAT-004", "CAT-008"),
    "ES": ("CAT-001", "CAT-009", "CAT-016"),
    "IT": ("CAT-003", "CAT-010", "CAT-017", "CAT-025"),
    "NL": ("CAT-005", "CAT-011"),
    "SE": ("CAT-004", "CAT-012", "CAT-018"),
    "CZ": ("CAT-001", "CAT-013", "CAT-019", "CAT-026"),
    "UK": ("CAT-002", "CAT-006", "CAT-020", "CAT-027", "CAT-033"),
    "US": ("CAT-001", "CAT-007", "CAT-021"),
    "BR": ("CAT-005", "CAT-014", "CAT-028"),
    "JP": ("CAT-002", "CAT-015", "CAT-029", "CAT-034"),
}


def scope_for(market: str):
    """Categories in scope for one market — the deterministic narrowing."""
    return [BY_ID[c] for c in MARKET_SCOPE[market]]


def all_categories():
    return list(CATEGORIES)


def rules_block(categories) -> str:
    return "\n\n".join(c.rules_text() for c in categories)


if __name__ == "__main__":
    print(f"categories       {len(CATEGORIES)}")
    print(f"markets          {len(MARKET_SCOPE)}")
    sizes = [len(MARKET_SCOPE[m]) for m in MARKET_SCOPE]
    print(f"scope per market {min(sizes)}-{max(sizes)} categories")
    print(f"full block       {len(rules_block(CATEGORIES)):,} characters")
    print(f"PL block         {len(rules_block(scope_for('PL'))):,} characters")
