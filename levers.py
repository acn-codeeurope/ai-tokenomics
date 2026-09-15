"""Two of the eleven principles, as code you can run rather than advice.

  * Principle 5, divide and conquer — route the cheap model first, escalate only
    when it visibly fails.
  * Principle 10, avoid uncontrolled loops — a hard spending ceiling that stops
    the run. None of the three agent SDKs surveyed for this talk (ADK,
    LangGraph, OpenAI Agents) has a budget primitive: all three count tokens,
    none of them enforces money. This is twenty lines and it is the difference
    between a bug that costs $500 and one that costs $5,000.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import feedback
import measure
import prompts
import rulebook
import runner


class BudgetExceeded(RuntimeError):
    """Raised the moment a run would cross its ceiling. Deliberately loud."""


@dataclass
class Budget:
    """A spending ceiling that actually stops things.

    Counts what has been spent from real usage, and refuses the next call when
    the ceiling is reached. A counter that only reports is a dashboard; this is
    a brake.
    """

    usd_limit: float
    price_in: float = measure.PRICE_IN
    price_out: float = measure.PRICE_OUT
    tokens_in: int = 0
    tokens_out: int = 0
    calls: int = 0
    stopped_at: str | None = None

    @property
    def spent(self) -> float:
        return (self.tokens_in * self.price_in + self.tokens_out * self.price_out) / 1e6

    @property
    def remaining(self) -> float:
        return max(0.0, self.usd_limit - self.spent)

    def check(self, where: str = "") -> None:
        """Call before spending, not after. After is an incident report.

        Honest limitation: this stops the *next* call once the ceiling has been
        passed. It cannot un-spend the call that crossed it, so a run can
        overshoot by up to one call's cost. Size the ceiling accordingly — a
        limit is not the same thing as a guarantee.
        """
        if self.spent >= self.usd_limit:
            self.stopped_at = where
            raise BudgetExceeded(
                f"stopped at {where}: ${self.spent:.4f} spent of ${self.usd_limit:.4f} "
                f"after {self.calls} calls"
            )

    def charge(self, usage: runner.Usage) -> None:
        self.tokens_in += usage.tokens_in
        self.tokens_out += usage.tokens_out
        self.calls += 1


# --- principle 5: route cheap first, escalate on a visible failure -----------

def needs_escalation(ids: tuple[str, ...], scope: int, max_ids: int = 2) -> str | None:
    """Why this answer should be retried on a stronger model, or None.

    Two signals, both cheap and both measured on the models in this notebook:
    answering nothing, and over-producing. `qwen3:8b` returns 2.7 categories per
    item where 1.3 are correct — over-production is its characteristic failure,
    and it is visible without knowing the right answer.
    """
    if not ids:
        return "returned nothing"
    if len(ids) > max_ids:
        return f"returned {len(ids)} categories"
    if len(ids) > scope:
        return "returned more categories than the market has"
    return None


@dataclass
class RouteResult:
    ids: tuple[str, ...]
    model: str
    escalated: bool
    reason: str | None = None


def classify_routed(item, cheap: str, strong: str, budget: Budget | None = None,
                    max_ids: int = 2, think: bool | None = False) -> RouteResult:
    """One item: cheap model first, strong model only if the cheap one visibly failed."""
    scope = len(rulebook.scope_for(item.market))
    _, prompt = prompts.attempt1_calls(item, rulebook.scope_for)[0]

    if budget:
        budget.check(f"{item.iid} (cheap)")
    u = runner.generate(prompt, model=cheap, think=think)
    if budget:
        budget.charge(u)
    ids = runner.parse_ids(u.text)

    why = needs_escalation(ids, scope, max_ids)
    if why is None:
        return RouteResult(ids=ids, model=cheap, escalated=False)

    if budget:
        budget.check(f"{item.iid} (escalation)")
    u2 = runner.generate(prompt, model=strong, think=think,
                         options={"num_predict": 1024})
    if budget:
        budget.charge(u2)
    return RouteResult(ids=runner.parse_ids(u2.text), model=strong,
                       escalated=True, reason=why)


def run_routed(items, cheap: str, strong: str, usd_limit: float | None = None,
               max_ids: int = 2, progress: bool = True) -> dict:
    """The mixture, with cost and accuracy, against a ceiling that can stop it."""
    budget = Budget(usd_limit) if usd_limit else None
    results, scores = [], []
    stopped = None

    for item in items:
        try:
            r = classify_routed(item, cheap, strong, budget, max_ids)
        except BudgetExceeded as e:
            stopped = str(e)
            break
        results.append(r)
        scores.append(runner.score(r.ids, item.truth))
        if progress:
            mark = f"-> {strong} ({r.reason})" if r.escalated else ""
            print(f"  {item.iid} {item.market}  {r.ids}  {mark}", flush=True)

    n = len(scores) or 1
    return {
        "items": len(results),
        "escalated": sum(r.escalated for r in results),
        "precision": sum(s["precision"] for s in scores) / n,
        "recall": sum(s["recall"] for s in scores) / n,
        "exact": sum(s["exact"] for s in scores) / n,
        "tokens_in": budget.tokens_in if budget else 0,
        "tokens_out": budget.tokens_out if budget else 0,
        "usd": budget.spent if budget else 0.0,
        "stopped": stopped,
    }


if __name__ == "__main__":
    import sys

    cheap = sys.argv[1] if len(sys.argv) > 1 else "qwen3:8b"
    strong = sys.argv[2] if len(sys.argv) > 2 else "deepseek-r1:8b"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    limit = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    if not runner.available():
        raise SystemExit("no Ollama on localhost:11434")
    print(f"routing {n} items: {cheap} first, {strong} on failure, ceiling ${limit}\n")
    out = run_routed(feedback.ITEMS[:n], cheap, strong, usd_limit=limit)
    print(f"\n  items      : {out['items']}  ({out['escalated']} escalated)")
    print(f"  precision  : {out['precision']:.0%}   exact: {out['exact']:.0%}")
    print(f"  tokens     : {out['tokens_in']:,} in / {out['tokens_out']:,} out")
    print(f"  spent      : ${out['usd']:.4f}")
    if out["stopped"]:
        print(f"  STOPPED    : {out['stopped']}")
