"""The three architectures, run against a real model instead of a tokenizer.

`crossover.py` answers "what would this cost" from tiktoken counts and an
assumed output. This module answers "what did it actually cost" from the
model's own counters, and — more usefully — shows how far apart those two
answers are.

Two things to keep honest:

1. A local model is not the billed model. Its tokenizer is not `o200k_base`, so
   a gap between estimate and measurement is expected and is the *point*: price
   with the wrong tokenizer and you are wrong before you start.
2. This samples. A 12B model chewing 15 incident prompts of ~18k tokens takes
   minutes. Everything here reports the measured sample AND the projection, and
   never lets you confuse them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import feedback
import measure
import prompts
import rulebook
import runner

ENC = "o200k_base"

ARCHITECTURES = {
    "incident": prompts.incident_calls,   # all 50 categories in every prompt
    "scoped": prompts.attempt1_calls,     # only the categories this market uses
    "fanout": prompts.attempt2_calls,     # one call per in-scope category
}


@dataclass
class Call:
    item: str
    market: str
    est_in: int
    est_out: int
    real_in: int
    real_out: int
    ms: float
    text: str = ""
    truncated: bool = False
    no_telemetry: bool = False
    format_ok: bool = True


@dataclass
class ArchResult:
    name: str
    model: str
    items: int
    calls: list[Call] = field(default_factory=list)
    scores: list[dict] = field(default_factory=list)

    def _sum(self, attr: str) -> int:
        return sum(getattr(c, attr) for c in self.calls)

    @property
    def real_in(self) -> int:
        return self._sum("real_in")

    @property
    def real_out(self) -> int:
        return self._sum("real_out")

    @property
    def est_in(self) -> int:
        return self._sum("est_in")

    @property
    def est_out(self) -> int:
        return self._sum("est_out")

    @property
    def truncated(self) -> int:
        """Calls whose prompt was silently cut to fit the context window."""
        return sum(c.truncated for c in self.calls)

    @property
    def format_drift(self) -> int:
        """Calls that answered with ids in a form the prompt did not ask for."""
        return sum(not c.format_ok for c in self.calls)

    @property
    def no_telemetry(self) -> int:
        """Calls Ollama answered without any token counters. Not measurements."""
        return sum(c.no_telemetry for c in self.calls)

    @property
    def precision(self) -> float:
        return sum(s["precision"] for s in self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def recall(self) -> float:
        return sum(s["recall"] for s in self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def calls_per_item(self) -> float:
        return len(self.calls) / self.items if self.items else 0.0

    @property
    def in_ratio(self) -> float:
        """Measured input over estimated input. 1.0 means the estimate held."""
        return self.real_in / self.est_in if self.est_in else 0.0

    @property
    def out_ratio(self) -> float:
        """Measured output over the output the cost model assumed."""
        return self.real_out / self.est_out if self.est_out else 0.0

    @property
    def exact(self) -> float:
        """Share of items where the model's id set matched ground truth exactly."""
        return sum(s["exact"] for s in self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def seconds(self) -> float:
        return self._sum("ms") / 1000

    def usd(self, runs: int = 1, price_in: float = None, price_out: float = None) -> float:
        """Measured tokens priced at the billed model's rates, scaled to `runs`."""
        p_in = measure.PRICE_IN if price_in is None else price_in
        p_out = measure.PRICE_OUT if price_out is None else price_out
        per_run = self.real_in * p_in / 1e6 + self.real_out * p_out / 1e6
        return per_run * runs


def run_architecture(
    name: str,
    items: list[feedback.Item],
    model: str = runner.DEFAULT_MODEL,
    host: str = runner.DEFAULT_HOST,
    progress: bool = True,
    think: bool | None = None,
    num_predict: int | None = None,
) -> ArchResult:
    """Send every call of one architecture to the model, keeping both counts."""
    calls_for = ARCHITECTURES[name]
    res = ArchResult(name=name, model=model, items=len(items))

    for item in items:
        predicted: set[str] = set()
        for prefix, full in calls_for(item, rulebook.scope_for):
            est_in = measure.count(full, ENC)
            u = runner.generate(full, model=model, host=host, expect_tokens=est_in, think=think,
                                options={"num_predict": num_predict} if num_predict else None)
            # The cost model assumes the answer is the ground-truth id list; keep
            # that as `est_out` so the two are comparable call for call.
            assumed = '["' + '", "'.join(item.truth) + '"]' if item.truth else ""
            res.calls.append(
                Call(
                    item=item.iid,
                    market=item.market,
                    est_in=est_in,
                    est_out=measure.count(assumed, ENC) if assumed else 0,
                    real_in=u.tokens_in,
                    real_out=u.tokens_out,
                    ms=u.ms,
                    text=u.text,
                    truncated=u.truncated,
                    no_telemetry=u.telemetry_missing,
                    format_ok=runner.id_format_ok(u.text) if u.text.strip() else True,
                )
            )
            predicted |= set(runner.parse_ids(u.text))
        res.scores.append(runner.score(tuple(sorted(predicted)), item.truth))
        if progress:
            print(f"  {name:8s} {item.iid} {item.market}  "
                  f"calls={len(res.calls):3d}  real_in={res.real_in:,}", flush=True)

    return res


def compare(results: list[ArchResult], runs: int = 1) -> None:
    """One table, measured only. No modelled cache numbers mixed in."""
    print(f"\nmeasured on {results[0].model} · {results[0].items} items · "
          f"${measure.PRICE_IN:.2f}/1M in · ${measure.PRICE_OUT:.2f}/1M out")
    head = (f"{'arch':9s} {'calls':>6s} {'real in':>10s} {'real out':>9s} "
            f"{'est in':>10s} {'in x':>6s} {'out x':>6s} {'prec':>6s} {'rec':>6s} "
            f"{'exact':>6s} {'cut':>4s} {'sec':>7s}")
    print(head)
    print("-" * len(head))
    for r in results:
        print(f"{r.name:9s} {len(r.calls):6d} {r.real_in:10,} {r.real_out:9,} "
              f"{r.est_in:10,} {r.in_ratio:6.2f} {r.out_ratio:6.2f} "
              f"{r.precision:6.0%} {r.recall:6.0%} {r.exact:6.0%} "
              f"{r.truncated:4d} {r.seconds:7.1f}")

    cut = [r for r in results if r.truncated]
    if cut:
        print("\n  ⚠ 'cut' counts prompts that filled the context window exactly — they were")
        print("    truncated before the model saw them, so those rows measure a SHORTER")
        print("    prompt than the architecture actually sends. Not a valid cost figure.")

    base = next((r for r in results if r.name == "incident"), None)
    if base and base.real_in:
        print("\nagainst the incident architecture, on measured tokens:")
        for r in results:
            if r is base:
                continue
            print(f"  {r.name:8s} {base.usd(runs) / r.usd(runs):5.2f}x cheaper"
                  if r.usd(runs) else f"  {r.name:8s}   n/a")


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    model = sys.argv[2] if len(sys.argv) > 2 else runner.DEFAULT_MODEL

    if not runner.available():
        raise SystemExit("no Ollama on localhost:11434 — start `ollama serve` first")

    sample = feedback.ITEMS[:n]
    print(f"sampling {len(sample)} of {len(feedback.ITEMS)} items on {model}\n")
    out = [run_architecture(name, sample, model=model) for name in ARCHITECTURES]
    compare(out)
