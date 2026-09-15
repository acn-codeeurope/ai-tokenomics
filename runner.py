"""Real token counters from a local model, instead of a tokenizer estimate.

Every other module here counts input with tiktoken and *assumes* the output.
That is an estimate on the way in and a guess on the way out. Ollama answers
with `prompt_eval_count` and `eval_count` — the model's own counters — so a run
through here is measured on both sides, with no API key and no billing account.

What this CANNOT measure: cache economics. Ollama does not bill, so there is no
cache-read counter to read. Cache savings stay a price-model calculation, only
now it is applied to real token counts instead of guessed ones. Say it that way.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:12b"

# Deterministic by default: a talk demo that answers differently on the second
# run is a talk demo that argues with you on stage.
DEFAULT_OPTIONS = {"temperature": 0.0, "seed": 7, "num_predict": 256}


@dataclass(frozen=True)
class Usage:
    """One call, as counted by whoever did the counting.

    `measured` is the honesty flag: True means the model reported these numbers,
    False means tiktoken guessed them. Never mix the two in one total without
    saying which is which.
    """

    tokens_in: int
    tokens_out: int
    model: str
    ms: float
    measured: bool
    text: str = ""
    truncated: bool = False
    ctx_limit: int = 0
    thinking_starved: bool = False
    telemetry_missing: bool = False

    @property
    def total(self) -> int:
        return self.tokens_in + self.tokens_out


def _post(path: str, payload: dict, host: str, timeout: float) -> dict:
    req = urllib.request.Request(
        f"{host}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def available(host: str = DEFAULT_HOST, timeout: float = 2.0) -> bool:
    """True if an Ollama server answers on `host`."""
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=timeout).read()
        return True
    except Exception:
        return False


def models(host: str = DEFAULT_HOST, timeout: float = 10.0) -> list[str]:
    """Model names the server can serve right now, without pulling anything."""
    with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return sorted(m["name"] for m in data.get("models", []))


_ctx_cache: dict[str, int] = {}


def model_context(model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> int:
    """The context window the model itself declares, in tokens.

    Needed because a prompt longer than this is not rejected — it is silently
    cut, and `prompt_eval_count` then reports the ceiling as if it were a
    measurement. That is how a bloated prompt hides.
    """
    if model in _ctx_cache:
        return _ctx_cache[model]
    data = _post("/api/show", {"model": model}, host, 30.0)
    info = data.get("model_info", {})
    limit = next(
        (int(v) for k, v in info.items() if k.endswith(".context_length")),
        0,
    )
    _ctx_cache[model] = limit
    return limit


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: float = 600.0,
    options: dict | None = None,
    expect_tokens: int = 0,
    think: bool | None = None,
) -> Usage:
    """One non-streaming completion, returning the model's own token counts.

    `stream: False` is load-bearing — streamed responses carry the counters only
    on the final chunk, and the reference notebooks throw that chunk away.

    `expect_tokens` is an independent estimate of the prompt length (a tokenizer
    count). Supplying it is the only reliable way to notice truncation: a count
    that comes back far below the estimate means input was dropped.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {**DEFAULT_OPTIONS, **(options or {})},
    }
    if think is not None:
        payload["think"] = think

    t0 = time.perf_counter()
    data = _post("/api/generate", payload, host, timeout)

    # A reasoning model with no stop token thinks until the output budget is
    # gone and hands back an empty answer — measured on gemma4:12b, which
    # consumed 4,096 tokens this way and returned nothing. The tokens are still
    # billed, at output rates. Retry once with reasoning off rather than record
    # a silent zero, and say so.
    starved = False
    if (think is None and not data.get("response")
            and data.get("done_reason") == "length" and data.get("eval_count")):
        starved = True
        payload["think"] = False
        data = _post("/api/generate", payload, host, timeout)
    ms = (time.perf_counter() - t0) * 1000

    # Ollama occasionally answers with no telemetry at all — observed on
    # gemma4:12b: four keys, a whitespace-only response, no counters. Crashing a
    # 20-minute sweep over one of these is worse than recording it, but silently
    # calling it zero tokens would corrupt the measurement. Flag it, count it,
    # and let the caller decide whether the run is still usable.
    missing = [k for k in ("prompt_eval_count", "eval_count") if k not in data]
    tokens_in = data.get("prompt_eval_count", 0)
    ceiling = model_context(model, host)

    # Two independent signals, because neither alone is trustworthy:
    #
    #  * landing exactly on the model's declared ceiling — measured behaviour:
    #    a prompt past the ceiling comes back pinned to it, with no error field.
    #  * coming back far under an independent tokenizer estimate.
    #
    # Note what is NOT a signal: exceeding a requested `num_ctx`. Ollama grows
    # the window to fit the prompt, so `num_ctx=512` on a 1,173-token prompt
    # still evaluates all 1,173. Verified 13 Sept 2026 — an earlier version of
    # this check treated that as truncation and was wrong.
    hit_ceiling = bool(ceiling) and tokens_in == ceiling
    under_estimate = bool(expect_tokens) and tokens_in < expect_tokens * 0.9

    return Usage(
        tokens_in=tokens_in,
        tokens_out=data.get("eval_count", 0),
        model=model,
        ms=ms,
        measured=True,
        text=data.get("response", ""),
        truncated=hit_ceiling or under_estimate,
        ctx_limit=ceiling,
        thinking_starved=starved,
        telemetry_missing=bool(missing),
    )


def estimate(prompt: str, expected_out: str = "", encoding: str = "o200k_base") -> Usage:
    """Tokenizer-only fallback, so the notebook still runs with no model present.

    Marked `measured=False` on purpose: this is the number the rest of the
    folder produces, and it is an estimate.
    """
    import measure

    return Usage(
        tokens_in=measure.count(prompt, encoding),
        tokens_out=measure.count(expected_out, encoding) if expected_out else 0,
        model=f"tiktoken/{encoding}",
        ms=0.0,
        measured=False,
        text=expected_out,
    )


CAT_ID = re.compile(r"CAT-\d{3}")


BARE_ID = re.compile(r'"(\d{3})"')
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_reasoning(text: str) -> str:
    """Drop an in-band <think> block before anything reads the answer.

    `deepseek-r1:8b` writes its reasoning into the response itself rather than
    into Ollama's separate `thinking` field, and that reasoning names every
    category it considered and rejected. A parser reading the whole response
    therefore harvests the model's scratchpad: measured on one item, four ids
    scraped where the model's actual answer was one, and correct. The model was
    right; the pipeline was wrong; nothing errored. Strip it first.
    """
    if not text:
        return ""
    cut = THINK_BLOCK.sub(" ", text)
    # an unterminated block (output budget ran out mid-thought) leaves an
    # opening tag and no closing one — everything after it is scratchpad too
    if "<think>" in cut.lower():
        cut = cut[: cut.lower().index("<think>")]
    return cut


def parse_ids(text: str, lenient: bool = True) -> tuple[str, ...]:
    """Category ids out of whatever the model actually said.

    Forgiving about wrapping — prose, JSON, code fences, bullet lists.

    `lenient` also accepts a bare three-digit id in a quoted list, because
    `deepseek-r1:8b` answers `["001"]` where the prompt asked for `CAT-001`.
    That is the right category and the wrong contract: substantively correct,
    and it would break any caller that expects the documented id. Scoring it as
    a miss overstates the error; scoring it silently as a hit hides a real
    integration failure. Parse it, and use `id_format_ok()` to report the gap.
    """
    body = strip_reasoning(text)
    strict = tuple(sorted(set(CAT_ID.findall(body))))
    if strict or not lenient:
        return strict
    return tuple(sorted({f"CAT-{n}" for n in BARE_ID.findall(body)}))


def id_format_ok(text: str) -> bool:
    """True when the model used the documented `CAT-nnn` form it was asked for."""
    return bool(CAT_ID.search(strip_reasoning(text)))


def score(predicted: tuple[str, ...], truth: tuple[str, ...]) -> dict:
    """Set comparison against ground truth. No partial credit for near-misses.

    The empty-prediction case is the one that matters and the one that is easy
    to get wrong. `tp/(tp+fp)` is undefined when the model returned nothing, and
    defaulting it to 1.0 hands a perfect precision score to a model that failed
    to answer — which is exactly how a broken model once looked like the most
    accurate one in this harness. Returning nothing scores 0 unless the truth
    was also empty, which is a real correct answer here: some feedback items
    genuinely have no category.
    """
    p, t = set(predicted), set(truth)
    tp, fp, fn = len(p & t), len(p - t), len(t - p)
    empty_is_right = not p and not t
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "exact": p == t,
        "precision": tp / (tp + fp) if (tp + fp) else (1.0 if empty_is_right else 0.0),
        "recall": tp / (tp + fn) if (tp + fn) else (1.0 if empty_is_right else 0.0),
        "answered": bool(p),
    }


if __name__ == "__main__":
    if not available():
        raise SystemExit("no Ollama on localhost:11434 — start `ollama serve` first")
    print("models:", ", ".join(models()) or "(none pulled)")
    print(f"context window of {DEFAULT_MODEL}: {model_context():,} tokens")
    u = generate("Reply with exactly: CAT-001, CAT-007", model=DEFAULT_MODEL)
    print(f"model {u.model}  in {u.tokens_in}  out {u.tokens_out}  {u.ms:.0f} ms  measured={u.measured}")
    print("parsed:", parse_ids(u.text))
    print("score vs (CAT-001, CAT-007):", score(parse_ids(u.text), ("CAT-001", "CAT-007")))
