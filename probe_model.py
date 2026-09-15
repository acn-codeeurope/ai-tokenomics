"""Three questions about a model, before spending 5 minutes benchmarking it.

  1. Is it a reasoning model that will eat the whole output budget?
  2. Does the incident prompt fit, or does it come back truncated?
  3. How far is its tokenizer from the tiktoken estimate the cost model uses?

Cheap: four calls. Run it before `bench.py`, not after.
"""

from __future__ import annotations

import json
import sys
import urllib.request

import feedback
import measure
import prompts
import rulebook
import runner


def raw(model: str, prompt: str, npred: int, think=None) -> dict:
    body = {"model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0, "seed": 7, "num_predict": npred}}
    if think is not None:
        body["think"] = think
    req = urllib.request.Request("http://localhost:11434/api/generate",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


def probe(model: str) -> dict:
    out = {"model": model, "ctx": runner.model_context(model)}

    # 1. reasoning check — a short prompt with a small budget
    d = raw(model, "Reply with exactly: CAT-001", 64)
    out["reasons"] = bool(d.get("eval_count")) and not d.get("response", "").strip()
    out["short_resp"] = (d.get("response") or "")[:40]
    out["short_done"] = d.get("done_reason")
    if out["reasons"]:
        d2 = raw(model, "Reply with exactly: CAT-001", 64, think=False)
        out["nothink_resp"] = (d2.get("response") or "")[:40]
        out["nothink_done"] = d2.get("done_reason")

    # 2 + 3. the incident prompt: does it fit, and how does it count?
    item = feedback.ITEMS[0]
    _, full = prompts.incident_calls(item, rulebook.scope_for)[0]
    est = measure.count(full, "o200k_base")
    d3 = raw(model, full, 16, think=False if out["reasons"] else None)
    tin = d3.get("prompt_eval_count", 0)
    out.update(est=est, real=tin, ratio=tin / est if est else 0,
               truncated=bool(tin) and tin < est * 0.9,
               cached=d3.get("prompt_eval_cached_count"))
    return out


if __name__ == "__main__":
    models = sys.argv[1:] or [runner.DEFAULT_MODEL]
    if not runner.available():
        raise SystemExit("no Ollama on localhost:11434")
    for m in models:
        try:
            p = probe(m)
        except Exception as e:
            print(f"{m}: FAILED {type(e).__name__}: {str(e)[:120]}")
            continue
        print(f"\n{p['model']}  (declared ctx {p['ctx']:,})")
        print(f"  reasoning model      : {p['reasons']}"
              + (f"  -> with think=false: {p.get('nothink_resp')!r} "
                 f"({p.get('nothink_done')})" if p["reasons"] else
                 f"  answer {p['short_resp']!r} ({p['short_done']})"))
        print(f"  incident prompt      : est {p['est']:,} -> real {p['real']:,} "
              f"(ratio {p['ratio']:.3f})")
        print(f"  truncated            : {p['truncated']}"
              + (f"   cached {p['cached']:,}" if p.get("cached") else ""))
