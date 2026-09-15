"""The 'after' architecture from the talk, as an actual LangGraph graph.

This is the picture on the architecture slide, made executable:

    START -> fetch_rules -> narrow_scope -> classify_shard (xN) -> join_ids -> persist_raw

Four of those five nodes never touch a model. That is the point of drawing it
this way: the agent logic is one node wide, and everything around it is
ordinary deterministic code you can read, test and put a limit on.

Why LangGraph and not a hand-rolled loop: it has a named primitive for each
move. `Send` is the dynamic fan-out, an `Annotated[..., operator.add]` reducer
is the join, and `add_conditional_edges` is the branch. A hand-rolled version
hides all three inside a list comprehension.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

import feedback
import prompts
import rulebook
import runner


class State(TypedDict, total=False):
    item: Any                                        # feedback.Item
    llm: Any                                         # ChatOllama; a channel, or the graph drops it
    rules: dict[str, str]
    scope: list[str]
    shards: Annotated[list[dict], operator.add]      # the reducer = the join
    usage: Annotated[list[dict], operator.add]
    ids: tuple[str, ...]
    persisted: bool


def make_llm(model: str = runner.DEFAULT_MODEL, **kw) -> ChatOllama:
    """ChatOllama reports real counters — verified, not assumed.

    `usage_metadata` carries input_tokens/output_tokens, and `response_metadata`
    keeps Ollama's raw `prompt_eval_count` / `eval_count` alongside it. Nothing
    has to be re-counted with a tokenizer.
    """
    return ChatOllama(model=model, temperature=0.0, num_predict=256, **kw)


# --- deterministic nodes -----------------------------------------------------

def fetch_rules(state: State) -> dict:
    """Rules arrive from storage, not from the prompt. No model involved.

    In the talk this is an MCP `resources/read` issued by application code.
    Resources are application-controlled by design, which is the only reason
    this node can honestly be labelled '0 LLM calls'.
    """
    return {"rules": {c.cid: c for c in rulebook.all_categories()}}


def narrow_scope(state: State) -> dict:
    """Scope comes from a field on the item, not from a judgement by the model."""
    return {"scope": [c.cid for c in rulebook.scope_for(state["item"].market)]}


def join_ids(state: State) -> dict:
    """Wait for every shard, keep unique ids. Workers answer with ids, not names."""
    seen = {i for shard in state.get("shards", []) for i in shard["ids"]}
    return {"ids": tuple(sorted(seen))}


def persist_raw(state: State) -> dict:
    """An MCP `tools/call` issued from application code — again, model-free."""
    return {"persisted": True}


# --- the one node that costs tokens -----------------------------------------

def classify_shard(payload: dict) -> dict:
    """One category, one call. This is the only place a model is invoked."""
    item, category = payload["item"], payload["category"]
    _, full = prompts.attempt2(item, category)
    msg = payload["llm"].invoke(full)
    u = msg.usage_metadata or {}
    raw = msg.response_metadata or {}
    return {
        "shards": [{"cid": category.cid, "ids": runner.parse_ids(msg.content)}],
        "usage": [{
            "cid": category.cid,
            "input_tokens": u.get("input_tokens", raw.get("prompt_eval_count", 0)),
            "output_tokens": u.get("output_tokens", raw.get("eval_count", 0)),
        }],
    }


def fan_out(state: State) -> list[Send]:
    """`Send` is the fan-out: one worker per in-scope category, decided at runtime."""
    return [
        Send("classify_shard",
             {"item": state["item"], "category": state["rules"][cid], "llm": state["llm"]})
        for cid in state["scope"]
    ]


def build(model: str = runner.DEFAULT_MODEL):
    g = StateGraph(State)
    g.add_node("fetch_rules", fetch_rules)
    g.add_node("narrow_scope", narrow_scope)
    g.add_node("classify_shard", classify_shard)
    g.add_node("join_ids", join_ids)
    g.add_node("persist_raw", persist_raw)

    g.add_edge(START, "fetch_rules")
    g.add_edge("fetch_rules", "narrow_scope")
    g.add_conditional_edges("narrow_scope", fan_out, ["classify_shard"])
    g.add_edge("classify_shard", "join_ids")
    g.add_edge("join_ids", "persist_raw")
    g.add_edge("persist_raw", END)
    return g.compile()


def classify(item, app=None, llm=None, model: str = runner.DEFAULT_MODEL) -> dict:
    """Run one item through the graph and report ids, usage and accuracy."""
    app = app or build(model)
    llm = llm or make_llm(model)
    out = app.invoke({"item": item, "llm": llm})
    usage = out.get("usage", [])
    return {
        "ids": out["ids"],
        "truth": item.truth,
        "score": runner.score(out["ids"], item.truth),
        "llm_calls": len(usage),
        "input_tokens": sum(u["input_tokens"] for u in usage),
        "output_tokens": sum(u["output_tokens"] for u in usage),
    }


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else runner.DEFAULT_MODEL
    if not runner.available():
        raise SystemExit("no Ollama on localhost:11434 — start `ollama serve` first")

    app, llm = build(model), make_llm(model)
    item = feedback.ITEMS[0]
    print(f"item {item.iid} · market {item.market} · truth {item.truth}\n")
    r = classify(item, app=app, llm=llm)
    print(f"ids           : {r['ids']}")
    print(f"llm calls     : {r['llm_calls']}  (one per in-scope category)")
    print(f"input tokens  : {r['input_tokens']:,}   output tokens: {r['output_tokens']:,}")
    print(f"precision     : {r['score']['precision']:.0%}   recall: {r['score']['recall']:.0%}")
