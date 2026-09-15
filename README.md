# AI Tokenomics — measuring a prompt instead of guessing at it

Most cost notebooks estimate tokens with a tokenizer and then *assume* what the
model will answer. This one asks the model, reads its own counters, and shows
you how far apart those two answers are.

It runs on a **free Colab runtime — no API key, no billing account** — by
serving a local model through [Ollama](https://ollama.com).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acn-codeeurope/ai-tokenomics/blob/main/tokenomics_colab.ipynb?hl=en)

Companion code for the talk *AI Tokenomics*, Code Europe, Warsaw,
15 September 2026.

---

## What is measured, and what is calculated

The distinction is the point of the repository, so it is stated before anything
else and repeated in the cell that produces each figure.

| claim | how it is established |
|---|---|
| input tokens per call | **measured** — the model reports `prompt_eval_count` |
| output tokens per call | **measured** — the model reports `eval_count` |
| accuracy against ground truth | **measured** — parsed ids compared to a hand-labelled set |
| prompt truncation | **measured** — the counter stops at the context ceiling |
| dollar cost | *calculated* — measured tokens × published list prices |
| cache savings | *calculated* — a local model does not bill, so there is no cache counter to read |

The last two rows are arithmetic on top of real token counts, not observations.

---

## Quick start

### On Colab (recommended — this is what it was built for)

1. Click the badge above, or open this link:
   `https://colab.research.google.com/github/acn-codeeurope/ai-tokenomics/blob/main/tokenomics_colab.ipynb`
   It opens the notebook straight from this repository — nothing to download,
   nothing to upload.
2. **Runtime → Change runtime type → T4 GPU.** The notebook runs on CPU, but a
   12B model on CPU is slow enough to be unpleasant. A T4 is enough: the largest
   model here needs about 15.5 GB resident and a T4 has 16 GB, which is why the
   notebook unloads each model before loading the next.
3. Run all. The first cell installs Ollama and pulls the model; expect a few
   minutes before anything interesting happens.

### Locally

```bash
git clone https://github.com/acn-codeeurope/ai-tokenomics.git
cd ai-tokenomics
pip install -r requirements.txt
# Ollama must be installed and running: https://ollama.com/download
ollama pull gemma3:12b
python3 run_live.py 2          # two items through all three architectures
python3 runner.py              # smoke test: is the model there, does it answer
```

Modules are plain Python and import each other directly, so run them from
inside this directory.

---

## Layout

| file | what it is |
|---|---|
| `tokenomics_colab.ipynb` | **the notebook** — open this in Colab |
| `feedback.py` | the synthetic workload — items to classify |
| `rulebook.py` | the taxonomy and the per-market scoping rules |
| `prompts.py` | the three prompt architectures under test |
| `measure.py` | token counting; the tokenizer-side estimate |
| `runner.py` | calls the model, reads its counters, scores against ground truth |
| `cost_model.py` | list prices → dollars, from measured token counts |
| `crossover.py` | where one architecture stops being cheaper than another |
| `levers.py` | two of Google's eleven principles as runnable code: a router and a budget |
| `graph_app.py` | the same workload as a LangGraph graph |
| `bench.py` | machine-to-machine comparison, with a version-mismatch guard |
| `probe_model.py` | what the model actually advertises: context ceiling, build |
| `run_live.py` | one end-to-end pass |
| `cache_report.py` | standalone CLI: distinct prefixes per run, and how often each repeats |
| `notebook-modules.zip` | the modules above, for Colab when a clone is not possible — upload it to `/content/` and the notebook finds it |
| `requirements.txt` | pinned versions the published results were produced with |
| `LICENSE`, `NOTICE` | Apache-2.0 and the attribution it requires |

### The measured runs

One of these is read by the notebook; the rest ship as evidence so you can
compare your own run against them.

| file | what it is |
|---|---|
| `results-macbook-ollama034.json` | **the baseline the notebook loads** if it is present — the only data file anything reads |
| `results-macbook-gemma3-12b.json` | MacBook M4 Pro, gemma3:12b |
| `results-macbook-gemma4-12b.json` | MacBook M4 Pro, gemma4:12b |
| `results-macbook-qwen3-8b.json` | MacBook M4 Pro, qwen3:8b |
| `results-macbook-deepseek-r1-8b.json` | MacBook M4 Pro, deepseek-r1:8b |
| `bench-colab-tesla-t4-gemma3-12b.json` | Colab T4, gemma3:12b |
| `bench-colab-tesla-t4-gemma4-12b.json` | Colab T4, gemma4:12b |
| `bench-colab-tesla-t4-qwen3-8b.json` | Colab T4, qwen3:8b |
| `bench-colab-tesla-t4-deepseek-r1-8b.json` | Colab T4, deepseek-r1:8b |
| `executed-tokenomics_colab-t4.ipynb` | a full run with its outputs, T4 |

`cache_report.py` is a CLI tool and is not imported by the notebook — run it
directly.

Prices are stated and sourced in `cost_model.py` itself: vendor-neutral figures
derived from three current workhorse models, verified against the vendors' own
pricing pages on 12 September 2026.

### Dependencies

`tiktoken`, `langchain-ollama`, `langgraph`, `grandalf`, and Ollama itself.
Pinned in `requirements.txt` to the versions the published results were
produced with. The Colab notebook installs them for you.

---

## This is demonstration code, not production code

Everything here exists to make a cost argument visible and reproducible. It is
**not** written to be deployed.

* No error handling worth the name, no retry policy, no rate limiting, no
  authentication, no input validation, no secrets management.
* The classification prompts are built to be *measurable*, not to be good.
  Accuracy tops out around 73% — a property of this setup, not a recommendation.
* **The taxonomy, the feedback corpus and the ground-truth labels are
  synthetic.** They are modelled on a real incident; they are not its data, and
  no client is identifiable from anything in this repository.
* Prices are list prices at a point in time and will go stale.

Read it as *"here is one way this could be implemented, and here is what it
costs"*, not as something to lift into a system.

---

## Reproducing the numbers

Every figure in the notebook comes from a run you can repeat, and the result
files from the runs behind it ship alongside it. Two things will bite you:

* **The experiment is deterministic on one machine, not between machines.**
  `temperature=0` and a fixed seed give you the same output on the same
  hardware. Cost transfers across machines; the model's answers do not.
* **Ollama truncates silently.** It grows the context window to fit the prompt
  only while memory allows. When it does not, `prompt_eval_count` reports the
  ceiling and nothing anywhere says input was thrown away. `runner.py` carries
  an independent estimate specifically to catch this.

---

## Built on

The Colab bootstrap sequence for Ollama — the `apt` prerequisites, the
installer, `Popen(['ollama','serve'])` and the readiness loop — is adapted from
[`Legard777/eskadra-bielik-misja2`](https://github.com/Legard777/eskadra-bielik-misja2)
(Apache-2.0), itself a fork of `avedave`'s work.

The eleven principles referenced in section 11 are from
[Google Cloud's guide to AI tokenomics](https://cloud.google.com/blog/topics/developers-practitioners/guide-to-ai-tokenomics-eleven-principles-for-token-efficient-software-engineering)
(Alex "Sandu" Astrum and Luke Schlangen, 17 July 2026). Big-T notation is
Dan Neff's (Adobe), published by the Tokenomics Foundation as a working draft,
June 2026.

The eleven principle titles are reproduced verbatim and unchanged, in the
article's own order, with the source named at the point of use. The commentary
around them — what each was measured against here, and the verdict — is this
repository's own.

## Licence

[Apache License 2.0](LICENSE). Use it, change it, build on it — commercially or
not. Two conditions, both light: **keep the attribution** (the `NOTICE` file and
the copyright line), and **say what you changed** if you redistribute a modified
version.

Apache-2.0 rather than MIT for two reasons: the Colab bootstrap this borrows
from is already under it, and it carries an explicit patent grant, which matters
if anyone wants to use this inside a company.

## Citing this

```
Wasilewski, Grzegorz (2026). AI Tokenomics: measuring a prompt instead of guessing
at it. Notebook accompanying the talk at Code Europe, Warsaw,
15 September 2026. https://github.com/acn-codeeurope/ai-tokenomics
```

```bibtex
@misc{wasilewski2026tokenomics,
  author       = {Wasilewski, Grzegorz},
  title        = {AI Tokenomics: measuring a prompt instead of guessing at it},
  year         = {2026},
  month        = {9},
  howpublished = {Notebook accompanying the talk at Code Europe, Warsaw},
  note         = {Presented 15 September 2026},
  url          = {https://github.com/acn-codeeurope/ai-tokenomics}
}
```

## Author

Grzegorz Wasilewski, Data & AI Senior Manager at Accenture —
[linkedin.com/in/legard77](https://www.linkedin.com/in/legard77)

The views and the measurements here are the author's own and do not necessarily
reflect the official policy or position of Accenture.
