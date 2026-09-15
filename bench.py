"""Run the benchmark and record what it ran on, so two machines can be compared.

A number without a machine attached to it is not a benchmark. Every run written
here carries its own environment block — CPU, GPU, RAM, Ollama version, model
and declared context — and reports throughput as well as totals, so a 3-item run
on one box is still comparable to a 15-item run on another.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone

import feedback
import run_live
import runner


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def _ram_gb() -> float:
    if sys.platform == "darwin":
        out = _sh(["sysctl", "-n", "hw.memsize"])
        return round(int(out) / 1024**3, 1) if out.isdigit() else 0.0
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024**2, 1)
    except Exception:
        pass
    return 0.0


def _gpu() -> str:
    """Whatever is doing the matrix multiplication, named.

    `nvidia-smi` covers Colab. On Apple Silicon there is no discrete GPU and no
    separate VRAM — Ollama runs on Metal against unified memory, which is why a
    laptop with 48 GB can hold a context that would not fit a 16 GB T4.
    """
    nvidia = _sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if nvidia:
        return nvidia
    if sys.platform == "darwin":
        chip = _sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        return f"{chip} — Metal, unified memory" if chip else "Apple Silicon — Metal, unified memory"
    return "none detected"


def environment(model: str, host: str = runner.DEFAULT_HOST) -> dict:
    """Everything needed to tell two runs apart after the fact."""
    gpu = _gpu()
    try:
        with urllib.request.urlopen(f"{host}/api/version", timeout=5) as r:
            ollama_version = json.loads(r.read())["version"]
    except Exception:
        ollama_version = "unknown"

    # Derive the label from what is actually there. An assumed label ("colab T4")
    # once mislabelled an A100-80GB run and made a bad comparison look plausible.
    #
    # NEVER platform.node(). It used to be the local fallback and it wrote a
    # managed-laptop asset tag straight into a published result file. The
    # hostname says nothing useful about a benchmark and everything about whose
    # machine it was. Describe the hardware instead — that is the part a reader
    # of the result actually needs.
    if "google.colab" in sys.modules:
        card = gpu.split(",")[0].strip().replace("NVIDIA ", "") if "," in gpu else "CPU"
        label = f"colab {card}"
    else:
        chip = gpu.split("—")[0].strip() if "—" in gpu else (platform.machine() or "local")
        label = f"local {chip}".strip()

    return {
        "label": label,
        "colab": "google.colab" in sys.modules,
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "disk_free_gb": round(__import__("shutil").disk_usage(".").free / 1024**3),
        "gpu": gpu,
        "python": platform.python_version(),
        "ollama": ollama_version,
        "model": model,
        "declared_ctx": runner.model_context(model, host),
    }


def run(n_items: int = 3, model: str = None, out_path: str = None,
        host: str = runner.DEFAULT_HOST, progress: bool = True,
        label: str = None, warmup: bool = True, think: bool | None = None,
        num_predict: int | None = None) -> dict:
    """Benchmark all three architectures and write a tagged result file."""
    model = model or runner.DEFAULT_MODEL
    env = environment(model, host)
    if label:
        env["label"] = label
    env["think"] = think
    env["num_predict"] = num_predict
    sample = feedback.ITEMS[:n_items]

    print(f"benchmark · {env['label']} · {model} · {len(sample)} items")
    print(f"  gpu: {env['gpu']}   ram: {env['ram_gb']} GB   ctx: {env['declared_ctx']:,}\n")

    # The first call to a cold model pays for loading weights into memory. On a
    # small sample that lands entirely on one architecture and makes it look
    # slow. Burn it here instead, where it belongs to nobody.
    if warmup:
        t = runner.generate("ping", model=model, host=host, options={"num_predict": 1})
        print(f"  warmup {t.ms/1000:.1f}s (model load, excluded from results)\n", flush=True)
        env["warmup_seconds"] = round(t.ms / 1000, 1)

    payload = {"env": env, "items": len(sample), "arch": {}}
    for name in run_live.ARCHITECTURES:
        r = run_live.run_architecture(name, sample, model=model, host=host,
                                      progress=progress, think=think, num_predict=num_predict)
        payload["arch"][name] = {
            "calls": len(r.calls), "real_in": r.real_in, "real_out": r.real_out,
            "est_in": r.est_in, "est_out": r.est_out, "truncated": r.truncated,
            "no_telemetry": r.no_telemetry, "format_drift": r.format_drift,
            "in_ratio": r.in_ratio, "out_ratio": r.out_ratio,
            "precision": r.precision, "recall": r.recall, "exact": r.exact,
            "seconds": r.seconds,
            # throughput is what survives a different item count
            "sec_per_call": r.seconds / len(r.calls) if r.calls else 0.0,
            "in_tokens_per_sec": r.real_in / r.seconds if r.seconds else 0.0,
        }
        print(f"  {name:9s} done · {r.seconds/60:.1f} min", flush=True)

    if out_path:
        json.dump(payload, open(out_path, "w"), indent=1)
        print(f"\nwritten to {out_path}")
    return payload


def load(path: str) -> dict:
    return json.load(open(path))


def compare(runs: list[dict]) -> None:
    """Side by side. Totals differ with item count; throughput does not."""
    # Column width from the actual labels — a hardcoded width silently ragged
    # the table the moment a label got longer than it.
    w = max(len(r["env"]["label"]) for r in runs)

    print("machines")
    for r in runs:
        e = r["env"]
        print(f"  {e['label']}")
        print(f"      {e['model']} · {r['items']} items · ollama {e.get('ollama', '?')}")
        print(f"      {e['gpu']} · {e['ram_gb']} GB RAM · {e['processor']}")

    versions = {r["env"].get("ollama", "?") for r in runs}
    if len(versions) > 1:
        print("\n" + "!" * 70)
        print("!! DIFFERENT OLLAMA VERSIONS — speed here is NOT purely hardware")
        print("!! " + "  vs  ".join(f"{r['env']['label']}: {r['env'].get('ollama','?')}" for r in runs))
        print("!! Measured 13 Sept 2026: upgrading 0.20.4 -> 0.34.0 alone gave 10.8x on")
        print("!! 14k-token prompts (sliding-window attention support) and nothing on")
        print("!! short ones. Hold the version constant before blaming the GPU.")
        print("!" * 70)

    models = {r["env"]["model"] for r in runs}
    if len(models) > 1:
        print("\n" + "!" * 70)
        print("!! DIFFERENT MODELS — THIS IS NOT A MACHINE COMPARISON")
        print("!! " + "  vs  ".join(f"{r['env']['label']}: {r['env']['model']}" for r in runs))
        print("!! Speed differences below are dominated by model size, not hardware.")
        print("!" * 70)

    print("\ntokens and accuracy — these must MATCH if it is the same experiment")
    head = (f"  {'arch':9s} {'machine':<{w}s} {'tokens in':>11s} {'tokens out':>11s} "
            f"{'prec':>6s} {'exact':>6s}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for arch in ("incident", "scoped", "fanout"):
        for r in runs:
            a = r.get("arch", {}).get(arch)
            if a:
                print(f"  {arch:9s} {r['env']['label']:<{w}s} {a['real_in']:11,} "
                      f"{a['real_out']:11,} {a['precision']:6.0%} {a['exact']:6.0%}")

    print("\nwall clock — these are EXPECTED to differ")
    for arch in ("incident", "scoped", "fanout"):
        present = [r for r in runs if arch in r.get("arch", {})]
        if not present:
            continue
        print(f"\n{arch}")
        head = (f"  {'machine':<{w}s} {'items':>5s} {'calls':>6s} {'minutes':>8s} "
                f"{'sec/call':>9s} {'in tok/s':>9s}")
        print(head)
        print("  " + "-" * (len(head) - 2))
        for r in present:
            a = r["arch"][arch]
            print(f"  {r['env']['label']:<{w}s} {r['items']:5d} {a['calls']:6d} "
                  f"{a['seconds']/60:8.1f} {a['sec_per_call']:9.1f} {a['in_tokens_per_sec']:9.1f}")

        if len(present) == 2:
            a, b = (p["arch"][arch] for p in present)
            if a["in_tokens_per_sec"] and b["in_tokens_per_sec"]:
                x = b["in_tokens_per_sec"] / a["in_tokens_per_sec"]
                faster = present[1]["env"]["label"] if x > 1 else present[0]["env"]["label"]
                print(f"  -> {faster} is {max(x, 1/x):.2f}x faster on input throughput")

    if len(models) == 1 and len(versions) == 1:
        print("\nSame model and same Ollama version. Token counts should match exactly;")
        print("only the clock should move. Where they match, the measurement is")
        print("portable and the cost conclusions hold on any machine.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    model = sys.argv[2] if len(sys.argv) > 2 else runner.DEFAULT_MODEL
    out = sys.argv[3] if len(sys.argv) > 3 else None
    label = sys.argv[4] if len(sys.argv) > 4 else None
    think = {"think": True, "nothink": False}.get(sys.argv[5]) if len(sys.argv) > 5 else None
    npred = int(sys.argv[6]) if len(sys.argv) > 6 else None
    if not runner.available():
        raise SystemExit("no Ollama on localhost:11434 — start `ollama serve` first")
    run(n, model, out, label=label, think=think, num_predict=npred)
