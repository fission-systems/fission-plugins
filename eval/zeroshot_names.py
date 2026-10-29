"""Few-shot variable-name recovery with a base model, scored against the source variable names.

Runs on a GPU (Colab) or anywhere with torch + transformers >= 5.21 (Qwen3.5 needs it).

For every target (a Fission placeholder such as `param_1`) the prompt ends with
`param_1 ->` and the model continues with one name. Targets are scored independently
(no earlier answers are shown). Two input conditions:
  pseudo       decompiled C only
  pseudo+asm   decompiled C and the function's assembly
Few-shot examples come from the other project (prepare_names.py).

Metrics: exact match (case-insensitive) and token F1 over the identifier's words
(split on `_` and camelCase). Baseline `type`: most frequent gold name for the same
Fission type in the other project (no model).

usage: zeroshot_names.py <data-dir> <out-dir> [--model ID] [--limit N] [--batch N]
"""
from __future__ import annotations

import collections
import json
import re
import sys
import time
from pathlib import Path

HEAD = ("Each task shows decompiled C code. The decompiler gave the variables placeholder names. "
        "After the code come those placeholders, each followed by the name the original source code used.\n\n")


def block(item: dict, with_asm: bool, answers: bool) -> str:
    s = f"### Decompiled code\n{item['pseudo'].strip()}\n\n"
    if with_asm:
        s += f"### Assembly\n{item['asm']}\n\n"
    s += "### Original names\n"
    if answers:
        s += "".join(f"{t['fission']} -> {t['gold']}\n" for t in item["targets"]) + "\n"
    return s


def prompt(shots: list[dict], item: dict, target: dict, with_asm: bool) -> str:
    return (HEAD + "".join(block(s, with_asm, True) for s in shots)
            + block(item, with_asm, False) + f"{target['fission']} ->")


def words(name: str) -> list[str]:
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower().split("_")
    return [p for p in parts if p]


def f1(pred: str, gold: str) -> float:
    p, g = collections.Counter(words(pred)), collections.Counter(words(gold))
    common = sum((p & g).values())
    if not common:
        return 0.0
    prec, rec = common / sum(p.values()), common / sum(g.values())
    return 2 * prec * rec / (prec + rec)


def clean(text: str) -> str:
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", text)
    return m.group(1) if m else ""


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    def opt(flag, default):
        if flag not in sys.argv:
            return default
        v = sys.argv[sys.argv.index(flag) + 1]
        args.remove(v)
        return v

    model_id = opt("--model", "Qwen/Qwen3.5-2B-Base")
    limit = opt("--limit", None)
    batch = int(opt("--batch", 4))
    data, out = Path(args[0]), Path(args[1])
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda").eval()

    rows = []
    for project in ("capstone", "isl"):
        items = [json.loads(l) for l in open(data / f"{project}.jsonl")]
        shots = json.loads((data / f"{project}.shots.json").read_text())
        if limit:
            items = items[: int(limit)]
        for it in items:
            for t in it["targets"]:
                rows.append((project, it, t, shots))
    print(f"{len(rows)} targets", flush=True)

    results = []
    t0 = time.time()
    for cond in ("pseudo", "pseudo+asm"):
        with_asm = cond == "pseudo+asm"
        prompts = [prompt(shots, it, t, with_asm) for _, it, t, shots in rows]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths per batch
        preds = [None] * len(prompts)
        skipped = 0
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
            if enc["input_ids"].shape[1] > 8192:
                skipped += len(idx)
                continue
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
            for i, g in zip(idx, gen):
                preds[i] = clean(tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True))
            if (b // batch) % 25 == 0:
                print(f"{cond} {b}/{len(order)} {time.time() - t0:.0f}s", flush=True)
        for (project, it, t, _), p in zip(rows, preds):
            results.append({"cond": cond, "project": project, "opt": it["opt"], "id": it["id"], "kind": t["kind"],
                            "fission": t["fission"], "gold": t["gold"], "pred": p, "baseline_type": t["baseline_type"]})
        print(f"{cond}: skipped {skipped} over-long prompts", flush=True)

    with open(out / "predictions.jsonl", "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    summary = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in results:
        for key in ("all", r["project"], f"{r['project']}-{r['opt']}", f"kind-{r['kind']}"):
            if r["pred"] is not None:
                summary[(key, r["cond"])]["exact"].append(r["pred"].lower() == r["gold"].lower())
                summary[(key, r["cond"])]["f1"].append(f1(r["pred"], r["gold"]))
    for r in results:
        if r["cond"] == "pseudo":
            for key in ("all", r["project"], f"{r['project']}-{r['opt']}", f"kind-{r['kind']}"):
                summary[(key, "baseline-type")]["exact"].append(r["baseline_type"].lower() == r["gold"].lower())
                summary[(key, "baseline-type")]["f1"].append(f1(r["baseline_type"], r["gold"]))
    table = {f"{k}|{c}": {"n": len(v["exact"]), "exact": sum(v["exact"]) / len(v["exact"]),
                          "f1": sum(v["f1"]) / len(v["f1"])} for (k, c), v in sorted(summary.items())}
    (out / "summary.json").write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
