"""Name accuracy of a (LoRA-tuned) model on a D0 split: generate the edit list, parse it, score each target.

For every example the model sees the same prompt it was trained on and continues greedily until EOS (or a cap of
16 + 16 tokens per target, computed from the prompt, not from the gold answer). Scored per target variable:

  exact        predicted name == source name (case-sensitive; `exact_ci` ignores case)
  f1           token F1 over the identifier's words (split on `_` and camelCase), as in eval/zeroshot_names.py
  missing      the target is absent from the parsed edit list (an unparseable completion counts all its targets missing)
  rejected     with --matrix: the applier (`apply_edits` on the original Fission record) refused the proposed rename
  confidence   summed log-probability of the generated tokens that overlap the name; used for the coverage curve
               (precision of the most confident k% of targets)

Masked examples (function name replaced with FUN_<address>) are reported separately. Val is in-project (isl only):
use it to compare models trained on the same data, not as a generalization estimate. Nothing here reads test.

usage: eval_names.py --model Qwen/Qwen3.5-2B-Base [--adapter <dir>] --data <d0 dir> --out <new dir>
                     [--split val] [--functions N] [--matrix <corpus matrix dir>] [--batch 8]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))
sys.path.insert(0, str(HERE.parent / "src"))

from finetune import function_key, load_jsonl, pick_functions  # noqa: E402
from zeroshot_names import f1 as token_f1  # noqa: E402

COVERAGES = (1.0, 0.8, 0.6, 0.4, 0.2)
_PAIR = re.compile(r'"((?:[^"\\]|\\.)*)"\s*:\s*"((?:[^"\\]|\\.)*)"')


def parse_edits(text: str) -> dict[str, str] | None:
    """The `renames` object of the first JSON value in `text`, or None if it is not a {str: str} edit list."""
    try:
        value, _ = json.JSONDecoder().raw_decode(text.lstrip())
    except ValueError:
        return None
    renames = value.get("renames") if isinstance(value, dict) else None
    if not isinstance(renames, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in renames.items()):
        return None
    return renames


def value_spans(text: str) -> dict[str, tuple[int, int]]:
    """Character span of each `"old": "new"` value in the generated text (first occurrence of each key)."""
    spans: dict[str, tuple[int, int]] = {}
    for m in _PAIR.finditer(text):
        spans.setdefault(json.loads(f'"{m.group(1)}"'), m.span(2))
    return spans


def span_logprob(span: tuple[int, int], offsets: list[tuple[int, int]], logprobs: list[float]) -> float:
    """Summed log-probability of the tokens whose characters overlap span."""
    lo, hi = span
    return sum(lp for (a, b), lp in zip(offsets, logprobs) if a < hi and b > lo and b > a)


def score_example(example: dict, pred: dict[str, str] | None, conf: dict[str, float],
                  rejected: set[str]) -> list[dict]:
    gold = json.loads(example["completion"])["renames"]
    rows = []
    for t in example["targets"]:
        p = None if pred is None else pred.get(t)
        rows.append(dict(id=example["id"], masked=bool(example["function_name_masked"]), opt=example["opt"],
                         target=t, kind="param" if t.startswith("param_") else "local", gold=gold[t], pred=p,
                         exact=p == gold[t], exact_ci=p is not None and p.lower() == gold[t].lower(),
                         f1=token_f1(p, gold[t]) if p else 0.0, missing=p is None,
                         rejected=p is not None and t in rejected, confidence=conf.get(t)))
    return rows


def coverage_curve(rows: list[dict]) -> dict[str, dict]:
    """Exact-match precision among the most confident fraction of all targets (missing ones rank last)."""
    ranked = sorted(rows, key=lambda r: (r["confidence"] is None, -(r["confidence"] or 0.0)))
    out = {}
    for c in COVERAGES:
        k = max(1, round(c * len(ranked))) if ranked else 0
        top = ranked[:k]
        out[f"{c:.1f}"] = dict(n=k, exact=sum(r["exact"] for r in top) / max(k, 1),
                               min_logprob=None if not top or top[-1]["confidence"] is None
                               else round(top[-1]["confidence"], 4))
    return out


def summarize(rows: list[dict], parse_failures: int, examples: int) -> dict:
    def agg(rs: list[dict]) -> dict:
        n = len(rs)
        if not n:
            return dict(targets=0)
        mean = lambda key: round(sum(float(r[key]) for r in rs) / n, 4)  # noqa: E731
        return dict(targets=n, exact=mean("exact"), exact_ci=mean("exact_ci"), f1=mean("f1"),
                    missing=mean("missing"), rejected=mean("rejected"))

    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        for key in ("all", "masked" if r["masked"] else "unmasked", f"opt-{r['opt']}", f"kind-{r['kind']}"):
            groups[key].append(r)
    return dict(examples=examples, parse_failures=parse_failures,
                by=dict(sorted((k, agg(v)) for k, v in groups.items())),
                coverage={k: coverage_curve(groups[k]) for k in ("all", "unmasked", "masked") if groups[k]})


def load_records(matrix: Path, ids: set[str]) -> dict[str, dict]:
    want = {i.split("#")[0] for i in ids}
    out = {}
    for path in sorted(matrix.glob("ex-*/records.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["id"] in want:
                out[row["id"]] = row
    return out


def record_for(example: dict, records: dict[str, dict]):
    from fission_plugins import FunctionRecord
    from prepare_edits import masked_code

    row = records[example["id"].split("#")[0]]
    pseudo = dict(row["views"]["pseudo"])
    name = row["name"]
    if example["function_name_masked"]:
        pseudo["code"] = masked_code(pseudo["code"], row["name"], row["address"])
        name = f"FUN_{row['address'].lower().removeprefix('0x').zfill(8)}"
    return FunctionRecord.from_fission_json({"address": row["address"], "name": name, **pseudo})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["val", "xproj"], help="<split>.jsonl + <split>-masked.jsonl")
    p.add_argument("--functions", type=int, default=None, help="first N functions in the fixed order (default: all)")
    p.add_argument("--subset-seed", type=int, default=20261110)
    p.add_argument("--matrix", type=Path, default=None, help="corpus matrix dir, to check edits with apply_edits")
    p.add_argument("--batch", type=int, default=8, help="prompts per generate call (left-padded)")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"refusing to overwrite {a.out}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    from fission_plugins import EditList, apply_edits

    device = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    examples = load_jsonl(a.data / f"{a.split}.jsonl") + load_jsonl(a.data / f"{a.split}-masked.jsonl")
    examples = pick_functions(examples, a.functions, a.subset_seed)
    records = load_records(a.matrix, {e["id"] for e in examples}) if a.matrix else {}

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype)
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model = model.to(device).eval()

    prompts = [tok(e["prompt"], add_special_tokens=False)["input_ids"] for e in examples]
    order = sorted(range(len(examples)), key=lambda i: len(prompts[i]))
    a.out.mkdir(parents=True)
    gen_log = (a.out / "generations.jsonl").open("w")
    rows, parse_failures, started = [], 0, time.time()
    for b in range(0, len(order), a.batch):
        idx = order[b : b + a.batch]
        width = max(len(prompts[i]) for i in idx)
        ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, width - len(prompts[i]) :] = torch.tensor(prompts[i])
            mask[r, width - len(prompts[i]) :] = 1
        cap = min(400, 16 + 16 * max(len(examples[i]["targets"]) for i in idx))
        config = GenerationConfig(max_new_tokens=cap, do_sample=False, eos_token_id=tok.eos_token_id,
                                  pad_token_id=pad_id)
        with torch.no_grad():
            out = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), generation_config=config,
                                 output_scores=True, return_dict_in_generate=True)
        logprobs = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True).float().cpu()
        for r, i in enumerate(idx):
            gen = out.sequences[r, width:].tolist()
            n = gen.index(tok.eos_token_id) if tok.eos_token_id in gen else len(gen)
            gen = gen[:n]
            text, offsets = "", []
            for k in range(n):  # character span of each generated token
                nxt = tok.decode(gen[: k + 1])
                offsets.append((len(text), len(nxt)))
                text = nxt
            pred = parse_edits(text)
            parse_failures += pred is None
            spans = value_spans(text)
            conf = {t: span_logprob(spans[t], offsets, logprobs[r, :n].tolist()) for t in (pred or {}) if t in spans}
            rejected = set()
            if pred and records:
                result = apply_edits(record_for(examples[i], records), EditList(renames=pred))
                rejected = {w.split("->")[0] for w, _ in result.rejected}
            rows += score_example(examples[i], pred, conf, rejected)
            gen_log.write(json.dumps(dict(id=examples[i]["id"], text=text, eos=n < len(out.sequences[r, width:]),
                                          parsed=pred is not None, rejected=sorted(rejected))) + "\n")
        gen_log.flush()
        print(f"{b + len(idx)}/{len(order)} examples {time.time() - started:.0f}s", flush=True)

    with (a.out / "targets.jsonl").open("w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    summary = summarize(rows, parse_failures, len(examples))
    summary.update(model=a.model, adapter=str(a.adapter) if a.adapter else None, split=a.split, functions=a.functions,
                   functions_evaluated=len({function_key(e) for e in examples}), device=device, batch=a.batch,
                   applier_checked=bool(records), seconds=round(time.time() - started, 1))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("examples", "parse_failures", "seconds")} | {"all": summary["by"]["all"]}))


if __name__ == "__main__":
    main()
