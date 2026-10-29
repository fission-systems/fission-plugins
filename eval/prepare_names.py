"""Build the variable-name recovery eval set from extracted corpus records.

One item per function; its targets are the Fission variables whose DWARF counterpart is `unique`
(see corpus/README.md). The gold label is the source variable name, taken from DWARF, not from
any decompiler. Only arm64 is used: its alignment rules were checked by hand (m0/README.md),
the x86-64 ones were not.

Few-shot examples for a project come from the *other* project, so the prompt never contains
code from the project being scored.

Also writes simple baselines (no model): for each target, the most frequent gold name among the
other project's targets with the same Fission type string (falling back to the overall most
frequent name).

usage: prepare_names.py <matrix-dir> <out-dir (must not exist)> [--per-build N]
"""
from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path

MAX_CODE_CHARS = 2500
MAX_ASM = 120
MAX_TARGETS = 4
BUILDS = [("capstone", "O0"), ("capstone", "O1"), ("isl", "O0"), ("isl", "O1")]


def asm_text(insns: list[dict]) -> str:
    lines = [i["instruction"] for i in insns[:MAX_ASM]]
    if len(insns) > MAX_ASM:
        lines.append("...")
    return "\n".join(lines)


def targets_of(r: dict) -> list[dict]:
    code = r["views"]["pseudo"]["code"]
    out = []
    for v in r["variables"]:
        if v["outcome"] != "unique" or v["tag"] not in ("param", "local") or not v["name"]:
            continue
        f = v["fission"][0]
        if f["name"] == v["name"] or not re.search(rf"\b{re.escape(f['name'])}\b", code):
            continue
        out.append({"fission": f["name"], "kind": f["kind"], "type": f.get("type"), "gold": v["name"]})
    # params come first in DWARF order; keep locals in the sample instead of letting params fill the cap
    params = [t for t in out if t["kind"] == "arg"]
    others = [t for t in out if t["kind"] != "arg"]
    return params[: MAX_TARGETS // 2] + others[: MAX_TARGETS // 2] if others else params[:MAX_TARGETS]


def load(matrix: Path, project: str, opt: str) -> list[dict]:
    items = []
    for line in open(matrix / f"ex-{project}-arm64-{opt}" / "records.jsonl"):
        r = json.loads(line)
        code = r["views"]["pseudo"]["code"]
        t = targets_of(r)
        if t and r["views"]["asm_ok"] and len(code) <= MAX_CODE_CHARS:
            items.append({"id": r["id"], "project": project, "opt": opt, "name": r["name"], "pseudo": code,
                          "asm": asm_text(r["views"]["asm"]), "targets": t})
    return items


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    per_build = 100
    if "--per-build" in sys.argv:
        per_build = int(sys.argv[sys.argv.index("--per-build") + 1])
        args.remove(str(per_build))
    matrix, out = Path(args[0]), Path(args[1])
    out.mkdir(parents=True)
    rng = random.Random(0)
    pool = {b: load(matrix, *b) for b in BUILDS}
    projects = sorted({p for p, _ in BUILDS})
    stats = {}
    for project in projects:
        other = [p for p in projects if p != project][0]
        others = [i for (p, _), items in pool.items() if p == other for i in items]
        by_type, overall = collections.defaultdict(collections.Counter), collections.Counter()
        for it in others:
            for t in it["targets"]:
                by_type[t["type"]][t["gold"]] += 1
                overall[t["gold"]] += 1
        shots = sorted((i for i in pool[(other, "O0")] if len(i["targets"]) >= 2 and len(i["pseudo"]) < 500
                        and len(i["asm"]) < 900), key=lambda i: i["id"])
        shots = rng.sample(shots, 3)
        eval_items = []
        for (p, opt), items in pool.items():
            if p != project:
                continue
            with_locals = [i for i in items if any(t["kind"] != "arg" for t in i["targets"])]
            only_params = [i for i in items if i not in with_locals]
            half = per_build * 6 // 10  # most functions have only parameters; over-sample those with locals
            sample = rng.sample(with_locals, min(half, len(with_locals)))
            sample += rng.sample(only_params, min(per_build - len(sample), len(only_params)))
            for it in sample:
                for t in it["targets"]:
                    c = by_type.get(t["type"]) or overall
                    t["baseline_type"] = c.most_common(1)[0][0]
                eval_items.append(it)
        stats[project] = {"eval_functions": len(eval_items),
                          "eval_targets": sum(len(i["targets"]) for i in eval_items),
                          "shot_ids": [s["id"] for s in shots]}
        with open(out / f"{project}.jsonl", "w") as fh:
            for it in eval_items:
                fh.write(json.dumps(it) + "\n")
        (out / f"{project}.shots.json").write_text(json.dumps(shots, indent=1))
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
