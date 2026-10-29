"""M0: how many DWARF source variables can be matched to a Fission variable?

Inputs: output of dwarf_vars.py for the unstripped build, and `fission_cli decomp --all --json`
of the same build with debug info stripped. Functions are matched by start address.

Alignment rules (positional only; nothing here checks that the data flow agrees):
  param  i      -> Fission `arg` variable with arg_index == i
  stack  slot   -> Fission `stack` variable with stack_offset == slot (SP-relative slots only)
  reg    N      -> Fission register variable named xN / wN
A DWARF variable is `unique` when exactly one Fission variable is a candidate and no other DWARF
variable of the function claims it; `shared` when several DWARF variables claim the same Fission
variable (the label would be ambiguous); `none` when no candidate exists.
As a sanity check, aligned pairs are compared by coarse type class (pointer / integer / other).
"""
from __future__ import annotations

import collections
import json
import re
import sys

REGNAME = re.compile(r"^[xw](\d+)$")


def type_class(t: str | None) -> str:
    if not t:
        return "?"
    if "*" in t or t.endswith("]"):
        return "ptr"
    if re.search(r"float|double", t):
        return "fp"
    if re.search(r"int|long|short|char|bool|size_t|uint|u?int\d+_t|unsigned|signed|enum", t):
        return "int"
    return "other"


def base_reg(frame_base: str | None) -> int | None:
    m = re.search(r"DW_OP_reg(\d+)", frame_base or "")
    return int(m.group(1)) if m else None


def candidates(var: dict, fvars: list[dict], fbase: int | None, param_index: int | None) -> tuple[str, list[dict]]:
    loc = var.get("location") or {"kind": "none"}
    kind = loc["kind"]
    if var["tag"] == "param" and param_index is not None:
        got = [v for v in fvars if v["kind"] == "arg" and v.get("arg_index") == param_index]
        if got:
            return "param-index", got
    if kind in ("fbreg", "breg"):
        sp_relative = (kind == "fbreg" and fbase == 31) or (kind == "breg" and loc.get("reg") == 31)
        got = [v for v in fvars if v["kind"] == "stack" and v.get("stack_offset") == loc["offset"]]
        return ("stack-sp" if sp_relative else "stack-fp"), got
    if kind == "reg":
        got = [v for v in fvars if v["kind"] == "reg" and (m := REGNAME.match(v["name"]))
               and int(m.group(1)) in loc["regs"]]
        return "register", got
    return kind, []


def run(dwarf_path: str, fission_path: str, label: str) -> dict:
    dwarf = json.load(open(dwarf_path))
    fission = {int(r["address"], 16): r for r in json.load(open(fission_path)) if "code" in r}
    stats = collections.Counter()
    by_method = collections.defaultdict(collections.Counter)
    type_agree = collections.Counter()
    examples = collections.defaultdict(list)
    for f in dwarf:
        row = fission.get(f["low_pc"])
        if row is None:
            stats["function_missing_in_fission"] += 1
            stats["vars_in_missing_functions"] += len(f["variables"])
            continue
        stats["functions_matched"] += 1
        fvars = row.get("variables", [])
        fbase = base_reg(f["frame_base"])
        claims: dict[str, list[str]] = collections.defaultdict(list)
        results = []
        pidx = 0
        for v in f["variables"]:
            idx = None
            if v["tag"] == "param":
                idx, pidx = pidx, pidx + 1
            method, cands = candidates(v, fvars, fbase, idx)
            results.append((v, method, cands))
            for c in cands[:1] if method == "param-index" else cands:
                claims[c["name"]].append(v["name"])
        for v, method, cands in results:
            stats["dwarf_vars"] += 1
            if not cands:
                outcome = "none"
            elif len(cands) == 1 and len(claims[cands[0]["name"]]) == 1:
                outcome = "unique"
            elif all(len(claims[c["name"]]) == 1 for c in cands) and len(cands) > 1:
                outcome = "multi-candidate"
            else:
                outcome = "shared"
            stats[outcome] += 1
            by_method[method][outcome] += 1
            if outcome == "unique":
                tc_d, tc_f = type_class(v.get("type")), type_class(cands[0].get("type"))
                type_agree["agree" if tc_d == tc_f else "disagree"] += 1
                if tc_d != tc_f and len(examples["type_disagree"]) < 6:
                    examples["type_disagree"].append((f["name"], v["name"], v.get("type"), cands[0]["name"], cands[0].get("type")))
            elif len(examples[outcome]) < 6:
                examples[outcome].append((f["name"], v["name"], method, [c["name"] for c in cands]))
    return {"label": label, "stats": dict(stats), "by_method": {k: dict(v) for k, v in by_method.items()},
            "type_agreement_on_unique": dict(type_agree), "examples": dict(examples)}


def main():
    out = [run(f"dwarf_{o}.json", f"fission_{o}.json", o) for o in sys.argv[1:]]
    json.dump(out, sys.stdout, indent=1, default=list)


if __name__ == "__main__":
    main()
