"""Does applying edits keep what the pseudocode compiles to? Measured on real corpus records.

For each sampled function the script (1) compiles the Fission pseudocode with `oracle.compile_assembly`,
(2) applies edits built from the DWARF source names (a perfect renamer), a function rename and comments,
and requires identical assembly, and (3) as a power check applies two unsafe mutations that bypass the
applier (merge two variables; change a literal) and counts how often the oracle notices.

usage: semantic_check.py <matrix-dir> <report.json (must not exist)> [--per-build N] [--seed S]
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fission_plugins import Comment, EditList, FunctionRecord, apply_edits  # noqa: E402
from fission_plugins.apply import _rewrite  # noqa: E402
from fission_plugins.oracle import compare, compile_assembly, find_clang  # noqa: E402

BUILDS = [f"ex-{p}-{a}-{o}" for p, a, o in
          [("capstone", "arm64", "O0"), ("capstone", "arm64", "O1"), ("isl", "arm64", "O0"),
           ("isl", "arm64", "O1"), ("isl", "x86_64", "O0"), ("isl", "x86_64", "O1")]]
_LITERAL = re.compile(r"(?<![\w.])([2-9]|[1-9]\d+)(?![\w.])")


def gold_renames(row: dict) -> dict[str, str]:
    """Fission variable -> DWARF name for uniquely aligned variables, skipping duplicates."""
    out: dict[str, str] = {}
    for v in row["variables"]:
        if v["outcome"] != "unique" or not v["name"] or v["tag"] not in ("param", "local"):
            continue
        out.setdefault(v["fission"][0]["name"], v["name"])
    return out


def merge_variables(code: str, names: list[str], rng: random.Random) -> str | None:
    if len(names) < 2:
        return None
    a, b = rng.sample(names, 2)
    # keep b's declaration (its first occurrence) and redirect the later uses to a
    first = re.search(rf"\b{re.escape(b)}\b", code)
    if not first:
        return None
    return code[: first.end()] + _rewrite(code[first.end():], {b: a})


def tweak_literal(code: str, rng: random.Random) -> str | None:
    body = code.split("{", 1)[-1]
    found = list(_LITERAL.finditer(body))
    if not found:
        return None
    m = rng.choice(found)
    start = len(code) - len(body) + m.start()
    return code[:start] + str(int(m.group(1)) + 1) + code[start + len(m.group(1)):]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("matrix", type=Path)
    p.add_argument("report", type=Path)
    p.add_argument("--per-build", type=int, default=60)
    p.add_argument("--seed", type=int, default=11)
    a = p.parse_args()
    clang = find_clang()
    if not clang:
        sys.exit("clang not found (set CLANG)")
    if a.report.exists():
        sys.exit(f"refusing to overwrite {a.report}")
    rng = random.Random(a.seed)
    report: dict = {"seed": a.seed, "per_build": a.per_build, "clang": clang, "builds": {}}
    for build in BUILDS:
        rows = [json.loads(line) for line in open(a.matrix / build / "records.jsonl")]
        rows = [r for r in rows if r["views"].get("pseudo", {}).get("code")]
        c: collections.Counter = collections.Counter()
        for row in rng.sample(rows, min(a.per_build, len(rows))):
            record = FunctionRecord.from_fission_json({
                "address": row["address"], "name": row["name"], **row["views"]["pseudo"]})
            c["sampled"] += 1
            if not compile_assembly(record.code, clang).ok:
                c["baseline_uncompilable"] += 1
                continue
            c["compilable"] += 1
            gold = gold_renames(row)
            edits = {
                "gold_renames": EditList(renames=gold),
                "function_rename": EditList(function_name="renamed_" + record.name),
                "comment": EditList(comments=[Comment(1, "checked entry")]),
            }
            for label, edit in edits.items():
                if label == "gold_renames" and not gold:
                    continue
                result = apply_edits(record, edit)
                symbols = {record.name: edit.function_name} if edit.function_name else {}
                verdict = compare(record.code, result.code, clang, symbols)
                c[f"{label}:{verdict.kind}"] += 1
                c[f"{label}:rejected_edits"] += len(result.rejected)
                c[f"{label}:applied_renames"] += len(result.applied_renames)
            names = [v.name for v in record.variables]
            for label, mutated in (("merge_variables", merge_variables(record.code, names, rng)),
                                   ("tweak_literal", tweak_literal(record.code, rng))):
                if mutated is None or mutated == record.code:
                    continue
                c[f"{label}:{compare(record.code, mutated, clang).kind}"] += 1
        report["builds"][build] = dict(c)
        print(build, dict(c), flush=True)
    a.report.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
