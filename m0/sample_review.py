"""Draw a reproducible sample of uniquely aligned pairs and print a review sheet.

For each sampled pair the sheet shows the C source of the function (found by name in the
source dir), Fission's pseudocode, and the proposed DWARF-name -> Fission-name pairs.
Only functions whose pseudocode has at most MAX_LINES lines are eligible, so the sample is
biased toward small functions.

usage: sample_review.py <workdir with dwarf_*.json fission_*.json> <capstone-src> <O0|O1> <n> <seed>
"""
from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import align  # noqa: E402

MAX_LINES = 45


def find_source(src: Path, name: str) -> str:
    for f in sorted(src.glob("*.c")):
        text = f.read_text()
        for m in re.finditer(rf"^[A-Za-z_][^\n;{{}}]*\b{re.escape(name)}\s*\(", text, re.M):
            i = text.find("{", m.start())
            if i < 0 or ";" in text[m.start():i]:
                continue
            depth, j = 0, i
            while j < len(text):
                depth += (text[j] == "{") - (text[j] == "}")
                j += 1
                if depth == 0:
                    break
            return text[m.start():j]
    return "<source not found>"


def pairs_for(work: Path, level: str):
    dw = json.loads((work / f"dwarf_{level}.json").read_text())
    fi = {int(r["address"], 16): r for r in json.loads((work / f"fission_{level}.json").read_text()) if "code" in r}
    out = []
    for f in dw:
        row = fi[f["low_pc"]]
        fvars, fbase, pi, res = row["variables"], align.base_reg(f["frame_base"]), 0, []
        claims = collections.defaultdict(list)
        for v in f["variables"]:
            idx = None
            if v["tag"] == "param":
                idx, pi = pi, pi + 1
            m, c = align.candidates(v, fvars, fbase, idx)
            res.append((v, m, c))
            for x in (c[:1] if m == "param-index" else c):
                claims[x["name"]].append(v["name"])
        for v, m, c in res:
            if len(c) == 1 and len(claims[c[0]["name"]]) == 1 and row["code"].count("\n") <= MAX_LINES:
                out.append({"function": f["name"], "dwarf": v["name"], "dwarf_type": v.get("type"), "method": m,
                            "fission": c[0]["name"], "fission_type": c[0].get("type")})
    return out, fi, dw


def main():
    work, src, level, n, seed = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    pairs, fi, dw = pairs_for(work, level)
    rng = random.Random(seed)
    easy = [p for p in pairs if p["method"] == "param-index"]
    hard = [p for p in pairs if p["method"] != "param-index"]
    pick = rng.sample(hard, min(n // 2, len(hard))) + rng.sample(easy, min(n - n // 2, len(easy)))
    by_func = collections.defaultdict(list)
    for p in pick:
        by_func[p["function"]].append(p)
    addr = {f["name"]: f["low_pc"] for f in dw}
    json.dump(pick, sys.stdout.buffer if False else open(work / f"sample_{level}.json", "w"), indent=1)
    print(f"# {level}: {len(pick)} pairs in {len(by_func)} functions "
          f"(eligible: {len(hard)} non-param, {len(easy)} param-index)")
    for name, ps in by_func.items():
        print(f"\n===== {name} =====")
        source, code = find_source(src, name), fi[addr[name]]["code"].strip()
        if all(p["method"] == "param-index" for p in ps):  # positional rule: the signatures are enough
            print(source[: source.find("{")].strip())
            print(next((l for l in code.splitlines() if l.startswith(("int ", "void", "uchar", "unsigned", "long", "bool", "const", "char", "uint")) and name in l), code.splitlines()[0]))
        else:
            print(source)
            print("----- fission -----")
            print(code)
        print("----- pairs (dwarf -> fission) -----")
        for p in ps:
            print(f"  [{p['method']}] {p['dwarf']} ({p['dwarf_type']}) -> {p['fission']} ({p['fission_type']})")


if __name__ == "__main__":
    main()
