"""Turn one build into per-function records with several input views and source-derived labels.

For every DWARF function that Fission also lists (matched by start address):
  views.asm      llvm-objdump instructions (address, bytes, text) for the function's address range.
                 Not Fission: `fission_cli` re-analyses the whole binary on every call (~7 s for isl),
                 and assembly does not depend on any decompiler anyway.
  views.pseudo   Fission `decomp` pseudocode (`code`) and its variable table
  views.pcode    Fission `raw-pcode --json`, stored gzipped next to the JSONL (rendering into
                 a training lane is left to the training code). One Fission call per function costs
                 the same whole-binary analysis, so only an evenly spaced sample of --pcode-sample
                 functions per build (default 100) gets it; the rest have `pcode_file: null`.
  source         the original C function text sliced from the DWARF decl file/line
  variables      DWARF variable name/type/location plus how it aligns to a Fission variable

Labels come from the source and DWARF, never from a decompiler. The alignment outcome is kept
for every variable (unique / shared / multi-candidate / none); filtering is a training-time
decision.

usage: extract.py <build-dir> <fission_cli> <out-dir (must not exist)> [--limit N] [--pcode-sample N]
"""
from __future__ import annotations

import collections
import bisect
import concurrent.futures
import gzip
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from m0 import align, dwarf_vars  # noqa: E402


def sh(cmd, timeout=300):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)


def json_from(text: str):
    """Parse the JSON after any log lines Fission prints first; None when there is none."""
    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not starts:
        return None
    try:
        return json.loads(text[min(starts):])
    except json.JSONDecodeError:
        return None


OBJDUMP = shutil.which("llvm-objdump") or "/opt/homebrew/opt/llvm/bin/llvm-objdump"
ASM_LINE = re.compile(r"^\s*([0-9a-f]+):\t((?:[0-9a-f]{2} ?)+)\t(.+)$")


def disassemble(binary: Path, arch: str) -> tuple[list[int], list[dict]]:
    """Every instruction of the binary's code, sorted by address. Symbol labels are dropped on purpose."""
    out = sh([OBJDUMP, "-d", "--macho", f"--arch={arch}", binary], timeout=600).stdout
    insns = []
    for line in out.splitlines():
        m = ASM_LINE.match(line)
        if m:
            insns.append({"address": hex(int(m[1], 16)), "bytes": m[2].strip(), "instruction": " ".join(m[3].split())})
    insns.sort(key=lambda i: int(i["address"], 16))
    return [int(i["address"], 16) for i in insns], insns


def asm_of(addrs: list[int], insns: list[dict], start: int, size: int | None) -> list[dict]:
    lo = bisect.bisect_left(addrs, start)
    hi = bisect.bisect_left(addrs, start + size) if size else lo
    return insns[lo:hi]


def slice_source(path: str | None, line: int | None, name: str) -> dict:
    """Cut the function text starting at DW_AT_decl_line; verify it contains the function name."""
    if not path or not line or not Path(path).is_file():
        return {"ok": False, "why": "no decl file/line"}
    lines = Path(path).read_text(errors="replace").split("\n")
    start = line - 1
    for _ in range(3):  # return type / attributes on earlier lines
        prev = lines[start - 1].rstrip() if start > 0 else ""
        if not prev or prev.endswith((";", "}", "*/")) or prev.lstrip().startswith(("#", "//")):
            break
        start -= 1
    text = "\n".join(lines[start:])
    i = text.find("{")
    if i < 0 or name not in text[:i] or ";" in text[:i]:
        return {"ok": False, "why": "name not in signature (macro-generated?)"}
    depth, j, n = 0, i, len(text)
    while j < n:
        c = text[j]
        if text.startswith("//", j):
            j = text.find("\n", j)
            j = n if j < 0 else j
            continue
        if text.startswith("/*", j):
            j = text.find("*/", j + 2)
            j = n if j < 0 else j + 2
            continue
        if c in "\"'":
            k = j + 1
            while k < n and text[k] != c:
                k += 2 if text[k] == "\\" else 1
            j = k + 1
            continue
        depth += (c == "{") - (c == "}")
        j += 1
        if depth == 0:
            break
    if depth != 0:
        return {"ok": False, "why": "unbalanced braces"}
    return {"ok": True, "file": path, "line": start + 1, "text": text[:j]}


def align_function(f: dict, fvars: list[dict], arch: str) -> list[dict]:
    fbase, pidx, res = align.base_reg(f["frame_base"]), 0, []
    claims = collections.defaultdict(list)
    for v in f["variables"]:
        idx = None
        if v["tag"] == "param":
            idx, pidx = pidx, pidx + 1
        method, cands = align.candidates(v, fvars, fbase, idx, arch)
        res.append((v, method, cands))
        for c in (cands[:1] if method == "param-index" else cands):
            claims[c["name"]].append(v["name"])
    out = []
    for v, method, cands in res:
        if not cands:
            outcome = "none"
        elif len(cands) == 1 and len(claims[cands[0]["name"]]) == 1:
            outcome = "unique"
        elif len(cands) > 1 and all(len(claims[c["name"]]) == 1 for c in cands):
            outcome = "multi-candidate"
        else:
            outcome = "shared"
        out.append({"name": v["name"], "tag": v["tag"], "type": v.get("type"),
                    "location": (v.get("location") or {}).get("kind", "none"),
                    "method": method, "outcome": outcome,
                    "fission": [{"name": c["name"], "type": c.get("type"), "kind": c["kind"]} for c in cands]})
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    def opt(flag, default):
        if flag not in sys.argv:
            return default
        v = sys.argv[sys.argv.index(flag) + 1]
        args.remove(v)
        return int(v)

    limit = opt("--limit", None)
    pcode_sample = opt("--pcode-sample", 100)
    build_dir, fission, out = Path(args[0]), args[1], Path(args[2])
    if (out / "stats.json").exists():
        sys.exit(f"already extracted: {out}")
    (out / "pcode").mkdir(parents=True, exist_ok=True)  # a partial earlier run is overwritten, never removed
    manifest = json.loads((build_dir / "build.json").read_text())
    stripped = build_dir / "lib.stripped.dylib"

    dw = dwarf_vars.parse(sh(["dwarfdump", "--debug-info", build_dir / "lib.dylib.dSYM"]).stdout)
    decomp = json_from(sh([fission, "decomp", stripped, "--all", "--json", "--timeout-ms", "20000"], timeout=3600).stdout)
    rows = {int(r["address"], 16): r for r in decomp or [] if "code" in r}
    funcs = [f for f in dw if f["low_pc"] in rows]
    if limit:
        funcs = funcs[:limit]
    addrs, insns = disassemble(stripped, manifest["arch"])
    step = max(1, len(funcs) // pcode_sample) if pcode_sample else 0
    sampled = {f["low_pc"] for f in funcs[::step][:pcode_sample]} if step else set()

    def one(f):
        addr = hex(f["low_pc"])
        row = rows[f["low_pc"]]
        asm = asm_of(addrs, insns, f["low_pc"], row.get("size"))
        raw = sh([fission, "raw-pcode", stripped, "--addr", addr, "--json", "--no-db"]).stdout if f["low_pc"] in sampled else ""
        pcode_path = None
        if raw and json_from(raw) is not None:
            pcode_path = out / "pcode" / f"{f['low_pc']:x}.json.gz"
            pcode_path.write_bytes(gzip.compress(raw[raw.find("{"):].encode()))
        return {
            "id": f"{manifest['project']}:{manifest['arch']}:{manifest['opt']}:{addr}",
            "project": manifest["project"], "license": manifest["license"],
            "arch": manifest["arch"], "opt": manifest["opt"], "address": addr, "name": f["name"],
            "size": row.get("size"),
            "source": slice_source(f.get("decl_file"), f.get("decl_line"), f["name"]),
            "views": {
                "asm": asm,
                "asm_ok": bool(asm),
                "pseudo": {"code": row["code"], "engine": row.get("engine_used"), "variables": row.get("variables", [])},
                "pcode_file": str(pcode_path.relative_to(out)) if pcode_path else None,
            },
            "variables": align_function(f, row.get("variables", []), manifest["arch"]),
        }

    with concurrent.futures.ThreadPoolExecutor(3) as pool:  # each fission_cli call already uses ~4 cores
        records = list(pool.map(one, funcs))
    with open(out / "records.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    outcomes = collections.Counter(v["outcome"] for r in records for v in r["variables"])
    stats = {"build": manifest["project"] + ":" + manifest["arch"] + ":" + manifest["opt"],
             "dwarf_functions": len(dw), "in_fission": len([f for f in dw if f["low_pc"] in rows]),
             "records": len(records), "source_ok": sum(r["source"]["ok"] for r in records),
             "asm_ok": sum(r["views"]["asm_ok"] for r in records),
             "pcode_sampled": len(sampled),
             "pcode_ok": sum(r["views"]["pcode_file"] is not None for r in records),
             "variable_outcomes": dict(outcomes)}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
