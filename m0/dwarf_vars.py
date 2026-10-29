"""Extract source variable names and locations from `dwarfdump --debug-info` text.

M0 ground truth: for every DW_TAG_subprogram with a low_pc, its parameters and local
variables (including those in nested lexical blocks) with name, type name and location.
Locations are kept as raw strings plus a coarse classification:
  fbreg  -> stack slot, with the offset and the function's frame base register
  reg    -> a DWARF register number (loclist entries are merged, ranges kept)
  other  -> anything else (constants, optimized out, unsupported expressions)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys

TAG = re.compile(r"^(0x[0-9a-f]+):(\s+)(DW_TAG_\w+)")
ATTR = re.compile(r"^\s+(DW_AT_\w+)\s*\((.*)\)\s*$")
STR = re.compile(r'^"(.*)"$')
FBREG = re.compile(r"DW_OP_fbreg ([+-]?\d+)")
BREG = re.compile(r"DW_OP_breg(\d+) \w+([+-]\d+)")
REG = re.compile(r"DW_OP_reg(\d+)")
LOCLIST_START = re.compile(r"^\s+DW_AT_location\s*\(0x[0-9a-f]+:\s*$")
RANGE = re.compile(r"\[(0x[0-9a-f]+), (0x[0-9a-f]+)\):\s*(.*)")


def _str(v: str) -> str | None:
    m = STR.match(v.strip())
    return m.group(1) if m else None


def classify(loc_lines: list[str]) -> dict:
    out = {"kind": "other", "raw": loc_lines}
    text = " ".join(loc_lines)
    if len(loc_lines) == 1 and not RANGE.search(text):
        if (m := FBREG.search(text)):
            out.update(kind="fbreg", offset=int(m.group(1)))
        elif (m := BREG.search(text)):
            out.update(kind="breg", reg=int(m.group(1)), offset=int(m.group(2)))
        elif (m := REG.search(text)):
            out.update(kind="reg", regs=[int(m.group(1))])
        return out
    regs = []
    for l in loc_lines:
        r = RANGE.search(l)
        if r and (m := REG.search(r.group(3))):
            regs.append(int(m.group(1)))
    if regs:
        out.update(kind="reg", regs=sorted(set(regs)), loclist=True)
    return out


def parse(text: str) -> list[dict]:
    funcs: list[dict] = []
    cur = None
    var = None
    loc_lines: list[str] | None = None
    in_loclist = False
    stack: list[tuple[int, str]] = []  # (indent, tag)
    cur_tag = None

    def finish_var():
        nonlocal var, loc_lines
        if var is not None and cur is not None:
            if loc_lines:
                var["location"] = classify(loc_lines)
            cur["variables"].append(var)
        var, loc_lines = None, None

    for line in text.splitlines():
        m = TAG.match(line)
        if m:
            finish_var()
            in_loclist = False
            indent, tag = len(m.group(2)), m.group(3)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, tag))
            cur_tag = tag
            if tag == "DW_TAG_subprogram":
                cur = {"name": None, "low_pc": None, "high_pc": None, "frame_base": None, "variables": []}
                funcs.append(cur)
            elif tag in ("DW_TAG_formal_parameter", "DW_TAG_variable") and cur is not None \
                    and any(t == "DW_TAG_subprogram" for _, t in stack[:-1]):
                var = {"tag": "param" if tag == "DW_TAG_formal_parameter" else "local",
                       "name": None, "order": len(cur["variables"]), "type": None}
            elif tag == "DW_TAG_compile_unit":
                cur = None
            continue
        if loc_lines is not None and in_loclist:
            if line.strip().startswith("[") or line.strip().startswith("0x"):
                loc_lines.append(line.strip())
                continue
            in_loclist = False
        if var is not None and LOCLIST_START.match(line):
            loc_lines, in_loclist = [], True
            continue
        a = ATTR.match(line)
        if not a:
            continue
        k, v = a.groups()
        if var is not None:
            target = var
        elif cur_tag == "DW_TAG_subprogram" and cur is not None:
            target = cur
        else:
            continue
        if k == "DW_AT_name":
            target["name"] = _str(v)
        elif k == "DW_AT_low_pc" and var is None:
            target["low_pc"] = int(v, 16)
        elif k == "DW_AT_high_pc" and var is None:
            target["high_pc"] = int(v, 16)
        elif k == "DW_AT_frame_base" and var is None:
            target["frame_base"] = v
        elif k == "DW_AT_type" and var is not None:
            t = re.search(r'"([^"]+)"', v)
            var["type"] = t.group(1) if t else v
        elif k == "DW_AT_location" and var is not None:
            loc_lines = [v]
            in_loclist = True
        elif k == "DW_AT_const_value" and var is not None:
            var["location"] = {"kind": "const", "raw": [v]}
    finish_var()
    return [f for f in funcs if f["low_pc"] is not None and f["name"]]


def main():
    dsym = sys.argv[1]
    text = subprocess.run(["dwarfdump", "--debug-info", dsym], capture_output=True, text=True, check=True).stdout
    json.dump(parse(text), sys.stdout, indent=1)


if __name__ == "__main__":
    main()
