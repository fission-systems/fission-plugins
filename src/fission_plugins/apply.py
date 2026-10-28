"""Apply an `EditList` to Fission pseudocode without changing its meaning.

Only whole identifier tokens are rewritten; string/char literals, comments and
member names (`.x`, `->x`) are never touched. Any edit that could capture or
collide with another identifier is rejected, not repaired.
"""
from __future__ import annotations

import re

from .schema import ApplyResult, EditList, FunctionRecord

C_KEYWORDS = frozenset(
    """auto break case char const continue default do double else enum extern float for
    goto if inline int long register restrict return short signed sizeof static struct
    switch typedef union unsigned void volatile while bool true false NULL
    uchar uint ushort ulong uint8_t uint16_t uint32_t uint64_t int8_t int16_t int32_t
    int64_t size_t""".split()
)

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")
# Order matters: comments and literals are consumed before identifiers.
_TOKEN = re.compile(
    r"""//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|(?P<id>[A-Za-z_][A-Za-z0-9_]*)""",
    re.S,
)


def identifiers(code: str) -> set[str]:
    return {m.group("id") for m in _TOKEN.finditer(code) if m.group("id")}


def _rewrite(code: str, mapping: dict[str, str]) -> str:
    out: list[str] = []
    last = 0
    for m in _TOKEN.finditer(code):
        name = m.group("id")
        if name is None or name not in mapping:
            continue
        before = code[: m.start()].rstrip()
        if before.endswith(".") or before.endswith("->"):
            continue
        out.append(code[last : m.start()])
        out.append(mapping[name])
        last = m.end()
    out.append(code[last:])
    return "".join(out)


def apply_edits(record: FunctionRecord, edits: EditList) -> ApplyResult:
    rejected: list[tuple[str, str]] = []
    declared = {v.name for v in record.variables}
    used = identifiers(record.code)
    wanted: dict[str, str] = {}

    proposals = dict(edits.renames)
    if edits.function_name:
        proposals[record.name] = edits.function_name
    for old, new in proposals.items():
        if old != record.name and old not in declared:
            rejected.append((f"{old}->{new}", "not a variable Fission reported"))
        elif not _IDENT.match(new):
            rejected.append((f"{old}->{new}", "not a C identifier"))
        elif new in C_KEYWORDS:
            rejected.append((f"{old}->{new}", "C keyword or builtin type"))
        elif new in wanted.values():
            rejected.append((f"{old}->{new}", "duplicate new name"))
        elif new != old and new in used and new not in proposals:
            rejected.append((f"{old}->{new}", "collides with an existing identifier"))
        else:
            wanted[old] = new

    code = _rewrite(record.code, wanted)

    lines = code.split("\n")
    for c in edits.comments:
        if not 1 <= c.line <= len(lines):
            rejected.append((f"comment@{c.line}", "line out of range"))
            continue
        text = " ".join(c.text.split()).replace("*/", "")
        lines[c.line - 1] = f"{lines[c.line - 1]}  /* {text} */"
    return ApplyResult("\n".join(lines), wanted, rejected)
