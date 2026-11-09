"""Compile-and-compare oracle: did an edit change what the pseudocode compiles to?

Fission pseudocode is not valid C as emitted (Ghidra-style type names, project types it does not
declare, globals it does not define), so `compile_assembly` adds a fixed prelude and then stubs only
what clang reports as missing: unknown type names become `long long`, undeclared identifiers become
`extern long long`. The same stubs are reused for the edited code, so a difference in the generated
assembly can only come from the edit.

Renaming a local variable must not change optimized assembly (clang discards value names), so
`compare` calls an edit *identical* only when both versions compile and their assembly is equal
after mapping any renamed function symbol back. This is a sufficient check, not a necessary one:
`different` is a flag to look at, `identical` is proof for this compiler and target.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field

PRELUDE = """#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
typedef unsigned char uchar; typedef unsigned char byte; typedef unsigned short ushort;
typedef unsigned int uint; typedef unsigned long ulong; typedef long long longlong;
typedef unsigned long long ulonglong; typedef signed char sbyte;
typedef unsigned char undefined; typedef unsigned char undefined1; typedef unsigned short undefined2;
typedef unsigned int undefined4; typedef unsigned long long undefined8;
"""

FLAGS = [
    "-std=gnu89", "-O1", "-g0", "-w", "-S", "-fno-asynchronous-unwind-tables",
    "-Wno-error=int-conversion", "-Wno-error=incompatible-pointer-types",
    "-Wno-error=return-mismatch", "-Wno-error=incompatible-function-pointer-types",
    "-Wno-error=int-to-pointer-cast", "-x", "c", "-", "-o", "-",
]
TARGET = "arm64-apple-macos"
_UNKNOWN_TYPE = re.compile(r"unknown type name '(\w+)'")
_UNDECLARED = re.compile(r"use of undeclared identifier '(\w+)'")


def find_clang() -> str | None:
    for candidate in (os.environ.get("CLANG"), "/opt/homebrew/opt/llvm/bin/clang", shutil.which("clang")):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


@dataclass
class Compiled:
    ok: bool
    assembly: str = ""
    stubs: list[str] = field(default_factory=list)
    error: str = ""


def compile_assembly(code: str, clang: str, stubs: list[str] | None = None,
                     target: str = TARGET, max_rounds: int = 8) -> Compiled:
    """Compile `code` to assembly, adding stubs for missing types and identifiers as clang asks."""
    stubs = list(stubs or [])
    error = ""
    for _ in range(max_rounds):
        source = PRELUDE + "\n".join(stubs) + "\n" + code
        proc = subprocess.run([clang, "-target", target, *FLAGS], input=source,
                              capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            return Compiled(True, proc.stdout, stubs)
        error = proc.stderr
        new = [f"typedef long long {m};" for m in _UNKNOWN_TYPE.findall(error)]
        new += [f"extern long long {m};" for m in _UNDECLARED.findall(error)]
        new = [s for s in dict.fromkeys(new) if s not in stubs]
        if not new:
            break
        stubs += new
    first = next((ln for ln in error.splitlines() if "error:" in ln), error[:200])
    return Compiled(False, "", stubs, first)


@dataclass
class Verdict:
    kind: str  # identical | different | baseline_uncompilable | edited_uncompilable
    detail: str = ""


def _normalize(assembly: str, symbols: dict[str, str]) -> str:
    """Map renamed function symbols back and drop the `; @name` comments clang writes."""
    for old, new in symbols.items():
        # also matches compiler-made local symbols such as `l_switch.table.<function>`
        assembly = re.sub(rf"(?<![A-Za-z0-9$]){re.escape(new)}(?![\w$])", old, assembly)
    return "\n".join(re.sub(r"\s*;.*$", "", line).rstrip() for line in assembly.splitlines())


def compare(before: str, after: str, clang: str, symbols: dict[str, str] | None = None) -> Verdict:
    """Compare two versions of one function; `symbols` maps renamed functions old -> new."""
    base = compile_assembly(before, clang)
    if not base.ok:
        return Verdict("baseline_uncompilable", base.error)
    edited = compile_assembly(after, clang, base.stubs)
    if not edited.ok:
        return Verdict("edited_uncompilable", edited.error)
    if _normalize(edited.assembly, symbols or {}) == _normalize(base.assembly, {}):
        return Verdict("identical")
    return Verdict("different")
