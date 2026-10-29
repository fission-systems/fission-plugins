"""Build one project for one (arch, optimization) into a shared library with DWARF.

Output directory layout:
  objs/*.o          compiled units that succeeded
  lib.dylib         linked with `-undefined dynamic_lookup` (unresolved imports stay unresolved)
  lib.dylib.dSYM/   DWARF, from dsymutil
  lib.stripped.dylib  `strip -S` copy: this is what Fission sees
  build.json        flags, tool versions, per-unit success/failure

macOS only (clang, dsymutil, strip). Never executes the built code.

usage: build.py <project> <arch: arm64|x86_64> <opt: O0|O1|O2> <out-dir (must not exist)>
"""
from __future__ import annotations

import concurrent.futures
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECTS = json.loads((Path(__file__).parent / "projects.json").read_text())["projects"]


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


def units(cfg: dict) -> list[Path]:
    src = Path(cfg["src_dir"])
    names = cfg.get("files") or sorted(p.name for p in src.glob(cfg.get("glob", "*.c")))
    skip = cfg.get("exclude_substrings", [])
    out = [src / n for n in names if not any(s in n for s in skip)]
    if cfg.get("skip_files_defining_main"):  # command-line tools next to the library share helper names
        out = [p for p in out if not re.search(r"^int\s+main\s*\(", p.read_text(errors="replace"), re.M)]
    return out


def build(project: str, arch: str, opt: str, out: Path) -> dict:
    cfg = PROJECTS[project]
    out.mkdir(parents=True)
    (out / "objs").mkdir()
    gen = out / "generated"
    for rel, text in cfg.get("generated_files", {}).items():
        (gen / rel).parent.mkdir(parents=True, exist_ok=True)
        (gen / rel).write_text(text)
    incs = []
    for i in cfg.get("includes", []):
        incs += ["-I", str(gen) if i == "{generated}" else str(Path(cfg["src_dir"]) / i)]
    defs = [f"-D{d}" for d in cfg.get("defines", [])]
    flags = ["-g", f"-{opt}", "-arch", arch, "-w", *incs, *defs]

    def compile_one(src: Path):
        obj = out / "objs" / (src.stem + ".o")
        r = run(["clang", *flags, "-c", src, "-o", obj])
        return src.name, r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1][:200]

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        results = list(pool.map(compile_one, units(cfg)))
    ok = [n for n, good, _ in results if good]
    failed = {n: why for n, good, why in results if not good}
    objs = sorted((out / "objs").glob("*.o"))
    lib = out / "lib.dylib"
    link = run(["clang", "-dynamiclib", "-arch", arch, "-undefined", "dynamic_lookup", "-o", lib, *objs])
    linked = link.returncode == 0
    if linked:
        run(["dsymutil", lib])
        stripped = out / "lib.stripped.dylib"
        stripped.write_bytes(lib.read_bytes())
        run(["strip", "-S", stripped])
    manifest = {
        "project": project, "arch": arch, "opt": opt, "license": cfg["license"], "flags": flags,
        "clang": run(["clang", "--version"]).stdout.splitlines()[0],
        "units_ok": ok, "units_failed": failed,
        "linked": linked, "link_error": None if linked else link.stderr[-500:],
    }
    (out / "build.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    project, arch, opt, out = sys.argv[1:5]
    m = build(project, arch, opt, Path(out))
    print(f"{project} {arch} {opt}: {len(m['units_ok'])} units ok, {len(m['units_failed'])} failed, linked={m['linked']}")
    if m["link_error"]:
        print(m["link_error"])


if __name__ == "__main__":
    main()
