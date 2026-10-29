"""Build and extract every (project, arch, opt) combination that is not already done.

usage: run_matrix.py <work-dir> <fission_cli> [project ...] [--pcode-sample N]
Existing output directories are left alone (never overwritten or removed).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ARCHS = ["arm64", "x86_64"]
OPTS = ["O0", "O1", "O2"]


def main():
    work, fission = Path(sys.argv[1]), sys.argv[2]
    rest = sys.argv[3:]
    extract_args = []
    if "--pcode-sample" in rest:
        i = rest.index("--pcode-sample")
        extract_args, rest = rest[i:i + 2], rest[:i] + rest[i + 2:]
    projects = rest or ["capstone", "isl"]
    work.mkdir(parents=True, exist_ok=True)
    for project in projects:
        for arch in ARCHS:
            for opt in OPTS:
                name = f"{project}-{arch}-{opt}"
                build_dir, ex_dir = work / name, work / f"ex-{name}"
                if not build_dir.exists():
                    subprocess.run([sys.executable, HERE / "build.py", project, arch, opt, build_dir], check=False)
                if build_dir.exists() and not (ex_dir / "stats.json").exists() and (build_dir / "lib.stripped.dylib").exists():
                    subprocess.run([sys.executable, HERE / "extract.py", build_dir, fission, ex_dir, *extract_args], check=False)


if __name__ == "__main__":
    main()
