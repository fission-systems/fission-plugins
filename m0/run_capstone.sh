#!/bin/bash
# Reproduce the M0 pilot: build Capstone's core C files with -g, strip a copy, decompile the
# stripped copy with Fission, and measure how many DWARF variables align to Fission variables.
#   usage: run_capstone.sh <capstone-src-dir> <fission_cli> <empty-work-dir>
# macOS/arm64 only (dsymutil, dwarfdump, strip). Capstone is BSD-3-Clause.
set -eu
SRC="$1"; FISSION="$2"; WORK="$3"
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -z "$(ls -A "$WORK" 2>/dev/null)" ] || { echo "work dir must be empty: $WORK" >&2; exit 1; }
mkdir -p "$WORK"; cd "$WORK"
for O in O0 O1; do
  (cd "$SRC" && clang -g -$O -w -I. -Iinclude -DCAPSTONE_USE_SYS_DYN_MEM -dynamiclib -undefined dynamic_lookup \
     -o "$WORK/cscore_$O.dylib" MCInst.c MCInstPrinter.c MCInstrDesc.c MCRegisterInfo.c Mapping.c SStream.c cs.c utils.c)
  cp "cscore_$O.dylib" "cscore_$O.stripped.dylib"; strip -S "cscore_$O.stripped.dylib"   # clang already wrote cscore_$O.dylib.dSYM
  python3 "$HERE/dwarf_vars.py" "cscore_$O.dylib.dSYM" > "dwarf_$O.json"
  "$FISSION" decomp "cscore_$O.stripped.dylib" --all --json --timeout-ms 20000 2>/dev/null \
    | python3 -c "import sys,json; s=sys.stdin.read(); json.dump(json.loads(s[s.index('['):]), open('fission_$O.json','w'))"
done
python3 "$HERE/align.py" O0 O1 > report.json
echo "wrote $WORK/report.json"
