# Corpus pipeline: several input views, source-derived labels

Goal: give a model more than one decompiler's output. Each record pairs one binary function with
several **views** of the same code, and **labels taken from the source and DWARF**, not from any
decompiler. Which input combination works best (asm only, pseudocode only, both, P-code) is
an open experiment this data is meant to enable; nothing here has been trained or compared yet.

## Record (one JSON object per function, `records.jsonl`)

| field | content |
|---|---|
| `id`, `project`, `license`, `arch`, `opt`, `address`, `name`, `size` | identity and build provenance |
| `source` | `{ok, file, line, text}`: the original C function sliced from DWARF `decl_file`/`decl_line`; `ok=false` for macro-generated functions or unbalanced braces |
| `views.asm` | `llvm-objdump` instructions `{address, bytes, instruction}` for the function's address range (not Fission: each `fission_cli` call re-analyses the whole binary, ~7 s on isl) |
| `views.pseudo` | Fission `decomp` `{code, engine, variables}` on the **stripped** binary |
| `views.pcode_file` | gzipped Fission `raw-pcode --json`, relative to the extraction dir, **only for an evenly spaced sample** (`--pcode-sample`, default 100 per build) because of the same per-call cost; `null` otherwise or if raw-pcode failed. Rendering into a training lane is left to the training code |
| `variables[]` | DWARF `name`, `tag`, `type`, `location`; and its alignment to Fission variables: `method`, `outcome` (`unique`/`shared`/`multi-candidate`/`none`) and the candidate Fission variables |

The alignment outcome is stored for every variable. Filtering (for example "unique only") is a
training-time decision. `m0/README.md` has the manual precision check for the arm64 rules.

## Build and extract

```bash
python3 corpus/build.py <project> <arm64|x86_64> <O0|O1|O2> <new-dir>     # clang -g, dsymutil, strip -S copy
python3 corpus/extract.py <build-dir> <fission_cli> <new-dir> [--pcode-sample N]   # records.jsonl, pcode/, stats.json
python3 corpus/run_matrix.py <work-dir> <fission_cli> [project ...]      # all combinations, skipping finished ones
```

macOS with clang, dsymutil, dwarfdump, strip, llvm-objdump (Homebrew LLVM) and a built `fission_cli` only. The built code is never run.
Build recipes (source dir, includes, defines, hand-written config headers) are in `projects.json`;
project source trees are local checkouts under `Fission/vendor/`, not fetched.

## Projects so far

| project | license | units built | note |
|---|---|---|---|
| capstone | BSD-3-Clause | 8 core files | decoders for individual architectures are not built |
| isl | MIT | ~87 of ~110 | vendored in LLVM Polly; template fragments and command-line tools are skipped; config header written by hand |

`train_ok` is `null` for both: whether a project's license allows training released weights is a decision
for the maintainers, not something this pipeline decides. Two projects is the minimum for a project-level split
(train on one, test on the other) and is too few for conclusions. Candidates found locally without a ready build:
radare2 (LGPL, needs a generated `r_userconf.h`), zlib/bzip2/libedit snapshots under `fission-benchmark/decbench-data/sources`
(source files only, internal headers missing), many GPL projects (coreutils, bash, u-boot, ...) whose license needs a decision first.
Malware samples in those folders are excluded on purpose.

## First full matrix (arm64 and x86_64, O0/O1/O2; records = functions)

| project | arm64 O0 / O1 / O2 | x86_64 O0 / O1 / O2 |
|---|---|---|
| capstone | 192 / 160 / 160 | 192 / 160 / 159 |
| isl | 8343 / 4884 / 4865 | 8343 / 4882 / ~4880 |

Alignment is clean at O0 (isl arm64 O0: 27968 unique, 3747 none, 150 multi-candidate) and degrades at O1/O2
(isl arm64 O1: 13315 unique, **30330 shared**, 15911 none): the optimized code keeps several source variables in one
register, so many labels are ambiguous. Per-build numbers are in each build's `stats.json`.
Whole-function asm for 3 of 160 capstone arm64 O1 functions was longer with objdump than with Fission's `disasm --function`
(for example 28 vs 12 instructions); the cause on the Fission side was not investigated.

## Known limits

- **x86-64 alignment is partial.** Parameters and register locals are mapped; stack slots (CFA/RBP-based) are not, so x86-64
  stack variables always come out `none`. The x86-64 rules have not been checked by hand.
- **Alignment is positional** (argument index, stack offset, register name); it does not verify that values agree. On arm64 the
  manual check found no wrong unique pairs among 87, but two register cases were only partly right.
- **Source slices** start at the return type (up to three lines above the declared line) and do not include preceding comments;
  macro-generated functions are kept as records with `source.ok=false`.
- **Mach-O only.** ELF/PE binaries are a different DWARF/ABI story and are not covered; the user's HF corpus
  (`sjkim1127/decbench-multilang-corpus`, cJSON and pugixml, ELF x86-64) is a possible next source.
- Fission is the only decompiler so far. The point of the design is to add others (for example Ghidra) as extra `views.pseudo_*`
  fields on the same records; none is implemented.
- Records from different optimization levels or architectures of the same project share source text, so splits must be by project.
