# M0 pilot: DWARF ↔ Fission variable alignment

Question: if we train a model to restore source variable names, how many Fission variables can we
label with a source name from DWARF, and how trustworthy are those labels?

## Setup (one project, one arch, two optimization levels)

- Source: Capstone 6.0.0-Alpha5 core C files (BSD-3-Clause), 8 files / 4,112 lines, from `Fission/vendor/`.
- Build: `clang -g -O0|-O1`, arm64 macOS dylib. DWARF comes from the `.dSYM`.
- Fission input: a copy with `strip -S`, so Fission cannot read source names from debug info. Function symbols remain.
- Fission: `fission_cli 0.2.3` from the `Fission-wt-abA` worktree build (commit not recorded), `decomp --all --json`.
- Functions are matched by start address: 192/192 (O0) and 160/160 (O1) DWARF functions are present in Fission's output.
- Reproduce: `m0/run_capstone.sh <capstone-dir> <fission_cli> <empty-dir>`. Raw numbers: `m0/results/capstone-core.report.json`.

## Result

| | O0 | O1 |
|---|---:|---:|
| DWARF variables (params + locals) | 574 | 579 |
| aligned to exactly one Fission variable ("unique") | 516 (89.9%) | 356 (61.5%) |
| no Fission candidate | 56 (9.8%) | 147 (25.4%) |
| shared with another DWARF variable | 0 | 70 (12.1%) |
| several candidates | 2 | 6 |
| Fission variables in those functions | 1,580 | 1,458 |
| Fission variables with a unique label | 516 (33%) | 356 (24%) |

By rule: parameters aligned by index are unique in 364/364 (O0) and 292/292 (O1) cases where the rule applied.
SP-relative stack slots at O0 match `stack_offset` exactly in 142 of 196 slots. At O1, register locals
(`DW_OP_regN` ↔ Fission `xN`/`wN`) are the weak point: 63 unique, 64 shared, 96 none out of 229.

Type sanity check on unique pairs (coarse class ptr/int/other): 394/516 agree at O0, 279/356 at O1. Most disagreements
are DWARF pointer vs Fission integer (79 of 122 at O0): Fission types pointers as `long long`/`unsigned long long`.
That is a typing weakness, not evidence of misalignment, so this check cannot detect wrong alignments.

## What this does and does not show

- Parameters can be labeled almost for free; locals at -O0 mostly; register locals at -O1 poorly. Naming labels for optimized
  code will be dominated by parameters unless alignment improves (e.g. use Fission's per-variable definition addresses
  against DWARF location-list ranges, which this pilot does not do).
- "Unique" is positional. Nothing verified that the aligned variables carry the same values. A wrong but unique match is
  indistinguishable here. Hand-check a sample before training on it.
- Roughly two thirds of Fission's variables (register temporaries like `xVar10`, `iVar37`) get no source name. A model must
  either leave them alone or the task needs a different label source.
- At -O0 a parameter appears twice in Fission (`param_1` and its stack copy `local_8`, same source variable). The applier
  rejects duplicate new names, so this case needs explicit handling.
- One 4k-line library, one architecture, no project-level split, no cross-compiler check. Numbers are a pilot, not a
  dataset statistic. x86-64 / PE and larger C projects are untested.
- Fission's O0 output for some functions looked semantically off in the earlier smoke test; alignment says nothing about that.

## Manual check of the "unique" labels (87 pairs)

Sample: `sample_review.py` seed 7, up to 25 non-parameter and 25 parameter pairs per level, drawn only from uniquely aligned pairs
in functions whose pseudocode has at most 45 lines (O1 had only 12 such non-parameter pairs, so 37 in total). I read each
function's C source against Fission's pseudocode and judged whether the Fission variable carries the source variable's value.
Parameter pairs were judged from signatures (position and parameter count), non-parameter pairs from the bodies. One reviewer
(an AI model, not a person); verdicts and notes are in `results/manual-review.json`.

| | pairs | correct | partial | wrong |
|---|---:|---:|---:|---:|
| O0 stack locals | 25 | 25 | 0 | 0 |
| O0 parameters | 25 | 25 | 0 | 0 |
| O1 register locals | 12 | 10 | 2 | 0 |
| O1 parameters | 25 | 25 | 0 | 0 |

"Partial" means the Fission register is shared between source variables or holds an off-by-one value:
`map_groups.i` ↔ `x11` (holds `i` or `i+1`), and `cs_strdup.new` ↔ `x0` (the same `x0` first holds `strlen(str)`, the source variable
`len`). Renaming such a variable to one source name would mislabel part of its uses.

What this supports: among small functions, uniquely aligned labels were right in 85 of 87 cases and not wrong in the other 2, with
the register-local category at -O1 the only one with any problems. What it does not: n is small (12 for the weakest category, whose
95% error bound is still about 25%), large functions were excluded by construction, and the 70 "shared" and 147 "none" O1 variables
were not examined at all. "Correct" refers to variable identity only; Fission's bodies were often wrong or incomplete in these
functions (for example `readBytes32` loses its computation and returns an uninitialised local, `MCOperand_CreateImm1` drops the
stores), so a model trained on these inputs sees damaged code.

## Next

1. Try address-range alignment for register locals at O1.
2. Handle registers shared between source variables (do not rename them, or split by live range).
3. Scale to several C projects with a project-level split, then build the training set.
