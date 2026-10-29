# Variable-name recovery, zero training

Question: before any fine-tuning, how well does a base model recover the source names of Fission's placeholder
variables (`param_1`, `local_xx`)? `prepare_names.py` builds the set (arm64 only, `unique`-aligned variables, gold =
DWARF name); `zeroshot_names.py` runs Qwen3.5-2B-Base with a 3-example prompt (examples from the *other* project)
on a Colab L4. Two inputs: decompiled C only, and C plus assembly. A no-model baseline predicts the most frequent
gold name for the same Fission type in the other project.

Result (1004 targets from 400 functions; capstone 479, isl 525; one run, greedy, no seeds involved):

| | exact match | token F1 |
|---|---|---|
| type baseline | 3.8% | 3.8% |
| pseudocode | 11.5% | 12.2% |
| pseudocode + asm | 10.7% | 11.4% |

By kind (exact): parameters 6.2% (baseline 0.3%), stack locals 30.6% (baseline 19.1%), register locals 14.7% (1.1%).
By build (pseudocode, exact): isl O0 20.5%, isl O1 9.9%, capstone O0 9.0%, capstone O1 4.7%.

What this does and does not show:
- A 3-shot base model beats the type baseline everywhere, but exact match is low, especially for parameters, where the
  right name depends on the original project's conventions (`MI`, `ctx`).
- Adding assembly did **not** help with this prompt; the difference (about one point) is within what a different
  sample could change. This is a statement about this prompt and a model that was never trained on assembly.
- The decompiled code usually contains the function's own name (exported symbols survive `strip -S`), which is a strong
  hint for parameter names and is not available for stripped, non-exported functions. Scores for hidden functions
  would likely be lower. This leak is not removed here.
- O1 is much worse than O0, and O1/O2 also lose most labels to ambiguous alignment (see ../corpus/README.md).
- The gold name is the source name, not "a good name": `inst` for `MCInst *` counted wrong when the model said `mi`.
  Exact match understates usefulness; no human rating was done.
- x86-64 is not evaluated (alignment unverified). Two projects only, and few-shot examples come from the other one.

Reproduce: `prepare_names.py <matrix-dir> <out-dir> --per-build 100`, upload the output to a Colab VM and run
`zeroshot_names.py <data-dir> <out-dir>`. About 10 minutes on an L4 for both conditions.
