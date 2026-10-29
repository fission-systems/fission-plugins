# SAE probe: Qwen3.5-2B-Base on Fission pseudocode

Question: does the model's internal state (Qwen-Scope sparse autoencoder features) treat Fission's anonymous
names (`param_1`, `w8`, `x0`) differently from source-like names (`buf`, `hash`)? Descriptive pilot, one function.

## Setup

- Model `Qwen/Qwen3.5-2B-Base` (Apache-2.0), fp32, CPU. Needs a newer `transformers` than the global 5.0.0rc1
  (`qwen3_5` unknown there); run in a local `.venv` with `transformers` 5.21.0 (`python3 -m venv --system-site-packages .venv`).
- SAE `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50`, layers 5, 11, 17, 23 (residual stream after the block, 32,768 features, top-50).
  Its `LICENSE` file is Apache-2.0 text, while the model card metadata says `other` (name `qwen`): inconsistent, check before redistributing anything derived from it.
- Input: `tests/fixtures/checksum.fission.json` (Fission output for a hash loop, arm64 -O1). The renamed copy replaces
  `param_1->buf, param_2->len, w8->hash, x0->p, x9->n` everywhere (a made-up mapping, not DWARF names).
- Run: `.venv/bin/python sae/probe.py tests/fixtures/checksum.fission.json out.json`. Raw output: `sae/results/checksum-o1-probe.json`.

## Findings

1. **The SAE only partly reconstructs these activations.** Mean relative error 0.38 / 0.46 / 0.43 / 0.43 (layers 5/11/17/23),
   using the encoder without subtracting `b_dec` (the card's recipe; subtracting it was worse, 0.54-0.69). I do not have a
   reference number for what this SAE achieves on its training data, so I cannot say whether 0.4 is normal for it or a sign of
   a mismatch (token position effects, scaling). Treat feature-level conclusions as provisional.
2. **Likelihood.** Total negative log-likelihood of the function body: 310.7 nats original (283 tokens) vs 302.0 nats renamed
   (258 tokens); per-token mean is higher for renamed (1.10 vs 1.17). The two texts have different tokenizations, so neither number
   shows that real names help. One function, one mapping.
3. **Feature overlap.** For the same variable, the set of features firing on its original-name tokens vs its renamed tokens has
   Jaccard 0.13-0.30 at layer 5 and 0.29-0.41 at layer 23, i.e. overlap grows with depth. Different original variables overlap
   0.19-0.32 on average. These Jaccards depend on how many tokens each name has, so they are not a clean separation measure.
4. **A shared candidate.** Layer 17 feature 14025 is among the top features for both `param_1` (mean 2.57) and `buf` (3.22);
   layer 11 feature 7850 for `param_1` (1.33), `buf` (1.01) and `hash` (1.05). That is consistent with a feature that tracks
   "variable role" independent of spelling, but it is a hypothesis: there are no feature labels here and no ablation/steering test.
5. Layer-23 top features (11156, 29838, 31860) fire on nearly every identifier; likely next-token machinery rather than naming.

## What this does not show

Nothing here says fine-tuning will work, or which layer a naming signal lives in. Next experiments if pursued: many functions from
`m0/` (with DWARF names as the "renamed" side), per-feature top-activating contexts to label features, and an ablation of
feature 14025 at layer 17 to see whether predicted names change.
