"""Probe Qwen-Scope SAE features on Fission pseudocode, with and without recovered names.

Descriptive only: the SAE features have no labels here, so nothing below says what a feature
"means". It measures (1) whether the SAE reconstructs these activations, (2) how the model's
likelihood of the code changes when Fission names are replaced, and (3) which features fire
on renamed identifiers but not on the originals.

usage: probe.py <fixture.json> <out.json> [layer ...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-2B-Base"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"
K = 50
RENAMES = {"param_1": "buf", "param_2": "len", "w8": "hash", "x0": "p", "x9": "n"}
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def decoder_layers(model):
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and len(mod) == 24:
            return mod
    raise RuntimeError("decoder layer list not found")


def encode_sae(sae, resid, subtract_bdec):
    x = resid - sae["b_dec"] if subtract_bdec else resid
    pre = x @ sae["W_enc"].T + sae["b_enc"]
    vals, idx = pre.topk(K, dim=-1)
    return vals, idx


def reconstruct(sae, vals, idx):
    out = torch.zeros(*vals.shape[:-1], sae["W_dec"].shape[1])
    out.scatter_(-1, idx, vals)
    return out @ sae["W_dec"].T + sae["b_dec"]


def run(code, tok, model, layers):
    enc = tok(code, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    captured = {}
    hooks = []
    for L in layers:
        def hook(_m, _i, out, L=L):
            captured[L] = (out[0] if isinstance(out, tuple) else out).detach().float()[0]
        hooks.append(decoder_layers(model)[L].register_forward_hook(hook))
    with torch.no_grad():
        logits = model(**enc).logits[0].float()
    for h in hooks:
        h.remove()
    ids = enc["input_ids"][0]
    nll = torch.nn.functional.cross_entropy(logits[:-1], ids[1:], reduction="none")
    return enc, offsets, captured, nll


def ident_tokens(code, offsets, names):
    """token index -> identifier, for tokens that start inside an occurrence of a wanted name."""
    spans = [(m.start(), m.end(), m.group()) for m in IDENT.finditer(code) if m.group() in names]
    out = {}
    for i, (a, b) in enumerate(offsets):
        while a < b and code[a].isspace():  # byte-level BPE offsets include the leading space
            a += 1
        for s, e, name in spans:
            if a >= s and b <= e and b > a:
                out.setdefault(name, []).append(i)
    return out


def main():
    fixture, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    layers = [int(x) for x in sys.argv[3:]] or [5, 11, 17, 23]
    row = json.loads(fixture.read_text())
    original = row["code"]
    renamed = IDENT.sub(lambda m: RENAMES.get(m.group(), m.group()), original)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    report = {"model": MODEL, "sae": SAE_REPO, "renames": RENAMES, "layers": {}}

    res = {}
    for tag, code in (("original", original), ("renamed", renamed)):
        enc, offsets, cap, nll = run(code, tok, model, layers)
        res[tag] = (code, offsets, cap, nll, enc["input_ids"].shape[1])
    # Per-token NLL restricted to the function body (everything after the first `{`).
    for tag in res:
        code, offsets, cap, nll, n = res[tag]
        start = code.index("{")
        body = [i for i, (a, _) in enumerate(offsets[1:], 1) if a >= start]
        report[f"{tag}_tokens"] = n
        report[f"{tag}_body_mean_nll"] = float(nll[[i - 1 for i in body]].mean())
        report[f"{tag}_body_tokens"] = len(body)

    for L in layers:
        sae = torch.load(hf_hub_download(SAE_REPO, f"layer{L}.sae.pt"), map_location="cpu")
        sae = {k: v.float() for k, v in sae.items()}
        info = {}
        # reconstruction sanity check on the original code, with and without the b_dec offset
        resid = res["original"][2][L]
        for sub in (False, True):
            vals, idx = encode_sae(sae, resid, sub)
            rec = reconstruct(sae, vals, idx)
            rel = ((rec - resid).norm(dim=-1) / resid.norm(dim=-1)).mean().item()
            info[f"relative_recon_error_subtract_bdec_{sub}"] = rel
        sub = info["relative_recon_error_subtract_bdec_True"] < info["relative_recon_error_subtract_bdec_False"]
        info["used_bdec_subtraction"] = sub
        feats = {}
        for tag in res:
            code, offsets, cap, nll, n = res[tag]
            vals, idx = encode_sae(sae, cap[L], sub)
            names = set(RENAMES) if tag == "original" else set(RENAMES.values())
            toks = ident_tokens(code, offsets, names)
            per = {}
            for name, ts in toks.items():
                per[name] = {"tokens": len(ts), "features": sorted({int(f) for t in ts for f in idx[t].tolist()}),
                             "mean_act": {}}
                acc = {}
                for t in ts:
                    for f, v in zip(idx[t].tolist(), vals[t].tolist()):
                        acc.setdefault(f, []).append(v)
                top = sorted(acc.items(), key=lambda kv: -sum(kv[1]))[:5]
                per[name]["top"] = [(f, round(sum(v) / len(v), 2), len(v)) for f, v in top]
            feats[tag] = per
        # feature sets that fire on renamed identifiers but on none of the original identifiers
        orig_set = {f for d in feats["original"].values() for f in d["features"]}
        new_only = {name: [f for f in d["features"] if f not in orig_set] for name, d in feats["renamed"].items()}
        orig_only = {name: [f for f in d["features"]] for name, d in feats["original"].items()}
        jacc = {}
        for new, old in ((RENAMES[o], o) for o in RENAMES):
            a, b = set(feats["original"][old]["features"]), set(feats["renamed"][new]["features"])
            jacc[f"{old}->{new}"] = round(len(a & b) / max(1, len(a | b)), 3)
        names_o = list(feats["original"])
        cross = {}
        for i, a in enumerate(names_o):
            for b in names_o[i + 1:]:
                A, B = set(feats["original"][a]["features"]), set(feats["original"][b]["features"])
                cross[f"{a}|{b}"] = round(len(A & B) / max(1, len(A | B)), 3)
        info["jaccard_original_vs_renamed_same_variable"] = jacc
        info["jaccard_between_original_names"] = cross
        info["features_only_on_renamed"] = {k: len(v) for k, v in new_only.items()}
        info["renamed_top_features"] = {k: v["top"] for k, v in feats["renamed"].items()}
        info["original_top_features"] = {k: v["top"] for k, v in feats["original"].items()}
        report["layers"][str(L)] = info
        print("layer", L, {k: info[k] for k in info if k.startswith("relative") or k == "used_bdec_subtraction"}, flush=True)
    out_path.write_text(json.dumps(report, indent=1))
    print("original body nll", report["original_body_mean_nll"], "renamed body nll", report["renamed_body_mean_nll"])


if __name__ == "__main__":
    main()
