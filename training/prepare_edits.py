"""Build the D0 fine-tuning set: Fission pseudocode + variable table -> edit-list JSON.

One example per (function, build). The prompt holds the pseudocode, the Fission variable table and the
names to restore; the completion is the edit list `{"renames": {...}}` and is the only part a trainer
should put loss on. Targets are the Fission variables whose DWARF counterpart is `unique` (see
corpus/README.md); `shared`/`none`/`multi-candidate` variables are never targets, so no loss is placed on
labels we do not have. Every gold edit is checked with `apply_edits`; entries the applier would reject
(duplicate new names, collisions) are dropped.

Splits are by group, never by row. Functions whose pseudocode is identical after replacing identifiers and
numbers (template clones such as `isl_*_list_*`) form one group in every build; a group is wholly in one
split. Groups larger than --big-group go to train. Capstone is a separate cross-project set (`xproj`) that
is never trained on. With two projects this is a pipeline-development set, not a generalization estimate.

usage: prepare_edits.py <matrix-dir> <out-dir (must not exist)> [--archs arm64] [--opts O0,O1,O2]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fission_plugins import EditList, FunctionRecord, apply_edits  # noqa: E402
from fission_plugins.apply import C_KEYWORDS  # noqa: E402

PROJECTS = {"isl": "train-pool", "capstone": "xproj"}
MAX_TARGETS = 20
MAX_TABLE = 60
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|0x[0-9a-fA-F]+|\d+|\S")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")
_C_WORDS = frozenset(
    "auto break case char const continue default do double else enum extern float for goto if inline int long "
    "register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while".split()
)


def body_hash(code: str) -> str:
    """Hash of the token shape: identifiers (except C keywords) and numbers collapsed."""
    out = []
    for tok in _TOKEN.findall(code):
        if tok[0].isalpha() or tok[0] == "_":
            out.append(tok if tok in _C_WORDS else "I")
        elif tok[0].isdigit():
            out.append("N")
        else:
            out.append(tok)
    return hashlib.sha1(" ".join(out).encode()).hexdigest()


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        self.parent[self.find(a)] = self.find(b)


def unit(key: str, seed: int) -> float:
    """Deterministic value in [0, 1) for a key."""
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:12], 16) / 16**12


def gold_edits(row: dict) -> tuple[dict[str, str], collections.Counter]:
    """Fission variable -> source name for targets that survive filtering and the applier."""
    drop: collections.Counter = collections.Counter()
    code = row["views"]["pseudo"]["code"]
    wanted: dict[str, str] = {}
    for v in row["variables"]:
        if v["outcome"] != "unique" or v["tag"] not in ("param", "local") or not v["name"]:
            drop["not_unique_or_unnamed"] += 1
            continue
        old, new = v["fission"][0]["name"], v["name"]
        if old == new:
            drop["name_unchanged"] += 1
        elif not _IDENT.match(new) or new in C_KEYWORDS:
            drop["invalid_new_name"] += 1
        elif not re.search(rf"\b{re.escape(old)}\b", code):
            drop["variable_absent_from_code"] += 1
        elif old in wanted:
            drop["duplicate_old_name"] += 1
        elif new in wanted.values():
            drop["duplicate_new_name"] += 1
        else:
            wanted[old] = new
    record = FunctionRecord.from_fission_json({"address": row["address"], "name": row["name"], **row["views"]["pseudo"]})
    for _ in range(3):  # drop what the applier rejects; repeat in case a drop frees a name
        result = apply_edits(record, EditList(renames=wanted))
        rejected = {w.split("->")[0] for w, _ in result.rejected}
        if not rejected:
            break
        for old in rejected:
            drop["applier_rejected"] += 1
            wanted.pop(old, None)
    return wanted, drop


def masked_code(code: str, name: str, address: str) -> str:
    digits = address.lower().removeprefix("0x").zfill(8)
    return re.sub(rf"\b{re.escape(name)}\b", f"FUN_{digits}", code)


def build_prompt(row: dict, targets: list[str], masked: bool) -> str:
    pseudo = row["views"]["pseudo"]
    code = masked_code(pseudo["code"], row["name"], row["address"]) if masked else pseudo["code"]
    table = []
    for v in pseudo["variables"][:MAX_TABLE]:
        where = f"arg {v['arg_index']}" if v.get("arg_index") is not None else v["kind"]
        table.append(f"- {v['name']}: {v['type']} ({where})")
    return (
        f"Fission pseudocode:\n```c\n{code.rstrip()}\n```\n"
        f"Variables:\n" + "\n".join(table) + "\n"
        f"Restore the source names for: {', '.join(targets)}\n"
        "Edit list (JSON):\n"
    )


def completion_for(renames: dict[str, str], targets: list[str]) -> str:
    return json.dumps({"renames": {t: renames[t] for t in targets}}, ensure_ascii=False)


def assign_splits(components: dict, sizes: dict, seed: int, big_group: int, val: float, test: float) -> dict:
    out = {}
    for comp in components:
        if sizes[comp] > big_group:
            out[comp] = "train"
            continue
        u = unit(str(comp), seed)
        out[comp] = "test" if u < test else "val" if u < test + val else "train"
    return out


def load_rows(matrix: Path, archs: list[str], opts: list[str]):
    for project in PROJECTS:
        for arch in archs:
            for opt in opts:
                path = matrix / f"ex-{project}-{arch}-{opt}" / "records.jsonl"
                if not path.exists():
                    continue
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    if (row["views"].get("pseudo") or {}).get("code"):
                        yield row


def count_tokens(texts: list[str], names: list[str]) -> dict[str, list[int]]:
    if not names:
        return {}
    from transformers import AutoTokenizer

    out = {}
    for name in names:
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        out[name] = [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("matrix", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--archs", default="arm64")
    p.add_argument("--opts", default="O0,O1,O2")
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--big-group", type=int, default=100, help="groups with more functions than this stay in train")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--tokenizers", default="Qwen/Qwen3.5-2B-Base,Qwen/Qwen2.5-Coder-1.5B")
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"refusing to overwrite {a.out}")

    rows = list(load_rows(a.matrix, a.archs.split(","), a.opts.split(",")))
    uf = UnionFind()
    for row in rows:
        key = (row["project"], row["name"])
        uf.union(key, ("body", row["project"], body_hash(row["views"]["pseudo"]["code"])))
    func_comp = {(r["project"], r["name"]): uf.find((r["project"], r["name"])) for r in rows}
    sizes = collections.Counter(func_comp.values())  # functions per group
    split_of = assign_splits(list(sizes), sizes, a.seed, a.big_group, a.val, a.test)

    stats: collections.Counter = collections.Counter()
    drops: collections.Counter = collections.Counter()
    examples = []
    seen = set()
    for row in rows:
        project = row["project"]
        split = "xproj" if PROJECTS[project] == "xproj" else split_of[func_comp[(project, row["name"])]]
        renames, drop = gold_edits(row)
        drops.update(drop)
        targets = [t for t in renames][:MAX_TARGETS]
        if len(renames) > MAX_TARGETS:
            stats["targets_truncated"] += 1
        if not targets:
            stats["no_targets"] += 1
            continue
        group = hashlib.sha1(str(func_comp[(project, row["name"])]).encode()).hexdigest()[:12]
        for masked in ((False, True) if split != "train" else (unit(f"{project}:{row['name']}", a.seed + 1) < 0.5,)):
            prompt = build_prompt(row, targets, masked)
            # the masked prompt carries the build's address, so dedupe on the unmasked text
            dedupe_key = (split, row["project"], row["name"], masked, build_prompt(row, targets, False))
            if dedupe_key in seen:
                stats["duplicate_across_builds"] += 1
                continue
            seen.add(dedupe_key)
            examples.append(dict(
                id=row["id"] + ("#masked" if masked else ""), split=split, group=group, project=project,
                license=row["license"], arch=row["arch"], opt=row["opt"], address=row["address"], name=row["name"],
                function_name_masked=masked, targets=targets, prompt=prompt, completion=completion_for(renames, targets)))

    tokens = count_tokens([e["prompt"] + e["completion"] for e in examples], [t for t in a.tokenizers.split(",") if t])
    keep = []
    for i, e in enumerate(examples):
        counts = {name: lens[i] for name, lens in tokens.items()}
        e["n_tokens"] = counts
        if counts and max(counts.values()) > a.max_tokens:
            stats["dropped_over_max_tokens"] += 1
            continue
        keep.append(e)

    a.out.mkdir(parents=True)
    by_file: dict[str, list] = collections.defaultdict(list)
    for e in keep:
        suffix = "-masked" if e["function_name_masked"] and e["split"] != "train" else ""
        by_file[f"{e['split']}{suffix}"].append(e)
    files = {}
    for name, items in sorted(by_file.items()):
        path = a.out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in items))
        files[f"{name}.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {
        name: dict(examples=len(items), functions=len({(e["project"], e["name"]) for e in items}),
                   groups=len({e["group"] for e in items}), targets=sum(len(e["targets"]) for e in items))
        for name, items in sorted(by_file.items())
    }
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    manifest = dict(
        script_commit=commit, args=vars(a) | {"matrix": str(a.matrix), "out": str(a.out)}, files=files, splits=summary,
        counters=dict(stats), gold_filter_drops=dict(drops),
        largest_groups=sorted(sizes.values(), reverse=True)[:5], groups_total=len(sizes),
        corpus_records_sha256={str(pth.parent.name): hashlib.sha256(pth.read_bytes()).hexdigest()
                               for pth in sorted(a.matrix.glob("ex-*/records.jsonl"))})
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    print(json.dumps({k: manifest[k] for k in ("splits", "counters", "gold_filter_drops", "largest_groups", "groups_total")}, indent=1))


if __name__ == "__main__":
    main()
