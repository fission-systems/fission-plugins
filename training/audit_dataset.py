"""Audit a prepare_edits.py output: split leakage, near-duplicate overlap and label vocabulary.

Checks (all on the unmasked files): no group, function name or exact token-shape body shared between train and
val/test/xproj; for every eval function the highest 4-gram Jaccard similarity of their token shape with any
train function; and how many eval gold names also occur as a train gold name (an upper bound for a model that
cannot invent names). Writes audit.json next to the data.

usage: audit_dataset.py <dataset-dir>
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|0x[0-9a-fA-F]+|\d+|\S")
_KW = frozenset(
    "auto break case char const continue default do double else enum extern float for goto if inline int long "
    "register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while".split()
)


def shape(code: str) -> list[str]:
    out = []
    for tok in _TOKEN.findall(code):
        if tok[0].isalpha() or tok[0] == "_":
            out.append(tok if tok in _KW else "I")
        elif tok[0].isdigit():
            out.append("N")
        else:
            out.append(tok)
    return out


def shingles(tokens: list[str], n: int = 4) -> set[int]:
    return {hash(tuple(tokens[i : i + n])) for i in range(max(1, len(tokens) - n + 1))}


RARE = 400  # candidate retrieval ignores 4-grams found in more than this many train examples


def code_of(example: dict) -> str:
    return example["prompt"].split("```c\n", 1)[1].split("\n```", 1)[0]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    a = p.parse_args()
    train = load(a.dataset / "train.jsonl")
    train_unmasked = [e for e in train if not e["function_name_masked"]] or train
    names_train = {(e["project"], e["name"]) for e in train}
    groups_train = {e["group"] for e in train}
    shape_train = {hash(tuple(shape(code_of(e)))) for e in train_unmasked}
    gold_train = collections.Counter(v for e in train for v in json.loads(e["completion"])["renames"].values())

    index: dict[int, list[int]] = collections.defaultdict(list)
    sets = []
    for i, e in enumerate(train_unmasked):
        s = shingles(shape(code_of(e)))
        sets.append(s)
        for h in s:
            index[h].append(i)

    nearest: dict[str, float] = {}
    report: dict = {"train": dict(examples=len(train), functions=len(names_train), distinct_gold_names=len(gold_train))}
    for split in ("val", "test", "xproj"):
        ev = load(a.dataset / f"{split}.jsonl")
        shapes = {hash(tuple(shape(code_of(e)))) for e in ev}
        gold = [v for e in ev for v in json.loads(e["completion"])["renames"].values()]
        best = []
        for e in ev:
            s = shingles(shape(code_of(e)))
            votes = collections.Counter(j for h in s for j in index.get(h, ()) if len(index[h]) <= RARE)
            top = max((len(s & sets[j]) / len(s | sets[j]) for j, _ in votes.most_common(20)), default=0.0)
            best.append(top)
            nearest[e["id"]] = round(top, 3)
        report[split] = dict(
            examples=len(ev),
            functions=len({(e["project"], e["name"]) for e in ev}),
            shared_groups_with_train=len({e["group"] for e in ev} & groups_train),
            shared_function_names_with_train=len({(e["project"], e["name"]) for e in ev} & names_train),
            shared_exact_shapes_with_train=len(shapes & shape_train),
            nearest_train_jaccard_median=round(statistics.median(best), 3),
            nearest_train_jaccard_p90=round(sorted(best)[int(len(best) * 0.9)], 3),
            frac_jaccard_ge_0_8=round(sum(b >= 0.8 for b in best) / len(best), 3),
            gold_targets=len(gold),
            gold_name_seen_in_train=round(sum(g in gold_train for g in gold) / len(gold), 3),
            gold_name_is_train_top20=round(sum(g in dict(gold_train.most_common(20)) for g in gold) / len(gold), 3),
        )
    (a.dataset / "nearest_train_jaccard.json").write_text(json.dumps(nearest) + "\n")
    report["train_top_gold_names"] = gold_train.most_common(15)
    (a.dataset / "audit.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
