import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import prepare_edits as pe  # noqa: E402


def row(name="f", code=None, variables=None):
    code = code or f"int {name}(uchar * param_1, int param_2)\n{{\n    int w8;\n    w8 = param_2 + 1;\n    return w8;\n}}\n"
    pseudo_vars = [
        {"name": "param_1", "kind": "arg", "type": "uchar *", "arg_index": 0},
        {"name": "param_2", "kind": "arg", "type": "int", "arg_index": 1},
        {"name": "w8", "kind": "reg", "type": "int"},
    ]
    dwarf = variables or [
        {"name": "buf", "tag": "param", "outcome": "unique", "fission": [{"name": "param_1"}]},
        {"name": "count", "tag": "param", "outcome": "unique", "fission": [{"name": "param_2"}]},
        {"name": "tmp", "tag": "local", "outcome": "shared", "fission": [{"name": "w8"}]},
    ]
    return {"id": f"isl:arm64:O0:{name}", "project": "isl", "license": "MIT", "arch": "arm64", "opt": "O0",
            "address": "0x40", "name": name, "variables": dwarf,
            "views": {"pseudo": {"code": code, "engine": "x", "variables": pseudo_vars}}}


class PrepareEdits(unittest.TestCase):
    def test_body_hash_ignores_identifiers_and_numbers_only(self):
        a = "int f(int a)\n{\n    return a + 1;\n}\n"
        b = "int g(int zz)\n{\n    return zz + 7;\n}\n"
        c = "int f(int a)\n{\n    return a - 1;\n}\n"
        self.assertEqual(pe.body_hash(a), pe.body_hash(b))
        self.assertNotEqual(pe.body_hash(a), pe.body_hash(c))

    def test_only_unique_variables_become_targets(self):
        renames, drops = pe.gold_edits(row())
        self.assertEqual(renames, {"param_1": "buf", "param_2": "count"})
        self.assertEqual(drops["not_unique_or_unnamed"], 1)

    def test_drops_duplicate_new_names_and_applier_rejections(self):
        vs = [
            {"name": "x", "tag": "param", "outcome": "unique", "fission": [{"name": "param_1"}]},
            {"name": "x", "tag": "local", "outcome": "unique", "fission": [{"name": "w8"}]},
            {"name": "int", "tag": "local", "outcome": "unique", "fission": [{"name": "param_2"}]},
        ]
        renames, drops = pe.gold_edits(row(variables=vs))
        self.assertEqual(renames, {"param_1": "x"})
        self.assertEqual(drops["duplicate_new_name"], 1)
        self.assertEqual(drops["invalid_new_name"], 1)

    def test_prompt_masks_own_function_name_only_when_asked(self):
        r = row(name="isl_secret")
        self.assertIn("isl_secret", pe.build_prompt(r, ["param_1"], masked=False))
        masked = pe.build_prompt(r, ["param_1"], masked=True)
        self.assertNotIn("isl_secret", masked)
        self.assertIn("FUN_00000040", masked)

    def test_completion_is_parseable_edit_list(self):
        text = pe.completion_for({"param_1": "buf", "param_2": "count"}, ["param_2", "param_1"])
        self.assertEqual(json.loads(text), {"renames": {"param_2": "count", "param_1": "buf"}})
        self.assertEqual(list(json.loads(text)["renames"]), ["param_2", "param_1"])

    def test_split_assignment_is_deterministic_and_keeps_big_groups_in_train(self):
        comps = [f"c{i}" for i in range(400)] + ["big"]
        sizes = {c: 1 for c in comps} | {"big": 500}
        one = pe.assign_splits(comps, sizes, seed=1, big_group=100, val=0.1, test=0.1)
        two = pe.assign_splits(comps, sizes, seed=1, big_group=100, val=0.1, test=0.1)
        self.assertEqual(one, two)
        self.assertEqual(one["big"], "train")
        share = {s: sum(v == s for v in one.values()) / len(one) for s in ("train", "val", "test")}
        self.assertTrue(0.05 < share["val"] < 0.15 and 0.05 < share["test"] < 0.15, share)


if __name__ == "__main__":
    unittest.main()
