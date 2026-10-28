import json
import unittest
from pathlib import Path

from fission_plugins import Comment, EditList, FunctionRecord, Variable, apply_edits
from fission_plugins.apply import _rewrite

FIXTURE = Path(__file__).parent / "fixtures" / "checksum.fission.json"


def record() -> FunctionRecord:
    return FunctionRecord.from_fission_json(json.loads(FIXTURE.read_text()))


class ApplyEdits(unittest.TestCase):
    def test_renames_variables_and_function(self):
        r = record()
        res = apply_edits(r, EditList(function_name="hash_bytes",
                                      renames={"param_1": "buf", "param_2": "len", "w8": "hash"}))
        self.assertEqual(res.rejected, [])
        self.assertIn("int hash_bytes(uchar * buf, int len)", res.code)
        self.assertNotIn("param_1", res.code)
        self.assertIn("hash = 0;", res.code)

    def test_rename_is_alpha_equivalent(self):
        r = record()
        mapping = {"param_1": "buf", "param_2": "len", "w8": "hash"}
        res = apply_edits(r, EditList(renames=mapping))
        back = _rewrite(res.code, {v: k for k, v in mapping.items()})
        self.assertEqual(back, r.code)

    def test_rejects_unsafe_names(self):
        r = record()
        res = apply_edits(r, EditList(renames={"param_1": "int", "param_2": "x0", "w8": "9a",
                                               "x9": "n", "x10": "n", "ghost": "y"}))
        reasons = dict(res.rejected)
        self.assertEqual(reasons["param_1->int"], "C keyword or builtin type")
        self.assertEqual(reasons["param_2->x0"], "collides with an existing identifier")
        self.assertEqual(reasons["w8->9a"], "not a C identifier")
        self.assertEqual(reasons["x10->n"], "duplicate new name")
        self.assertEqual(reasons["ghost->y"], "not a variable Fission reported")
        self.assertEqual(res.applied_renames, {"x9": "n"})

    def test_leaves_literals_comments_and_members_alone(self):
        r = FunctionRecord("0x0", "f", 'int f(int a) { // a\n  s.a = a; puts("a"); return \'a\' + p->a; }',
                           (Variable("a", "arg", "int", 0),))
        res = apply_edits(r, EditList(renames={"a": "count"}))
        self.assertEqual(res.code, 'int f(int count) { // a\n  s.a = count; puts("a"); '
                                   "return 'a' + p->a; }")

    def test_comment_is_sanitised(self):
        r = record()
        res = apply_edits(r, EditList(comments=[Comment(3, "rolling */ hash"), Comment(999, "x")]))
        self.assertIn("/* rolling  hash */", res.code.split("\n")[2])
        self.assertEqual(res.rejected, [("comment@999", "line out of range")])

    def test_failed_rows_are_not_records(self):
        with self.assertRaises(ValueError):
            FunctionRecord.from_fission_json({"name": "__mh_execute_header", "error": "unable to read bytes"})


if __name__ == "__main__":
    unittest.main()
