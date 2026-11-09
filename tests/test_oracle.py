import unittest

from fission_plugins import EditList, FunctionRecord, Variable, apply_edits
from fission_plugins.oracle import compare, compile_assembly, find_clang

CLANG = find_clang()
CODE = """uint sum(uint * param_1, uint param_2)
{
    uint w8;
    uint x9;

    w8 = 0;
    for (x9 = 0; x9 < param_2; x9 = x9 + 1) {
        w8 = w8 + param_1[x9];
    }
    return w8 * 3;
}
"""


def record(code: str = CODE) -> FunctionRecord:
    names = [("param_1", "arg"), ("param_2", "arg"), ("w8", "reg"), ("x9", "reg")]
    return FunctionRecord("0x0", "sum", code, tuple(Variable(n, k, "uint") for n, k in names))


@unittest.skipUnless(CLANG, "clang not available")
class Oracle(unittest.TestCase):
    def test_rename_function_and_comment_keep_assembly(self):
        edits = EditList(function_name="add_all", renames={"param_1": "values", "param_2": "count", "w8": "total", "x9": "i"})
        result = apply_edits(record(), edits)
        self.assertEqual(result.rejected, [])
        self.assertEqual(compare(CODE, result.code, CLANG, {"sum": "add_all"}).kind, "identical")

    def test_changed_constant_is_flagged(self):
        self.assertEqual(compare(CODE, CODE.replace("* 3", "* 4"), CLANG).kind, "different")

    def test_unknown_types_and_globals_are_stubbed(self):
        code = "int use(Widget * param_1)\n{\n    return table_size + (int)param_1;\n}\n"
        compiled = compile_assembly(code, CLANG)
        self.assertTrue(compiled.ok, compiled.error)
        self.assertEqual(sorted(compiled.stubs), ["extern long long table_size;", "typedef long long Widget;"])

    def test_classifies_uncompilable_code(self):
        self.assertEqual(compare("int f( {", CODE, CLANG).kind, "baseline_uncompilable")
        self.assertEqual(compare(CODE, "int f( {", CLANG).kind, "edited_uncompilable")


if __name__ == "__main__":
    unittest.main()
