import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import eval_names as ev  # noqa: E402


def example(targets, gold, masked=False):
    return dict(id="isl:arm64:O0:0x10" + ("#masked" if masked else ""), function_name_masked=masked, opt="O0",
                targets=targets, completion=json.dumps({"renames": gold}))


class ParseAndScore(unittest.TestCase):
    def test_parse_takes_the_first_json_value_and_ignores_trailing_text(self):
        self.assertEqual(ev.parse_edits(' {"renames": {"param_1": "ctx"}}\nmore'), {"param_1": "ctx"})

    def test_parse_rejects_malformed_or_wrongly_typed_output(self):
        for text in ('{"renames": {"param_1": "ctx"', '{"renames": ["ctx"]}', '{"renames": {"param_1": 3}}',
                     '["renames"]', ""):
            self.assertIsNone(ev.parse_edits(text), text)

    def test_value_spans_point_at_the_new_names(self):
        text = '{"renames": {"param_1": "ctx", "local_30": "list"}}'
        spans = ev.value_spans(text)
        self.assertEqual({k: text[a:b] for k, (a, b) in spans.items()}, {"param_1": "ctx", "local_30": "list"})

    def test_span_logprob_sums_overlapping_tokens_only(self):
        offsets = [(0, 2), (2, 5), (5, 6)]
        self.assertAlmostEqual(ev.span_logprob((3, 5), offsets, [-1.0, -0.5, -2.0]), -0.5)
        self.assertAlmostEqual(ev.span_logprob((1, 6), offsets, [-1.0, -0.5, -2.0]), -3.5)

    def test_score_counts_missing_case_and_rejection(self):
        ex = example(["param_1", "param_2", "local_30"], {"param_1": "ctx", "param_2": "n", "local_30": "list"})
        rows = ev.score_example(ex, {"param_1": "Ctx", "local_30": "list"}, {"local_30": -0.1}, {"local_30"})
        by = {r["target"]: r for r in rows}
        self.assertEqual((by["param_1"]["exact"], by["param_1"]["exact_ci"]), (False, True))
        self.assertTrue(by["param_2"]["missing"])
        self.assertTrue(by["local_30"]["exact"] and by["local_30"]["rejected"])
        self.assertEqual(by["param_1"]["kind"], "param")

    def test_unparsed_output_makes_every_target_missing(self):
        rows = ev.score_example(example(["param_1"], {"param_1": "ctx"}), None, {}, set())
        self.assertTrue(rows[0]["missing"] and not rows[0]["exact"])

    def test_coverage_ranks_confident_first_and_missing_last(self):
        rows = [dict(exact=True, confidence=-0.1), dict(exact=False, confidence=-3.0),
                dict(exact=True, confidence=-0.2), dict(exact=False, confidence=None), dict(exact=True, confidence=-1.0)]
        curve = ev.coverage_curve(rows)
        self.assertEqual(curve["0.4"]["n"], 2)
        self.assertEqual(curve["0.4"]["exact"], 1.0)
        self.assertEqual(curve["1.0"]["exact"], 3 / 5)
        self.assertEqual(curve["0.6"]["min_logprob"], -1.0)


if __name__ == "__main__":
    unittest.main()
