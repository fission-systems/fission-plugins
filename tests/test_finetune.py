import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import finetune as ft  # noqa: E402


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 90 + 1 for c in text]}


def ex(project, name, variant=0):
    return {"project": project, "name": name, "prompt": "ab\n", "completion": "{x}", "variant": variant}


class FineTuneData(unittest.TestCase):
    def test_function_subsets_are_nested_and_keep_all_variants(self):
        examples = [ex("p", f"f{i}", v) for i in range(30) for v in range(2)]
        small = ft.pick_functions(examples, 5, seed=1)
        large = ft.pick_functions(examples, 12, seed=1)
        small_keys = {ft.function_key(e) for e in small}
        large_keys = {ft.function_key(e) for e in large}
        self.assertEqual((len(small_keys), len(large_keys)), (5, 12))
        self.assertTrue(small_keys <= large_keys)
        self.assertEqual(len(small), 10)  # both variants of every chosen function
        self.assertEqual(len(ft.pick_functions(examples, None, seed=1)), 60)

    def test_subset_depends_on_seed_only(self):
        examples = [ex("p", f"f{i}") for i in range(30)]
        self.assertEqual(ft.pick_functions(examples, 8, 3), ft.pick_functions(examples, 8, 3))
        self.assertNotEqual(ft.pick_functions(examples, 8, 3), ft.pick_functions(examples, 8, 4))

    def test_loss_only_on_completion_and_eos(self):
        out = ft.encode(FakeTokenizer(), ex("p", "f"))
        n_prompt = len("ab\n")
        self.assertEqual(out["labels"][:n_prompt], [-100] * n_prompt)
        self.assertEqual(out["labels"][n_prompt:], out["input_ids"][n_prompt:])
        self.assertEqual(out["input_ids"][-1], 0)  # EOS is a target
        self.assertEqual(len(out["input_ids"]), n_prompt + len("{x}") + 1)

    def test_batches_cover_everything_once_and_respect_budget(self):
        lengths = [random.Random(2).randint(5, 300) for _ in range(5)] + list(range(10, 400, 7))
        batches = ft.make_batches(lengths, 1000, random.Random(0))
        flat = sorted(i for b in batches for i in b)
        self.assertEqual(flat, list(range(len(lengths))))
        for b in batches:
            if len(b) > 1:
                self.assertLessEqual(max(lengths[i] for i in b) * len(b), 1000)

    def test_collate_pads_right_and_masks(self):
        import torch

        items = [{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [8], "labels": [8]}]
        ids, labels, mask = ft.collate(items, pad_id=9, device="cpu")
        self.assertEqual(ids.tolist(), [[5, 6, 7], [8, 9, 9]])
        self.assertEqual(labels.tolist(), [[-100, 6, 7], [8, -100, -100]])
        self.assertEqual(mask.tolist(), [[1, 1, 1], [1, 0, 0]])
        self.assertIsInstance(ids, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
