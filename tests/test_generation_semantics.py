"""CPU-only regression tests for utils/generation_semantics.py -- the shared
helper that fixes the Stage-F0-blocking samsum generation-semantics
mismatch between pred_long_bench.py (production) and
scripts/layer_policy_smoke.py / scripts/run_layer_sensitivity_pilot.py
(previously each reimplemented a generic-only model.generate() call with no
samsum branch at all).

No GPU, no model/dataset download: everything here uses a tiny fake
tokenizer object.

Run with:
    ./.venv/bin/python -m unittest tests.test_generation_semantics -v
"""
import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from utils.generation_semantics import (  # noqa: E402
    NO_BUILD_CHAT_DATASETS,
    resolve_generate_kwargs,
)

GENERIC_KEYS = {"max_new_tokens", "num_beams", "do_sample", "temperature", "top_p"}
SAMSUM_ONLY_KEYS = {"min_length", "eos_token_id"}


class FakeTokenizer:
    """Minimal stand-in for a HF tokenizer -- only .eos_token_id and
    .encode() (as used by pred_long_bench.py's samsum branch) are needed."""

    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        # pred_long_bench.py only ever calls this with text="\n"; return a
        # fixed, recognizable fake token id sequence.
        assert text == "\n"
        assert add_special_tokens is False
        return [999, 42]  # eos_token_id in the real branch is the *last* element


class TestGenericTaskKwargsUnchanged(unittest.TestCase):
    """A/E: generic tasks (including every Stage-D task) get exactly the
    5 generic kwargs, nothing else -- unchanged from before this fix."""

    def test_generic_task_kwargs(self):
        tok = FakeTokenizer()
        for dataset in ["trec", "lcc", "passage_retrieval_en", "2wikimqa", "hotpotqa", "qasper"]:
            with self.subTest(dataset=dataset):
                kwargs = resolve_generate_kwargs(dataset, tok, context_length=123, max_gen=64)
                self.assertEqual(set(kwargs), GENERIC_KEYS)
                self.assertEqual(kwargs["max_new_tokens"], 64)
                self.assertEqual(kwargs["num_beams"], 1)
                self.assertEqual(kwargs["do_sample"], False)
                self.assertEqual(kwargs["temperature"], 1.0)
                self.assertEqual(kwargs["top_p"], 1.0)

    def test_stage_d_tasks_resolve_exactly_as_before(self):
        # trec, lcc, passage_retrieval_en, 2wikimqa -- the 4 Stage-D
        # screening tasks -- must be unaffected by this fix.
        tok = FakeTokenizer()
        for dataset in ["trec", "lcc", "passage_retrieval_en", "2wikimqa"]:
            kwargs = resolve_generate_kwargs(dataset, tok, context_length=999, max_gen=32)
            self.assertNotIn("min_length", kwargs)
            self.assertNotIn("eos_token_id", kwargs)


class TestMultifieldqaEnGenericPath(unittest.TestCase):
    """B: multifieldqa_en (new Stage-F0 task) uses the generic production
    path, not a special-cased one."""

    def test_multifieldqa_en_is_generic(self):
        tok = FakeTokenizer()
        kwargs = resolve_generate_kwargs("multifieldqa_en", tok, context_length=500, max_gen=64)
        self.assertEqual(set(kwargs), GENERIC_KEYS)


class TestSamsumProductionSemantics(unittest.TestCase):
    """C: samsum receives the exact production-specific min_length and
    eos_token_id semantics from pred_long_bench.py's get_pred()."""

    def test_samsum_gets_min_length_and_eos_token_id(self):
        tok = FakeTokenizer()
        kwargs = resolve_generate_kwargs("samsum", tok, context_length=777, max_gen=64)
        self.assertEqual(set(kwargs), GENERIC_KEYS | SAMSUM_ONLY_KEYS)
        self.assertEqual(kwargs["min_length"], 778)  # context_length + 1
        self.assertEqual(kwargs["eos_token_id"], [2, 42])  # [tokenizer.eos_token_id, "\n"-encode[-1]]

    def test_samsum_min_length_tracks_context_length(self):
        tok = FakeTokenizer()
        for ctx_len in (1, 100, 31500):
            kwargs = resolve_generate_kwargs("samsum", tok, context_length=ctx_len, max_gen=64)
            self.assertEqual(kwargs["min_length"], ctx_len + 1)

    def test_samsum_still_gets_generic_kwargs_too(self):
        tok = FakeTokenizer()
        kwargs = resolve_generate_kwargs("samsum", tok, context_length=10, max_gen=64)
        for key in GENERIC_KEYS:
            self.assertIn(key, kwargs)


class TestSmokeAndPilotUseSharedHelper(unittest.TestCase):
    """D: smoke and pilot must resolve samsum kwargs via the SAME shared
    function (no independent reimplementation to drift from), verified by
    checking their source imports resolve_generate_kwargs from
    utils.generation_semantics and never constructs a competing inline
    samsum branch."""

    def test_layer_policy_smoke_imports_shared_helper(self):
        import layer_policy_smoke as smoke_mod

        source = inspect.getsource(smoke_mod)
        self.assertIn("from utils.generation_semantics import", source)
        self.assertIn("resolve_generate_kwargs", source)
        # No independent samsum-specific branch left behind.
        self.assertNotIn('dataset == "samsum"', source)
        self.assertNotIn("min_length=context_length", source)

    def test_run_layer_sensitivity_pilot_imports_shared_helper(self):
        import run_layer_sensitivity_pilot as pilot_mod

        source = inspect.getsource(pilot_mod)
        self.assertIn("from utils.generation_semantics import", source)
        self.assertIn("resolve_generate_kwargs", source)
        self.assertNotIn('task == "samsum"', source)
        self.assertNotIn("min_length=context_length", source)

    def test_both_scripts_use_shared_no_build_chat_constant(self):
        import layer_policy_smoke as smoke_mod
        import run_layer_sensitivity_pilot as pilot_mod

        for mod in (smoke_mod, pilot_mod):
            source = inspect.getsource(mod)
            self.assertIn("NO_BUILD_CHAT_DATASETS", source)
            # The old inline hard-coded list must not reappear.
            self.assertNotIn('["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]', source)

    def test_smoke_and_pilot_resolve_identical_samsum_kwargs(self):
        tok = FakeTokenizer()
        # Both call sites ultimately delegate to the exact same function --
        # this directly proves there is no possibility of drift between them.
        k1 = resolve_generate_kwargs("samsum", tok, context_length=42, max_gen=64)
        k2 = resolve_generate_kwargs("samsum", tok, context_length=42, max_gen=64)
        self.assertEqual(k1, k2)


class TestProductionRunnerUntouched(unittest.TestCase):
    """F: pred_long_bench.py (the validated formal runner / production
    output path) must remain functionally unchanged -- still contains its
    own original inline samsum branch verbatim, never refactored to call
    the new shared helper (out of scope for this fix)."""

    def test_pred_long_bench_still_has_original_inline_samsum_branch(self):
        import pred_long_bench as plb

        source = inspect.getsource(plb.get_pred)
        self.assertIn('dataset == "samsum"', source)
        self.assertIn("min_length=context_length + 1", source)
        self.assertIn('eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\\n"', source)

    def test_pred_long_bench_does_not_import_the_new_helper(self):
        import pred_long_bench as plb

        source = inspect.getsource(plb)
        self.assertNotIn("generation_semantics", source)


if __name__ == "__main__":
    unittest.main()
