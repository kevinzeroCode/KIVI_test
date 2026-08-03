"""CPU-only unit tests for mixed K/V quantization infrastructure.

Covers sections A-D of the mixed-kv-ablation smoke-test plan:
  A. loader decision (which model class a bit config uses)
  B. output directory naming (no collisions, legacy names preserved)
  C. bit validation (2/4/16 allowed, everything else fails closed)
  D. run_config.json resume metadata (accept matching config, reject
     mismatches, fail closed on unverified mixed-bit directories)

No GPU, no model download, no dataset download required: this only exercises
the pure Python decision/naming/metadata functions in pred_long_bench.py.

Run with:
    ./.venv/bin/python -m unittest tests.test_mixed_kv_unit -v
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pred_long_bench as plb


class FakeModelArgs:
    def __init__(self, k_bits, v_bits, model_name_or_path="lmsys/longchat-7b-v1.5-32k",
                 group_size=32, residual_length=128):
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.model_name_or_path = model_name_or_path
        self.group_size = group_size
        self.residual_length = residual_length


class TestLoaderDecision(unittest.TestCase):
    """Section A: only 16/16 must use the plain HF model."""

    def test_symmetric_and_mixed_configs(self):
        cases = [
            ((16, 16), (False, False, False)),
            ((2, 2), (True, True, True)),
            ((4, 4), (True, True, True)),
            ((2, 16), (True, False, True)),
            ((16, 2), (False, True, True)),
            ((2, 4), (True, True, True)),
            ((4, 2), (True, True, True)),
        ]
        for (k_bits, v_bits), expected in cases:
            with self.subTest(k_bits=k_bits, v_bits=v_bits):
                self.assertEqual(plb.loader_decision(k_bits, v_bits), expected)

    def test_only_16_16_uses_hf_model(self):
        for k_bits, v_bits in [(16, 16)]:
            _, _, use_kivi_model = plb.loader_decision(k_bits, v_bits)
            self.assertFalse(use_kivi_model, f"{k_bits}/{v_bits} should use the plain HF model")
        for k_bits, v_bits in [(2, 2), (4, 4), (2, 16), (16, 2), (2, 4), (4, 2)]:
            _, _, use_kivi_model = plb.loader_decision(k_bits, v_bits)
            self.assertTrue(use_kivi_model, f"{k_bits}/{v_bits} should use LlamaForCausalLM_KIVI")


class TestOutputNaming(unittest.TestCase):
    """Section B: symmetric configs keep legacy names; mixed configs never collide."""

    def test_symmetric_configs_keep_legacy_format(self):
        self.assertEqual(
            plb.build_pred_dir_name("longchat-7b-v1.5-32k", 31500, 16, 16, 32, 128),
            "longchat-7b-v1.5-32k_31500_16bits_group32_residual128",
        )
        self.assertEqual(
            plb.build_pred_dir_name("longchat-7b-v1.5-32k", 31500, 2, 2, 32, 128),
            "longchat-7b-v1.5-32k_31500_2bits_group32_residual128",
        )
        self.assertEqual(
            plb.build_pred_dir_name("longchat-7b-v1.5-32k", 31500, 4, 4, 32, 128),
            "longchat-7b-v1.5-32k_31500_4bits_group32_residual128",
        )

    def test_mixed_configs_encode_both_bit_widths(self):
        self.assertEqual(
            plb.build_pred_dir_name("longchat-7b-v1.5-32k", 31500, 2, 16, 32, 128),
            "longchat-7b-v1.5-32k_31500_k2_v16_group32_residual128",
        )
        self.assertEqual(
            plb.build_pred_dir_name("longchat-7b-v1.5-32k", 31500, 16, 2, 32, 128),
            "longchat-7b-v1.5-32k_31500_k16_v2_group32_residual128",
        )

    def test_five_primary_configs_are_all_distinct(self):
        configs = [(16, 16), (2, 2), (4, 4), (2, 16), (16, 2)]
        names = [plb.build_pred_dir_name("longchat-7b-v1.5-32k", 31500, k, v, 32, 128) for k, v in configs]
        self.assertEqual(len(names), len(set(names)), f"collision among {names}")

    def test_all_seven_configs_are_distinct(self):
        configs = [(16, 16), (2, 2), (4, 4), (2, 16), (16, 2), (2, 4), (4, 2)]
        names = [plb.build_pred_dir_name("longchat-7b-v1.5-32k", 31500, k, v, 32, 128) for k, v in configs]
        self.assertEqual(len(names), len(set(names)), f"collision among {names}")

    def test_mixed_does_not_collide_with_symmetric(self):
        k2_v16 = plb.build_pred_dir_name("m", 100, 2, 16, 32, 128)
        k2_v2 = plb.build_pred_dir_name("m", 100, 2, 2, 32, 128)
        k16_v2 = plb.build_pred_dir_name("m", 100, 16, 2, 32, 128)
        k16_v16 = plb.build_pred_dir_name("m", 100, 16, 16, 32, 128)
        self.assertNotEqual(k2_v16, k2_v2)
        self.assertNotEqual(k16_v2, k16_v16)

    def test_existing_baseline_directory_names_unchanged(self):
        # These three strings are the actual committed baseline directory
        # names under pred/ as of commit 67591f4. If this test ever fails,
        # the naming function has silently changed what an existing baseline
        # run would be called.
        self.assertEqual(
            plb.build_pred_dir_name("longchat-7b-v1.5-32k", 31500, 16, 16, 32, 128),
            "longchat-7b-v1.5-32k_31500_16bits_group32_residual128",
        )
        self.assertEqual(
            plb.build_pred_dir_name("longchat-7b-v1.5-32k", 31500, 2, 2, 32, 128),
            "longchat-7b-v1.5-32k_31500_2bits_group32_residual128",
        )
        self.assertEqual(
            plb.build_pred_dir_name("longchat-7b-v1.5-32k", 31500, 4, 4, 32, 128),
            "longchat-7b-v1.5-32k_31500_4bits_group32_residual128",
        )


class TestBitValidation(unittest.TestCase):
    """Section C: 2/4/16 pass, everything else fails closed before model load."""

    def test_allowed_bits_pass(self):
        for k_bits in (2, 4, 16):
            for v_bits in (2, 4, 16):
                with self.subTest(k_bits=k_bits, v_bits=v_bits):
                    plb.validate_bits(k_bits, v_bits)  # must not raise

    def test_disallowed_bits_fail(self):
        for bad in (3, 8, 12):
            with self.subTest(k_bits=bad):
                with self.assertRaises(ValueError):
                    plb.validate_bits(bad, 2)
            with self.subTest(v_bits=bad):
                with self.assertRaises(ValueError):
                    plb.validate_bits(2, bad)

    def test_error_message_includes_actual_values(self):
        with self.assertRaises(ValueError) as ctx:
            plb.validate_bits(3, 8)
        message = str(ctx.exception)
        self.assertIn("k_bits=3", message)
        self.assertIn("v_bits=8", message)


class TestResumeMetadata(unittest.TestCase):
    """Section D: run_config.json guards resume/skip behavior."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="kivi_resume_test_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_config(self, **overrides):
        base = dict(
            model_name_or_path="lmsys/longchat-7b-v1.5-32k",
            k_bits=2,
            v_bits=16,
            group_size=32,
            residual_length=128,
            max_length=31500,
            seed=42,
            model_class="LlamaForCausalLM_KIVI",
            quantize_key=True,
            quantize_value=False,
            git_commit="deadbeef",
            transformers_version="4.43.4",
        )
        base.update(overrides)
        return base

    def test_fresh_directory_is_created_with_metadata(self):
        pred_dir = os.path.join(self.tmpdir, "fresh")
        run_config = self._run_config()
        status = plb.prepare_run_directory(pred_dir, run_config)
        self.assertEqual(status, "created")
        self.assertTrue(os.path.isdir(pred_dir))
        written = plb.load_run_config(pred_dir)
        self.assertEqual(written, run_config)

    def test_identical_config_resumes(self):
        pred_dir = os.path.join(self.tmpdir, "resume_ok")
        run_config = self._run_config()
        plb.prepare_run_directory(pred_dir, run_config)
        status = plb.prepare_run_directory(pred_dir, dict(run_config))
        self.assertEqual(status, "validated")

    def test_different_k_bits_rejected(self):
        pred_dir = os.path.join(self.tmpdir, "bad_k")
        plb.prepare_run_directory(pred_dir, self._run_config(k_bits=2))
        with self.assertRaises(RuntimeError):
            plb.prepare_run_directory(pred_dir, self._run_config(k_bits=4))

    def test_different_v_bits_rejected(self):
        pred_dir = os.path.join(self.tmpdir, "bad_v")
        plb.prepare_run_directory(pred_dir, self._run_config(v_bits=16))
        with self.assertRaises(RuntimeError):
            plb.prepare_run_directory(pred_dir, self._run_config(v_bits=2))

    def test_different_model_rejected(self):
        pred_dir = os.path.join(self.tmpdir, "bad_model")
        plb.prepare_run_directory(pred_dir, self._run_config(model_name_or_path="a/model-one"))
        with self.assertRaises(RuntimeError):
            plb.prepare_run_directory(pred_dir, self._run_config(model_name_or_path="a/model-two"))

    def test_different_group_size_rejected(self):
        pred_dir = os.path.join(self.tmpdir, "bad_group")
        plb.prepare_run_directory(pred_dir, self._run_config(group_size=32))
        with self.assertRaises(RuntimeError):
            plb.prepare_run_directory(pred_dir, self._run_config(group_size=64))

    def test_different_residual_length_rejected(self):
        pred_dir = os.path.join(self.tmpdir, "bad_residual")
        plb.prepare_run_directory(pred_dir, self._run_config(residual_length=128))
        with self.assertRaises(RuntimeError):
            plb.prepare_run_directory(pred_dir, self._run_config(residual_length=64))

    def test_legacy_symmetric_directory_without_metadata_is_left_alone(self):
        # Simulates an existing baseline dir (e.g. the real 2/2 KIVI-2 dir)
        # that predates run_config.json and must not be touched.
        pred_dir = os.path.join(self.tmpdir, "legacy_symmetric")
        os.makedirs(pred_dir)
        with open(os.path.join(pred_dir, "narrativeqa.jsonl"), "w") as f:
            f.write('{"pred": "x"}\n')
        run_config = self._run_config(k_bits=2, v_bits=2)
        status = plb.prepare_run_directory(pred_dir, run_config)
        self.assertEqual(status, "legacy_no_metadata")
        # must NOT have written a run_config.json into the legacy dir
        self.assertIsNone(plb.load_run_config(pred_dir))
        # must NOT have touched the existing prediction file
        with open(os.path.join(pred_dir, "narrativeqa.jsonl")) as f:
            self.assertEqual(f.read(), '{"pred": "x"}\n')

    def test_mixed_bit_directory_without_metadata_fails_closed(self):
        # A mixed-bit-named directory can never legitimately predate this
        # feature (mixed bits were unsupported before), so an existing
        # mixed-bit directory with no run_config.json must be refused.
        pred_dir = os.path.join(self.tmpdir, "mixed_no_metadata")
        os.makedirs(pred_dir)
        with open(os.path.join(pred_dir, "narrativeqa.jsonl"), "w") as f:
            f.write('{"pred": "x"}\n')
        run_config = self._run_config(k_bits=2, v_bits=16)
        with self.assertRaises(RuntimeError):
            plb.prepare_run_directory(pred_dir, run_config)


if __name__ == "__main__":
    unittest.main()
