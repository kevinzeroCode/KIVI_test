"""CPU-only unit tests for analysis/analyze_kv_ablation.py (3x3 K/V matrix).

Does not load a model, touch the GPU, or read any prediction JSONL from the
real pred/ directories -- everything here operates on small synthetic
in-memory/temp-dir fixtures.

Run with:
    ./.venv/bin/python -m unittest tests.test_kv_ablation_analysis -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.analyze_kv_ablation import (  # noqa: E402
    ALL_CONFIGS,
    FP16,
    ConfigDiscoveryError,
    PairingError,
    apply_contrast,
    build_key_trajectory_contrasts,
    build_value_trajectory_contrasts,
    compute_task_sample_scores,
    derive_seed,
    discover_configurations,
    interaction_contrast,
    bootstrap_overall_scores,
    leave_one_task_out,
    sample_score,
    validate_pairing,
)
from eval_long_bench import dataset2metric, scorer  # noqa: E402


def _write_run_config(dir_path, k_bits, v_bits):
    cfg = {
        "model_name_or_path": "lmsys/longchat-7b-v1.5-32k",
        "k_bits": k_bits,
        "v_bits": v_bits,
        "group_size": 32,
        "residual_length": 128,
        "max_length": 31500,
        "seed": 42,
        "model_class": "LlamaForCausalLM_KIVI",
        "quantize_key": k_bits != 16,
        "quantize_value": v_bits != 16,
    }
    (dir_path / "run_config.json").write_text(json.dumps(cfg), encoding="utf-8")


class TestConfigDiscovery(unittest.TestCase):
    """Configuration discovery must resolve all 9 (k,v) cells from either
    run_config.json (new-style dirs) or the legacy `_<N>bits_` directory
    name (old-style dirs, no run_config.json), and must never hard-code a
    specific directory-name string per configuration."""

    def test_discovers_all_nine_from_mixed_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 6 new-style dirs (run_config.json-driven), arbitrary names.
            new_style = [(16, 4), (16, 2), (4, 16), (4, 2), (2, 16), (2, 4)]
            for k, v in new_style:
                d = root / f"some_run_name_k{k}_v{v}"
                d.mkdir()
                _write_run_config(d, k, v)
            # 3 legacy-style dirs (no run_config.json).
            for name, bits in [
                ("longchat-7b-v1.5-32k_31500_16bits_group32_residual128", 16),
                ("longchat-7b-v1.5-32k_31500_4bits_group32_residual128", 4),
                ("longchat-7b-v1.5-32k_31500_2bits_group32_residual128", 2),
            ]:
                (root / name).mkdir()

            configs, log = discover_configurations(root)
            self.assertEqual(set(configs.keys()), set(ALL_CONFIGS))
            self.assertEqual(len(log), 9)

    def test_missing_configuration_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for k, v in [(16, 16), (16, 4)]:  # only 2 of 9
                d = root / f"run_k{k}_v{v}"
                d.mkdir()
                _write_run_config(d, k, v)
            with self.assertRaises(ConfigDiscoveryError):
                discover_configurations(root)

    def test_duplicate_configuration_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d1 = root / "run_a"
            d2 = root / "run_b"
            d1.mkdir()
            d2.mkdir()
            _write_run_config(d1, 16, 16)
            _write_run_config(d2, 16, 16)
            with self.assertRaises(ConfigDiscoveryError):
                discover_configurations(root)

    def test_mismatched_run_settings_are_skipped_not_matched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / "wrong_length_run"
            d.mkdir()
            cfg = {
                "model_name_or_path": "lmsys/longchat-7b-v1.5-32k",
                "k_bits": 16, "v_bits": 16,
                "group_size": 32, "residual_length": 128,
                "max_length": 4000,  # wrong -- must not be resolved as K16/V16
                "seed": 42,
            }
            (d / "run_config.json").write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaises(ConfigDiscoveryError):
                discover_configurations(root)


class TestSampleScorerMatchesEvalScorer(unittest.TestCase):
    """The per-sample scorer must reproduce eval_long_bench.scorer()'s
    behavior exactly, including its per-file (not per-row) all_classes and
    its `100 * mean` / round(..., 2) aggregation."""

    def test_qa_f1_task_matches_official_scorer(self):
        predictions = ["the cat sat on the mat", "completely unrelated text"]
        answers = [["the cat sat on the mat"], ["something else"]]
        all_classes = []

        official = scorer("qasper", predictions, answers, all_classes)

        rows = [
            {"pred": p, "answers": a, "all_classes": all_classes, "length": 100}
            for p, a in zip(predictions, answers)
        ]
        recomputed = round(float(compute_task_sample_scores("qasper", rows).mean()), 2)

        self.assertEqual(official, recomputed)

    def test_classification_task_matches_official_scorer(self):
        predictions = ["This is clearly a SPORTS story", "no idea what this is"]
        answers = [["SPORTS"], ["NEWS"]]
        all_classes = ["SPORTS", "NEWS", "WEATHER"]

        official = scorer("trec", predictions, answers, all_classes)

        rows = [
            {"pred": p, "answers": a, "all_classes": all_classes, "length": 100}
            for p, a in zip(predictions, answers)
        ]
        recomputed = round(float(compute_task_sample_scores("trec", rows).mean()), 2)

        self.assertEqual(official, recomputed)


class TestFirstLinePreprocessing(unittest.TestCase):
    def test_first_line_only_applied_for_trec(self):
        prediction = "SPORTS\nirrelevant trailing hallucinated text"
        ground_truths = ["SPORTS"]
        all_classes = ["SPORTS", "NEWS"]

        score = sample_score("trec", prediction, ground_truths, all_classes)
        self.assertEqual(score, dataset2metric["trec"]("SPORTS", "SPORTS", all_classes=all_classes))

    def test_first_line_not_applied_for_qasper(self):
        prediction = "answer line one\nanswer line two"
        ground_truths = ["answer line one\nanswer line two"]

        score_full = sample_score("qasper", prediction, ground_truths, [])
        score_truncated = dataset2metric["qasper"]("answer line one", ground_truths[0], all_classes=[])

        self.assertGreater(score_full, score_truncated)


class TestPairingValidation(unittest.TestCase):
    def _make_all_data(self, mutate=None):
        tasks = ["qasper", "trec"]
        base = {
            task: [
                {"answers": ["a1"], "all_classes": [], "length": 10, "pred": "x"},
                {"answers": ["a2"], "all_classes": [], "length": 20, "pred": "y"},
            ]
            for task in tasks
        }
        methods = {}
        for cfg in ALL_CONFIGS:
            methods[cfg] = {task: [dict(row) for row in rows] for task, rows in base.items()}
        if mutate:
            mutate(methods)
        return methods

    def test_pairing_passes_on_identical_metadata(self):
        methods = self._make_all_data()
        import analysis.analyze_kv_ablation as mod

        original_tasks = mod.EXPECTED_TASKS
        mod.EXPECTED_TASKS = ["qasper", "trec"]
        try:
            result = validate_pairing(methods)
            self.assertEqual(result["status"], "PASS")
        finally:
            mod.EXPECTED_TASKS = original_tasks

    def test_pairing_fails_on_mismatched_answers(self):
        def mutate(methods):
            methods[(2, 2)]["qasper"][1]["answers"] = ["different answer"]

        methods = self._make_all_data(mutate=mutate)
        import analysis.analyze_kv_ablation as mod

        original_tasks = mod.EXPECTED_TASKS
        mod.EXPECTED_TASKS = ["qasper", "trec"]
        try:
            with self.assertRaises(PairingError):
                validate_pairing(methods)
        finally:
            mod.EXPECTED_TASKS = original_tasks

    def test_pairing_fails_on_row_count_mismatch(self):
        def mutate(methods):
            methods[(4, 4)]["trec"].append(
                {"answers": ["extra"], "all_classes": [], "length": 5, "pred": "z"}
            )

        methods = self._make_all_data(mutate=mutate)
        import analysis.analyze_kv_ablation as mod

        original_tasks = mod.EXPECTED_TASKS
        mod.EXPECTED_TASKS = ["qasper", "trec"]
        try:
            with self.assertRaises(PairingError):
                validate_pairing(methods)
        finally:
            mod.EXPECTED_TASKS = original_tasks

    def test_pairing_never_reorders_rows(self):
        """A same-multiset-different-order answers list must be treated as a
        mismatch, not silently realigned."""
        def mutate(methods):
            methods[(2, 16)]["qasper"] = list(reversed(methods[(2, 16)]["qasper"]))

        methods = self._make_all_data(mutate=mutate)
        import analysis.analyze_kv_ablation as mod

        original_tasks = mod.EXPECTED_TASKS
        mod.EXPECTED_TASKS = ["qasper", "trec"]
        try:
            with self.assertRaises(PairingError):
                validate_pairing(methods)
        finally:
            mod.EXPECTED_TASKS = original_tasks


class TestTrajectorySignConvention(unittest.TestCase):
    """delta = score(destination) - score(source); negative = degradation."""

    def test_key_trajectory_sign(self):
        contrasts = build_key_trajectory_contrasts()
        score_map = {cfg: 0.0 for cfg in ALL_CONFIGS}
        score_map[(16, 16)] = 100.0
        score_map[(4, 16)] = 90.0  # degraded by 10 relative to K16
        score_map[(2, 16)] = 70.0  # degraded further

        self.assertAlmostEqual(apply_contrast(score_map, contrasts["K16->K4 @ V16"]), -10.0)
        self.assertAlmostEqual(apply_contrast(score_map, contrasts["K4->K2 @ V16"]), -20.0)
        self.assertAlmostEqual(apply_contrast(score_map, contrasts["K16->K2 @ V16"]), -30.0)

    def test_value_trajectory_sign(self):
        contrasts = build_value_trajectory_contrasts()
        score_map = {cfg: 0.0 for cfg in ALL_CONFIGS}
        score_map[(16, 16)] = 100.0
        score_map[(16, 4)] = 105.0  # an *improvement* -> positive delta
        score_map[(16, 2)] = 95.0

        self.assertAlmostEqual(apply_contrast(score_map, contrasts["V16->V4 @ K16"]), 5.0)
        self.assertAlmostEqual(apply_contrast(score_map, contrasts["V4->V2 @ K16"]), -10.0)
        self.assertAlmostEqual(apply_contrast(score_map, contrasts["V16->V2 @ K16"]), -5.0)


class TestInteractionFormula(unittest.TestCase):
    def test_interaction_matches_definition(self):
        score_map = {cfg: 0.0 for cfg in ALL_CONFIGS}
        score_map[(16, 16)] = 100.0
        score_map[(4, 16)] = 90.0   # K-only effect at V16 = -10
        score_map[(16, 4)] = 85.0   # V-only effect at K16 = -15
        score_map[(4, 4)] = 60.0    # joint = -40

        contrast = interaction_contrast(k_source=16, k_dest=4, v_source=16, v_dest=4)
        interaction = apply_contrast(score_map, contrast)
        # [S(4,4)-S(16,4)] - [S(4,16)-S(16,16)] = (60-85) - (90-100) = -25 - (-10) = -15
        self.assertAlmostEqual(interaction, -15.0, places=6)

    def test_interaction_zero_when_effects_are_additive(self):
        score_map = {cfg: 0.0 for cfg in ALL_CONFIGS}
        score_map[(16, 16)] = 100.0
        score_map[(4, 16)] = 90.0   # -10
        score_map[(16, 4)] = 95.0   # -5
        score_map[(4, 4)] = 85.0    # -15 == -10 + -5, purely additive

        contrast = interaction_contrast(k_source=16, k_dest=4, v_source=16, v_dest=4)
        interaction = apply_contrast(score_map, contrast)
        self.assertAlmostEqual(interaction, 0.0, places=6)


class TestBootstrapSeedReproducibility(unittest.TestCase):
    def test_derive_seed_is_deterministic(self):
        s1 = derive_seed(42, "overall", "qasper")
        s2 = derive_seed(42, "overall", "qasper")
        self.assertEqual(s1, s2)

    def test_derive_seed_differs_across_inputs(self):
        s1 = derive_seed(42, "overall", "qasper")
        s2 = derive_seed(42, "overall", "trec")
        s3 = derive_seed(43, "overall", "qasper")
        self.assertNotEqual(s1, s2)
        self.assertNotEqual(s1, s3)

    def test_bootstrap_overall_scores_reproducible_given_same_seed(self):
        import analysis.analyze_kv_ablation as mod

        original_tasks = mod.EXPECTED_TASKS
        mod.EXPECTED_TASKS = ["taskA", "taskB"]
        try:
            rng = np.random.default_rng(0)
            sample_scores = {
                cfg: {
                    "taskA": rng.uniform(0, 100, size=20),
                    "taskB": rng.uniform(0, 100, size=30),
                }
                for cfg in ALL_CONFIGS
            }
            boot1 = bootstrap_overall_scores(sample_scores, iterations=200, seed=42)
            boot2 = bootstrap_overall_scores(sample_scores, iterations=200, seed=42)
            for cfg in ALL_CONFIGS:
                np.testing.assert_array_equal(boot1[cfg], boot2[cfg])

            boot3 = bootstrap_overall_scores(sample_scores, iterations=200, seed=43)
            self.assertFalse(np.array_equal(boot1[FP16], boot3[FP16]))
        finally:
            mod.EXPECTED_TASKS = original_tasks


class TestEqualPerTaskWeighting(unittest.TestCase):
    """Overall bootstrap score must be an equal-weight mean across tasks, not
    a sample-count-weighted mean (e.g. lcc/repobench-p have 500 rows vs
    150-200 for most other tasks and must not dominate)."""

    def test_bootstrap_overall_score_is_unweighted_task_average(self):
        import analysis.analyze_kv_ablation as mod

        original_tasks = mod.EXPECTED_TASKS
        mod.EXPECTED_TASKS = ["small_task", "big_task"]
        try:
            sample_scores = {
                cfg: {
                    "small_task": np.array([50.0, 50.0]),  # 2 rows
                    "big_task": np.array([50.0] * 10),      # 10 rows
                }
                for cfg in ALL_CONFIGS
            }
            # K4/V16 improves only on the small (2-row) task.
            sample_scores[(4, 16)]["small_task"] = np.array([60.0, 60.0])

            boot = bootstrap_overall_scores(sample_scores, iterations=50, seed=42)
            observed_fp16 = boot[FP16].mean()
            observed_k4v16 = boot[(4, 16)].mean()

            # Equal-weight expectation: (+10 on small_task, +0 on big_task) / 2 = +5,
            # NOT the row-count-weighted (10*2 + 0*10)/12 = 1.67.
            self.assertAlmostEqual(observed_k4v16 - observed_fp16, 5.0, delta=0.5)
        finally:
            mod.EXPECTED_TASKS = original_tasks


class TestLeaveOneTaskOut(unittest.TestCase):
    def test_loto_identifies_dominant_task_and_sign_flip(self):
        import analysis.analyze_kv_ablation as mod

        original_tasks = mod.EXPECTED_TASKS
        mod.EXPECTED_TASKS = ["a", "b", "c"]
        try:
            task_mean_raw = {cfg: {"a": 50.0, "b": 50.0, "c": 50.0} for cfg in ALL_CONFIGS}
            # K4/V16 - FP16 is dominated by task "c": +30 on c, +0.1 on a/b.
            task_mean_raw[(4, 16)] = {"a": 50.1, "b": 50.1, "c": 80.0}

            contrasts = {"K4/V16 - FP16": {(4, 16): 1.0, FP16: -1.0}}
            result = leave_one_task_out(task_mean_raw, contrasts)

            r = result["K4/V16 - FP16"]
            self.assertEqual(r["min_task"], "c")  # removing the dominant task minimizes the remaining effect
            self.assertLess(r["loto_values"]["c"], r["loto_values"]["a"])
            self.assertLess(r["loto_values"]["c"], r["loto_values"]["b"])

        finally:
            mod.EXPECTED_TASKS = original_tasks

    def test_loto_sign_flip_detected(self):
        import analysis.analyze_kv_ablation as mod

        original_tasks = mod.EXPECTED_TASKS
        mod.EXPECTED_TASKS = ["a", "b", "c"]
        try:
            task_mean_raw = {cfg: {"a": 50.0, "b": 50.0, "c": 50.0} for cfg in ALL_CONFIGS}
            # Full mean effect is negative overall, but driven entirely by task "c";
            # removing "c" flips the sign positive.
            task_mean_raw[(2, 2)] = {"a": 50.5, "b": 50.5, "c": 20.0}

            contrasts = {"K2/V2 - FP16": {(2, 2): 1.0, FP16: -1.0}}
            result = leave_one_task_out(task_mean_raw, contrasts)
            r = result["K2/V2 - FP16"]

            self.assertLess(r["full_15_task_mean"], 0.0)
            self.assertIn("c", r["sign_changed_when_removing"])
        finally:
            mod.EXPECTED_TASKS = original_tasks


if __name__ == "__main__":
    unittest.main()
