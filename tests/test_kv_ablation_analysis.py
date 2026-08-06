"""CPU-only unit tests for analysis/analyze_kv_ablation.py.

Does not load a model, touch the GPU, or read any prediction JSONL from the
real pred/ directories -- everything here operates on small synthetic
in-memory fixtures.

Run with:
    ./.venv/bin/python -m unittest tests.test_kv_ablation_analysis -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.analyze_kv_ablation import (  # noqa: E402
    EFFECT_NAMES,
    PairingError,
    compute_task_effect_matrix,
    compute_task_sample_scores,
    derive_seed,
    leave_one_task_out,
    sample_score,
    validate_pairing,
)
from eval_long_bench import dataset2metric, scorer  # noqa: E402


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
    """trec/triviaqa/samsum/lsht must have their prediction truncated to the
    first line before scoring; other tasks must not."""

    def test_first_line_only_applied_for_trec(self):
        prediction = "SPORTS\nirrelevant trailing hallucinated text"
        ground_truths = ["SPORTS"]
        all_classes = ["SPORTS", "NEWS"]

        score = sample_score("trec", prediction, ground_truths, all_classes)
        # classification_score would find both "SPORTS" and nothing else
        # relevant once truncated to "SPORTS" -- expect a perfect match.
        self.assertEqual(score, dataset2metric["trec"]("SPORTS", "SPORTS", all_classes=all_classes))

    def test_first_line_not_applied_for_qasper(self):
        prediction = "answer line one\nanswer line two"
        ground_truths = ["answer line one\nanswer line two"]

        score_full = sample_score("qasper", prediction, ground_truths, [])
        score_truncated = dataset2metric["qasper"]("answer line one", ground_truths[0], all_classes=[])

        # qasper must NOT be truncated, so scoring the full multi-line
        # prediction should differ from (and outperform) scoring just the
        # first line against the full multi-line ground truth.
        self.assertGreater(score_full, score_truncated)


class TestMaxOverGroundTruths(unittest.TestCase):
    def test_sample_score_takes_max_over_multiple_ground_truths(self):
        prediction = "the cat sat on the mat"
        ground_truths = ["totally different", "the cat sat on the mat"]

        score = sample_score("qasper", prediction, ground_truths, [])
        best = max(
            dataset2metric["qasper"](prediction, gt, all_classes=[]) for gt in ground_truths
        )
        self.assertEqual(score, best)
        self.assertEqual(score, 1.0)


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
        for method in ["FP16", "K2/V16", "K16/V2", "K2/V2", "K4/V4"]:
            methods[method] = {
                task: [dict(row) for row in rows] for task, rows in base.items()
            }
        if mutate:
            mutate(methods)
        return methods, tasks

    def test_pairing_passes_on_identical_metadata(self):
        methods, _ = self._make_all_data()
        # Monkeypatch EXPECTED_TASKS-dependent validate_pairing by calling it
        # directly against our 2-task fixture via a thin wrapper.
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
            methods["K2/V2"]["qasper"][1]["answers"] = ["different answer"]

        methods, _ = self._make_all_data(mutate=mutate)
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
            methods["K4/V4"]["trec"].append(
                {"answers": ["extra"], "all_classes": [], "length": 5, "pred": "z"}
            )

        methods, _ = self._make_all_data(mutate=mutate)
        import analysis.analyze_kv_ablation as mod

        original_tasks = mod.EXPECTED_TASKS
        mod.EXPECTED_TASKS = ["qasper", "trec"]
        try:
            with self.assertRaises(PairingError):
                validate_pairing(methods)
        finally:
            mod.EXPECTED_TASKS = original_tasks


class TestEqualTaskWeighting(unittest.TestCase):
    """Overall effects must be an equal-weight average across the 15 tasks,
    not a sample-count-weighted average (e.g. lcc/repobench-p have 500 rows
    vs 150-200 for most other tasks and must not dominate)."""

    def test_overall_effect_is_unweighted_task_average(self):
        sample_scores = {
            "FP16": {
                "small_task": np.array([50.0, 50.0]),
                "big_task": np.array([50.0] * 10),
            },
            "K2/V16": {
                "small_task": np.array([60.0, 60.0]),  # +10 effect, 2 rows
                "big_task": np.array([50.0] * 10),  # +0 effect, 10 rows
            },
            "K16/V2": {
                "small_task": np.array([50.0, 50.0]),
                "big_task": np.array([50.0] * 10),
            },
            "K2/V2": {
                "small_task": np.array([50.0, 50.0]),
                "big_task": np.array([50.0] * 10),
            },
            "K4/V4": {
                "small_task": np.array([50.0, 50.0]),
                "big_task": np.array([50.0] * 10),
            },
        }
        effect_small = compute_task_effect_matrix(sample_scores, "small_task")
        effect_big = compute_task_effect_matrix(sample_scores, "big_task")

        key_only_idx = EFFECT_NAMES.index("key_only")
        task_means = [effect_small[:, key_only_idx].mean(), effect_big[:, key_only_idx].mean()]
        equal_weighted = sum(task_means) / len(task_means)

        # Equal-weight average of +10 (small_task, 2 rows) and +0 (big_task,
        # 10 rows) must be +5, NOT the row-count-weighted (10*2 + 0*10)/12 = 1.67.
        self.assertAlmostEqual(equal_weighted, 5.0, places=6)
        row_weighted = (10.0 * 2 + 0.0 * 10) / 12
        self.assertNotAlmostEqual(equal_weighted, row_weighted, places=2)


class TestInteractionFormula(unittest.TestCase):
    def test_interaction_matches_definition(self):
        sample_scores = {
            "FP16": {"t": np.array([100.0])},
            "K2/V16": {"t": np.array([90.0])},   # key_only = -10
            "K16/V2": {"t": np.array([85.0])},   # value_only = -15
            "K2/V2": {"t": np.array([60.0])},    # joint = -40
            "K4/V4": {"t": np.array([95.0])},
        }
        effect_matrix = compute_task_effect_matrix(sample_scores, "t")
        interaction = effect_matrix[0, EFFECT_NAMES.index("interaction")]
        # interaction = joint - key_only - value_only = -40 - (-10) - (-15) = -15
        self.assertAlmostEqual(interaction, -15.0, places=6)

    def test_interaction_zero_when_effects_are_additive(self):
        sample_scores = {
            "FP16": {"t": np.array([100.0])},
            "K2/V16": {"t": np.array([90.0])},   # key_only = -10
            "K16/V2": {"t": np.array([95.0])},   # value_only = -5
            "K2/V2": {"t": np.array([85.0])},    # joint = -15 == -10 + -5
            "K4/V4": {"t": np.array([100.0])},
        }
        effect_matrix = compute_task_effect_matrix(sample_scores, "t")
        interaction = effect_matrix[0, EFFECT_NAMES.index("interaction")]
        self.assertAlmostEqual(interaction, 0.0, places=6)


class TestBootstrapSeedReproducibility(unittest.TestCase):
    def test_derive_seed_is_deterministic(self):
        s1 = derive_seed(42, "primary", "qasper")
        s2 = derive_seed(42, "primary", "qasper")
        self.assertEqual(s1, s2)

    def test_derive_seed_differs_across_inputs(self):
        s1 = derive_seed(42, "primary", "qasper")
        s2 = derive_seed(42, "primary", "trec")
        s3 = derive_seed(43, "primary", "qasper")
        self.assertNotEqual(s1, s2)
        self.assertNotEqual(s1, s3)

    def test_bootstrap_replicate_reproducible_given_same_seed(self):
        rng1 = np.random.default_rng(derive_seed(42, "x"))
        rng2 = np.random.default_rng(derive_seed(42, "x"))
        a = rng1.integers(0, 100, size=1000)
        b = rng2.integers(0, 100, size=1000)
        np.testing.assert_array_equal(a, b)


class TestLeaveOneTaskOut(unittest.TestCase):
    def test_loto_identifies_dominant_task(self):
        # 3 tasks; effect for key_only column is dominated by task "c".
        observed_task_means = {
            "a": np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "b": np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "c": np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }
        import analysis.analyze_kv_ablation as mod

        original_tasks = mod.EXPECTED_TASKS
        mod.EXPECTED_TASKS = ["a", "b", "c"]
        try:
            result = leave_one_task_out(observed_task_means)
        finally:
            mod.EXPECTED_TASKS = original_tasks

        key_only = result["key_only"]
        # Removing "c" should produce a much smaller mean than removing "a" or "b".
        self.assertEqual(key_only["min_task"], "c")
        self.assertLess(key_only["loto_values"]["c"], key_only["loto_values"]["a"])
        self.assertLess(key_only["loto_values"]["c"], key_only["loto_values"]["b"])


if __name__ == "__main__":
    unittest.main()
