"""CPU-only unit tests for analysis/analyze_layer_sensitivity_f0.py.

Does not load a model, touch the GPU, or read any real prediction JSONL --
everything here operates on small synthetic in-memory fixtures.

Run with:
    ./.venv/bin/python -m unittest tests.test_layer_sensitivity_f0 -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.analyze_layer_sensitivity_f0 import (  # noqa: E402
    COMBINED_6_TASKS,
    FP16,
    LAYER00_KEY,
    LAYER00_VALUE,
    NEW_2_TASKS,
    ORIGINAL_4_TASKS,
    F0AnalysisError,
    aggregate,
    apply_gate,
    compute_task_sample_scores,
    leave_one_task_out,
    per_task_deltas_from_raw,
    sample_score,
    validate_pairing,
)


class TestExactScoringReproduction(unittest.TestCase):
    """Must reuse (not reimplement) Phase-1's exact scoring functions."""

    def test_compute_task_sample_scores_is_reused_object(self):
        from analysis.analyze_kv_ablation import compute_task_sample_scores as kv_fn

        self.assertIs(compute_task_sample_scores, kv_fn)

    def test_sample_score_matches_manual_metric_call(self):
        from eval_long_bench import dataset2metric

        score = sample_score("qasper", "the cat sat", ["the cat sat"], [])
        self.assertEqual(score, dataset2metric["qasper"]("the cat sat", "the cat sat", all_classes=[]))


def _raw(fp16, key, value):
    return {FP16: dict(fp16), LAYER00_KEY: dict(key), LAYER00_VALUE: dict(value)}


class TestAggregations(unittest.TestCase):
    def test_original_4_task_aggregation(self):
        task_mean_raw = _raw(
            {t: 50.0 for t in ORIGINAL_4_TASKS},
            {"trec": 50.0, "lcc": 60.0, "passage_retrieval_en": 50.0, "2wikimqa": 50.0},  # +10 on lcc only
            {t: 50.0 for t in ORIGINAL_4_TASKS},
        )
        key_deltas, value_deltas = per_task_deltas_from_raw(task_mean_raw, ORIGINAL_4_TASKS)
        agg = aggregate(key_deltas, value_deltas)
        # equal-weight: (+10 + 0 + 0 + 0) / 4 = 2.5, NOT sample-count-weighted
        self.assertAlmostEqual(agg["signed_key_sensitivity"], 2.5, places=6)

    def test_new_2_task_aggregation(self):
        task_mean_raw = _raw(
            {t: 50.0 for t in NEW_2_TASKS},
            {"multifieldqa_en": 60.0, "samsum": 50.0},
            {"multifieldqa_en": 50.0, "samsum": 40.0},
        )
        key_deltas, value_deltas = per_task_deltas_from_raw(task_mean_raw, NEW_2_TASKS)
        agg = aggregate(key_deltas, value_deltas)
        self.assertAlmostEqual(agg["signed_key_sensitivity"], 5.0, places=6)  # (10+0)/2
        self.assertAlmostEqual(agg["signed_value_sensitivity"], -5.0, places=6)  # (0-10)/2

    def test_combined_6_equal_weighting_not_pooled(self):
        # lcc (500 samples) must not dominate a pooled mean; equal-task
        # weighting means it counts the same as multifieldqa_en (150) here.
        fp16 = {t: 50.0 for t in COMBINED_6_TASKS}
        key = dict(fp16)
        key["lcc"] = 50.0 + 6.0  # only lcc moves
        value = dict(fp16)
        task_mean_raw = _raw(fp16, key, value)
        key_deltas, value_deltas = per_task_deltas_from_raw(task_mean_raw, COMBINED_6_TASKS)
        agg = aggregate(key_deltas, value_deltas)
        self.assertAlmostEqual(agg["signed_key_sensitivity"], 1.0, places=6)  # 6/6 tasks, not sample-weighted
        self.assertAlmostEqual(agg["signed_value_sensitivity"], 0.0, places=6)


class TestSignConventionAndAxisDifference(unittest.TestCase):
    def test_key_value_sign_convention(self):
        task_mean_raw = _raw({"trec": 60.0}, {"trec": 50.0}, {"trec": 70.0})
        key_deltas, value_deltas = per_task_deltas_from_raw(task_mean_raw, ["trec"])
        self.assertLess(key_deltas["trec"], 0)  # 50-60 = -10 -> degradation
        self.assertGreater(value_deltas["trec"], 0)  # 70-60 = +10 -> improvement

    def test_axis_difference_is_key_minus_value(self):
        task_mean_raw = _raw({"trec": 50.0}, {"trec": 60.0}, {"trec": 40.0})
        key_deltas, value_deltas = per_task_deltas_from_raw(task_mean_raw, ["trec"])
        agg = aggregate(key_deltas, value_deltas)
        # KeyDelta=+10, ValueDelta=-10 -> AxisDifference = 10 - (-10) = 20
        self.assertAlmostEqual(agg["axis_difference"], 20.0, places=6)
        self.assertAlmostEqual(agg["axis_difference"], agg["signed_key_sensitivity"] - agg["signed_value_sensitivity"], places=6)


class TestPairedBootstrapReproducibility(unittest.TestCase):
    def test_same_seed_gives_identical_bootstrap(self):
        from analysis.analyze_layer_sensitivity_f0 import bootstrap_condition_scores

        rng = np.random.default_rng(0)
        sample_scores = {
            FP16: {"trec": rng.uniform(0, 100, 20)},
            LAYER00_KEY: {"trec": rng.uniform(0, 100, 20)},
            LAYER00_VALUE: {"trec": rng.uniform(0, 100, 20)},
        }
        configs = [FP16, LAYER00_KEY, LAYER00_VALUE]
        b1 = bootstrap_condition_scores(sample_scores, ["trec"], configs, 300, 42)
        b2 = bootstrap_condition_scores(sample_scores, ["trec"], configs, 300, 42)
        for cfg in configs:
            np.testing.assert_array_equal(b1[cfg], b2[cfg])


class TestSharedResamplingIndices(unittest.TestCase):
    def test_identical_scores_yield_identical_boot_arrays_across_configs(self):
        from analysis.analyze_layer_sensitivity_f0 import bootstrap_condition_scores

        rng = np.random.default_rng(1)
        scores = rng.uniform(0, 100, 15)
        sample_scores = {
            FP16: {"t": scores.copy()},
            LAYER00_KEY: {"t": scores.copy()},
            LAYER00_VALUE: {"t": scores.copy()},
        }
        boot = bootstrap_condition_scores(sample_scores, ["t"], [FP16, LAYER00_KEY, LAYER00_VALUE], 300, 7)
        np.testing.assert_array_equal(boot[FP16], boot[LAYER00_KEY])
        np.testing.assert_array_equal(boot[FP16], boot[LAYER00_VALUE])


class TestLccRemovalCalculation(unittest.TestCase):
    def test_lcc_removed_matches_manual_mean_of_remaining_tasks(self):
        fp16 = {t: 50.0 for t in COMBINED_6_TASKS}
        key = dict(fp16)
        key.update({"trec": 51.0, "lcc": 100.0, "passage_retrieval_en": 52.0, "2wikimqa": 53.0, "multifieldqa_en": 54.0, "samsum": 55.0})
        task_mean_raw = _raw(fp16, key, dict(fp16))
        key_deltas, value_deltas = per_task_deltas_from_raw(task_mean_raw, COMBINED_6_TASKS)
        remaining = [t for t in COMBINED_6_TASKS if t != "lcc"]
        agg_minus_lcc = aggregate({t: key_deltas[t] for t in remaining}, {t: value_deltas[t] for t in remaining})
        expected = np.mean([1.0, 2.0, 3.0, 4.0, 5.0])  # deltas for trec,pre,2wiki,mfqa,samsum
        self.assertAlmostEqual(agg_minus_lcc["signed_key_sensitivity"], expected, places=6)

    def test_leave_one_task_out_lcc_entry_excludes_lcc(self):
        fp16 = {t: 50.0 for t in COMBINED_6_TASKS}
        key = dict(fp16)
        key["lcc"] = 90.0  # large lcc-only effect
        task_mean_raw = _raw(fp16, key, dict(fp16))
        key_deltas, value_deltas = per_task_deltas_from_raw(task_mean_raw, COMBINED_6_TASKS)
        loto = leave_one_task_out(key_deltas, value_deltas, COMBINED_6_TASKS)
        # removing lcc must eliminate its +40 effect entirely from the mean
        self.assertAlmostEqual(loto["lcc"]["signed_key_sensitivity"], 0.0, places=6)


class TestGateCriteria(unittest.TestCase):
    def _task_table(self, mfqa_axis_diff, samsum_axis_diff):
        return {
            "multifieldqa_en": {"axis_difference": mfqa_axis_diff},
            "samsum": {"axis_difference": samsum_axis_diff},
        }

    def test_all_five_criteria_pass_yields_go(self):
        combined6 = {"signed_key_sensitivity": 0.5, "signed_value_sensitivity": -0.5, "axis_difference": 1.0}
        minus_lcc = {"signed_key_sensitivity": 0.2, "signed_value_sensitivity": -0.1, "axis_difference": 0.3}
        table = self._task_table(mfqa_axis_diff=-0.1, samsum_axis_diff=0.2)
        criteria, gate_pass = apply_gate(combined6, minus_lcc, table)
        self.assertTrue(all(criteria.values()))
        self.assertTrue(gate_pass)

    def test_criterion_1_fails_when_signed_key_not_positive(self):
        combined6 = {"signed_key_sensitivity": -0.1, "signed_value_sensitivity": -0.5, "axis_difference": 1.0}
        minus_lcc = {"signed_key_sensitivity": 0.2, "signed_value_sensitivity": -0.1, "axis_difference": 0.3}
        table = self._task_table(-0.1, 0.2)
        criteria, gate_pass = apply_gate(combined6, minus_lcc, table)
        self.assertFalse(criteria["criterion_1_combined6_signed_key_gt_0"])
        self.assertFalse(gate_pass)

    def test_criterion_4_fails_when_lcc_removed_value_flips_sign(self):
        # Mirrors the real F0 result: lcc-removed SignedValueSensitivity
        # flips to positive, so criterion 4 must fail even though 1-3 pass.
        combined6 = {"signed_key_sensitivity": 0.6, "signed_value_sensitivity": -0.7, "axis_difference": 1.3}
        minus_lcc = {"signed_key_sensitivity": 0.2, "signed_value_sensitivity": +0.004, "axis_difference": 0.2}
        table = self._task_table(-0.1, 0.1)
        criteria, gate_pass = apply_gate(combined6, minus_lcc, table)
        self.assertTrue(criteria["criterion_1_combined6_signed_key_gt_0"])
        self.assertTrue(criteria["criterion_2_combined6_signed_value_lt_0"])
        self.assertTrue(criteria["criterion_3_combined6_axis_difference_gt_0"])
        self.assertFalse(criteria["criterion_4_lcc_removed_all_three_hold"])
        self.assertFalse(gate_pass)

    def test_criterion_5_requires_at_least_one_not_both(self):
        combined6 = {"signed_key_sensitivity": 0.5, "signed_value_sensitivity": -0.5, "axis_difference": 1.0}
        minus_lcc = {"signed_key_sensitivity": 0.2, "signed_value_sensitivity": -0.1, "axis_difference": 0.3}
        # multifieldqa_en negative, samsum positive -- "at least one" passes.
        table = self._task_table(mfqa_axis_diff=-0.5, samsum_axis_diff=0.3)
        criteria, _ = apply_gate(combined6, minus_lcc, table)
        self.assertTrue(criteria["criterion_5_at_least_one_new_task_axis_difference_gt_0"])
        # neither positive -- fails.
        table_both_neg = self._task_table(mfqa_axis_diff=-0.5, samsum_axis_diff=-0.3)
        criteria2, gate_pass2 = apply_gate(combined6, minus_lcc, table_both_neg)
        self.assertFalse(criteria2["criterion_5_at_least_one_new_task_axis_difference_gt_0"])
        self.assertFalse(gate_pass2)


class TestGoNoGoDetermination(unittest.TestCase):
    def test_go_requires_literally_all_five(self):
        base_pass = {"signed_key_sensitivity": 0.5, "signed_value_sensitivity": -0.5, "axis_difference": 1.0}
        base_minus_lcc_pass = {"signed_key_sensitivity": 0.2, "signed_value_sensitivity": -0.1, "axis_difference": 0.3}
        table_pass = {"multifieldqa_en": {"axis_difference": 0.1}, "samsum": {"axis_difference": -0.1}}
        _, gate_pass = apply_gate(base_pass, base_minus_lcc_pass, table_pass)
        self.assertTrue(gate_pass)

        # Flip any single criterion and the overall decision must be NO_GO.
        _, gate_pass_fail = apply_gate(
            {**base_pass, "axis_difference": -0.1}, base_minus_lcc_pass, table_pass
        )
        self.assertFalse(gate_pass_fail)

    def test_real_f0_result_is_no_go(self):
        # Exact reproduction of this round's real result.
        combined6 = {"signed_key_sensitivity": 0.6217, "signed_value_sensitivity": -0.6921, "axis_difference": 1.3138}
        minus_lcc = {"signed_key_sensitivity": 0.218, "signed_value_sensitivity": 0.0039, "axis_difference": 0.2141}
        table = {"multifieldqa_en": {"axis_difference": -0.1322}, "samsum": {"axis_difference": 0.1326}}
        criteria, gate_pass = apply_gate(combined6, minus_lcc, table)
        self.assertFalse(criteria["criterion_4_lcc_removed_all_three_hold"])
        self.assertFalse(gate_pass)


class TestDeterministicOutputOrdering(unittest.TestCase):
    def test_task_order_constants_are_fixed(self):
        self.assertEqual(ORIGINAL_4_TASKS, ["trec", "lcc", "passage_retrieval_en", "2wikimqa"])
        self.assertEqual(NEW_2_TASKS, ["multifieldqa_en", "samsum"])
        self.assertEqual(COMBINED_6_TASKS, ORIGINAL_4_TASKS + NEW_2_TASKS)

    def test_leave_one_task_out_preserves_task_set_as_keys(self):
        fp16 = {t: 50.0 for t in COMBINED_6_TASKS}
        task_mean_raw = _raw(fp16, dict(fp16), dict(fp16))
        key_deltas, value_deltas = per_task_deltas_from_raw(task_mean_raw, COMBINED_6_TASKS)
        loto = leave_one_task_out(key_deltas, value_deltas, COMBINED_6_TASKS)
        self.assertEqual(list(loto.keys()), COMBINED_6_TASKS)


class TestPairingFailsClosed(unittest.TestCase):
    def test_mismatched_metadata_raises(self):
        all_data = {
            FP16: {"trec": [{"answers": ["a"], "all_classes": [], "length": 1}]},
            LAYER00_KEY: {"trec": [{"answers": ["different"], "all_classes": [], "length": 1}]},
            LAYER00_VALUE: {"trec": [{"answers": ["a"], "all_classes": [], "length": 1}]},
        }
        with self.assertRaises(F0AnalysisError):
            validate_pairing(all_data, tasks=["trec"])

    def test_matching_metadata_passes(self):
        all_data = {
            FP16: {"trec": [{"answers": ["a"], "all_classes": [], "length": 1}]},
            LAYER00_KEY: {"trec": [{"answers": ["a"], "all_classes": [], "length": 1}]},
            LAYER00_VALUE: {"trec": [{"answers": ["a"], "all_classes": [], "length": 1}]},
        }
        result = validate_pairing(all_data, tasks=["trec"])
        self.assertEqual(result["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
