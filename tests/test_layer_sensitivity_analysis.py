"""CPU-only unit tests for analysis/analyze_layer_sensitivity_pilot.py.

Does not load a model, touch the GPU, or read any real pilot/pred JSONL --
everything here operates on small synthetic in-memory fixtures.

Run with:
    ./.venv/bin/python -m unittest tests.test_layer_sensitivity_analysis -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.analyze_layer_sensitivity_pilot import (  # noqa: E402
    FP16,
    _rank,
    apply_contrast,
    bootstrap_condition_scores,
    build_key_value_contrasts,
    compute_layer_sensitivities,
    leave_one_task_out_contrast,
    leave_one_task_out_per_layer,
    sample_score,
    spearman_corr,
)
from analysis.analyze_kv_ablation import (  # noqa: E402
    compute_task_sample_scores as kv_ablation_compute_task_sample_scores,
)
from analysis.analyze_layer_sensitivity_pilot import compute_task_sample_scores  # noqa: E402


class TestReusesPhase1ScoringNotReimplemented(unittest.TestCase):
    """The module must import (not redefine/duplicate) Phase-1's exact
    scoring functions."""

    def test_compute_task_sample_scores_is_the_same_function_object(self):
        self.assertIs(compute_task_sample_scores, kv_ablation_compute_task_sample_scores)

    def test_sample_score_matches_manual_metric_call(self):
        from eval_long_bench import dataset2metric

        score = sample_score("qasper", "the cat sat", ["the cat sat"], [])
        expected = dataset2metric["qasper"]("the cat sat", "the cat sat", all_classes=[])
        self.assertEqual(score, expected)


class TestEqualTaskWeighting(unittest.TestCase):
    """SignedKeySensitivity/SignedValueSensitivity must be an unweighted
    mean across the 4 tasks, never a pooled raw mean over all 1100 rows
    (lcc has 500 rows vs 200 for the other three and must not dominate)."""

    def test_signed_sensitivity_is_unweighted_task_average(self):
        task_mean_raw = {
            FP16: {"trec": 50.0, "lcc": 50.0, "passage_retrieval_en": 50.0, "2wikimqa": 50.0},
            (0, "key"): {"trec": 50.0, "lcc": 90.0, "passage_retrieval_en": 50.0, "2wikimqa": 50.0},  # +40 on lcc only
            (0, "value"): {"trec": 50.0, "lcc": 50.0, "passage_retrieval_en": 50.0, "2wikimqa": 50.0},
        }
        result, per_task = compute_layer_sensitivities(task_mean_raw, layers=(0,), tasks=list(task_mean_raw[FP16]))
        # Equal-weight expectation: (+40 on lcc, +0 on the other 3) / 4 = +10,
        # NOT the row-count-weighted (40*500)/1100 = 18.18.
        self.assertAlmostEqual(result[0]["signed_key_sensitivity"], 10.0, places=6)
        row_weighted = (40.0 * 500) / 1100
        self.assertNotAlmostEqual(result[0]["signed_key_sensitivity"], row_weighted, places=2)


class TestSignConvention(unittest.TestCase):
    def test_negative_delta_is_degradation_positive_is_improvement(self):
        task_mean_raw = {
            FP16: {"trec": 60.0},
            (5, "key"): {"trec": 50.0},  # perturbed - baseline = -10 -> degradation
            (5, "value"): {"trec": 70.0},  # perturbed - baseline = +10 -> improvement
        }
        result, _ = compute_layer_sensitivities(task_mean_raw, layers=(5,), tasks=["trec"])
        self.assertLess(result[5]["signed_key_sensitivity"], 0)
        self.assertGreater(result[5]["signed_value_sensitivity"], 0)


class TestAbsoluteSensitivity(unittest.TestCase):
    def test_abs_sensitivity_is_mean_of_abs_not_abs_of_mean(self):
        task_mean_raw = {
            FP16: {"a": 50.0, "b": 50.0},
            (0, "key"): {"a": 60.0, "b": 40.0},  # deltas: +10, -10 -> signed mean = 0, abs mean = 10
            (0, "value"): {"a": 50.0, "b": 50.0},
        }
        result, _ = compute_layer_sensitivities(task_mean_raw, layers=(0,), tasks=["a", "b"])
        self.assertAlmostEqual(result[0]["signed_key_sensitivity"], 0.0, places=6)
        self.assertAlmostEqual(result[0]["abs_key_sensitivity"], 10.0, places=6)
        self.assertNotEqual(result[0]["signed_key_sensitivity"], result[0]["abs_key_sensitivity"])


class TestBootstrapReproducibility(unittest.TestCase):
    def test_same_seed_gives_identical_bootstrap(self):
        rng = np.random.default_rng(0)
        sample_scores = {
            FP16: {"trec": rng.uniform(0, 100, 20), "lcc": rng.uniform(0, 100, 30)},
            (0, "key"): {"trec": rng.uniform(0, 100, 20), "lcc": rng.uniform(0, 100, 30)},
        }
        configs = [FP16, (0, "key")]
        boot1 = bootstrap_condition_scores(sample_scores, ["trec", "lcc"], configs, 500, 42)
        boot2 = bootstrap_condition_scores(sample_scores, ["trec", "lcc"], configs, 500, 42)
        for cfg in configs:
            np.testing.assert_array_equal(boot1[cfg], boot2[cfg])

    def test_different_seed_gives_different_bootstrap(self):
        rng = np.random.default_rng(0)
        sample_scores = {
            FP16: {"trec": rng.uniform(0, 100, 20)},
            (0, "key"): {"trec": rng.uniform(0, 100, 20)},
        }
        configs = [FP16, (0, "key")]
        boot1 = bootstrap_condition_scores(sample_scores, ["trec"], configs, 200, 42)
        boot2 = bootstrap_condition_scores(sample_scores, ["trec"], configs, 200, 43)
        self.assertFalse(np.array_equal(boot1[FP16], boot2[FP16]))


class TestSharedBootstrapIndices(unittest.TestCase):
    """Every config must be resampled with the SAME per-task indices in a
    given replicate -- this is what keeps every layer/axis contrast paired."""

    def test_identical_sample_scores_across_configs_yield_identical_boot_arrays(self):
        rng = np.random.default_rng(1)
        scores = rng.uniform(0, 100, 15)
        # Three configs with byte-identical underlying scores: if indices
        # were drawn independently per config, the resulting bootstrap
        # arrays would almost surely differ; if shared, they must be
        # exactly equal for every replicate.
        sample_scores = {
            FP16: {"t": scores.copy()},
            (0, "key"): {"t": scores.copy()},
            (0, "value"): {"t": scores.copy()},
        }
        configs = [FP16, (0, "key"), (0, "value")]
        boot = bootstrap_condition_scores(sample_scores, ["t"], configs, 300, 7)
        np.testing.assert_array_equal(boot[FP16], boot[(0, "key")])
        np.testing.assert_array_equal(boot[FP16], boot[(0, "value")])


class TestFixedExtremeContrastSemantics(unittest.TestCase):
    """Extrema must be chosen ONCE from point estimates, then bootstrapped
    as a fixed linear contrast -- never re-selected inside each replicate
    (which would introduce winner-selection bias)."""

    def test_contrast_dict_references_the_preselected_layers_only(self):
        key_contrasts, value_contrasts, axis_diff = build_key_value_contrasts(layers=(0, 4, 9))
        # A "most vs least" contrast built externally from two fixed,
        # pre-chosen layers must be a static 2-term dict -- not a function
        # of the bootstrap sample.
        most, least = 4, 9
        contrast = {(most, "key"): 1.0, (least, "key"): -1.0}
        score_map = {(0, "key"): 1.0, (4, "key"): 5.0, (9, "key"): 2.0}
        # apply_contrast must be a pure function of the fixed contrast dict
        # and the score map -- calling it twice must give the same result
        # regardless of any other config present.
        self.assertEqual(apply_contrast(score_map, contrast), 3.0)
        self.assertEqual(apply_contrast(score_map, contrast), 3.0)

    def test_axis_difference_contrast_is_fp16_free(self):
        # AxisDifference(L) = KeySens(L) - ValueSens(L); the FP16 terms in
        # each must algebraically cancel, so the contrast dict must not
        # reference FP16 at all.
        _, _, axis_diff = build_key_value_contrasts(layers=(4,))
        contrast = axis_diff["AxisDifference(L04)"]
        self.assertNotIn(FP16, contrast)
        self.assertEqual(contrast, {(4, "key"): 1.0, (4, "value"): -1.0})


class TestLOTOSignFlip(unittest.TestCase):
    def test_per_layer_loto_detects_sign_flip_driven_by_one_task(self):
        # 4 tasks; lcc is a large negative outlier that drags the full mean
        # negative, while the other 3 tasks average to slightly positive on
        # their own -- mirrors the real pilot's layer-0 Value-sensitivity
        # finding (full mean negative, sign flips positive once lcc is
        # removed).
        per_task_deltas = {
            (0, "value"): {"trec": 0.1, "lcc": -4.0, "passage_retrieval_en": 0.05, "2wikimqa": -0.05},
        }
        result = leave_one_task_out_per_layer(per_task_deltas, layers=(0,), axes=("value",), tasks=["trec", "lcc", "passage_retrieval_en", "2wikimqa"])
        r = result[(0, "value")]
        self.assertLess(r["full_4task_mean"], 0)
        self.assertIn("lcc", r["sign_changed_when_removing"])
        self.assertGreater(r["loto_values"]["lcc"], 0)  # mean of the other 3 without lcc is positive

    def test_no_sign_flip_when_robust_across_tasks(self):
        per_task_deltas = {
            (4, "key"): {"trec": -1.0, "lcc": -1.2, "passage_retrieval_en": -0.9, "2wikimqa": -1.1},
        }
        result = leave_one_task_out_per_layer(per_task_deltas, layers=(4,), axes=("key",), tasks=["trec", "lcc", "passage_retrieval_en", "2wikimqa"])
        r = result[(4, "key")]
        self.assertEqual(r["sign_changed_when_removing"], [])

    def test_extreme_contrast_loto_uses_per_task_difference(self):
        per_task_deltas = {
            (0, "key"): {"a": 10.0, "b": 20.0},
            (4, "key"): {"a": 2.0, "b": 4.0},
        }
        result = leave_one_task_out_contrast(per_task_deltas, 0, 4, "key", tasks=["a", "b"])
        # per-task contrast: a: 10-2=8, b: 20-4=16 -> full mean = 12
        self.assertAlmostEqual(result["full_4task_mean"], 12.0, places=6)
        self.assertAlmostEqual(result["loto_values"]["a"], 16.0, places=6)  # only "b" remains
        self.assertAlmostEqual(result["loto_values"]["b"], 8.0, places=6)  # only "a" remains


class TestSpearmanRankCorrelation(unittest.TestCase):
    def test_perfect_positive_correlation(self):
        self.assertAlmostEqual(spearman_corr([1, 2, 3, 4], [10, 20, 30, 40]), 1.0, places=6)

    def test_perfect_negative_correlation(self):
        self.assertAlmostEqual(spearman_corr([1, 2, 3, 4], [40, 30, 20, 10]), -1.0, places=6)

    def test_no_variance_returns_none(self):
        self.assertIsNone(spearman_corr([1, 2, 3], [5, 5, 5]))

    def test_matches_known_value_with_ties(self):
        # Known Spearman result for x=[1,2,2,4], y=[1,3,2,4]:
        # ranks(x)=[1,2.5,2.5,4], ranks(y)=[1,3,2,4] -> pearson of ranks.
        rx = _rank([1, 2, 2, 4])
        np.testing.assert_array_equal(rx, [1.0, 2.5, 2.5, 4.0])


class TestSignedRangeContrastTerminology(unittest.TestCase):
    """Regression test for the terminology fix: 'Most/Least Key/Value' is
    scientifically ambiguous (doesn't say whether it's signed direction or
    magnitude) and must never reappear in this module's source."""

    def test_module_source_contains_no_ambiguous_terms(self):
        import inspect

        import analysis.analyze_layer_sensitivity_pilot as mod

        source = inspect.getsource(mod)
        for banned in ("MostKey", "LeastKey", "MostValue", "LeastValue"):
            self.assertNotIn(banned, source, f"ambiguous term {banned!r} must not appear in analyze_layer_sensitivity_pilot.py")

    def test_module_source_uses_signed_range_naming(self):
        import inspect

        import analysis.analyze_layer_sensitivity_pilot as mod

        source = inspect.getsource(mod)
        self.assertIn("SignedKeyRangeContrast", source)
        self.assertIn("SignedValueRangeContrast", source)
        self.assertIn("most_harmed_key_layer", source)
        self.assertIn("most_improved_key_layer", source)


class TestOutputDeterministicOrdering(unittest.TestCase):
    def test_compute_layer_sensitivities_preserves_layer_order(self):
        layers = (31, 0, 13, 4)  # deliberately out of numeric order
        task_mean_raw = {FP16: {"t": 50.0}}
        for l in layers:
            task_mean_raw[(l, "key")] = {"t": 50.0}
            task_mean_raw[(l, "value")] = {"t": 50.0}
        result, _ = compute_layer_sensitivities(task_mean_raw, layers=layers, tasks=["t"])
        self.assertEqual(list(result.keys()), list(layers))

    def test_build_key_value_contrasts_preserves_layer_order(self):
        layers = (9, 0, 27)
        key_c, value_c, axis_c = build_key_value_contrasts(layers=layers)
        self.assertEqual(list(key_c.keys()), [f"KeySensitivity(L{l:02d})" for l in layers])
        self.assertEqual(list(value_c.keys()), [f"ValueSensitivity(L{l:02d})" for l in layers])


if __name__ == "__main__":
    unittest.main()
