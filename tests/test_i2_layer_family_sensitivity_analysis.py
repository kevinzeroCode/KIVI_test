"""CPU-only unit tests for analysis/analyze_i2_layer_family_sensitivity.py.

Does not load a model, touch the GPU, or read any real I2 generation
output -- everything here operates on small synthetic in-memory/temp-file
fixtures.

Run with:
    ./.venv/bin/python -m unittest tests.test_i2_layer_family_sensitivity_analysis -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.analyze_i2_layer_family_sensitivity import (  # noqa: E402
    C1_NON_SELECTION_NOTE,
    INTERPRETATION_CATEGORIES,
    KIVI,
    ROTATION,
    I2AnalysisError,
    bootstrap_family_effect_range,
    bootstrap_layer_aggregate_deltas,
    bootstrap_task_condition_scores,
    bootstrap_task_deltas,
    classify_interpretation,
    compute_layer_aggregate_deltas,
    compute_task_deltas,
    load_task_rows,
    strong_crossover_layers,
    task_conditioned_matrix,
    validate_index_integrity,
    validate_pairing,
)
from analysis.analyze_kv_ablation import (  # noqa: E402
    compute_task_sample_scores as kv_ablation_compute_task_sample_scores,
)
from analysis.analyze_i2_layer_family_sensitivity import compute_task_sample_scores  # noqa: E402,F811 -- reused import check below


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            json.dump(r, f)
            f.write("\n")


def _row(i, pred="p", answers=("a",), all_classes=(), length=10):
    return {"dataset_index": i, "pred": pred, "answers": list(answers), "all_classes": list(all_classes), "length": length}


class TestReusesPhase1ScoringNotReimplemented(unittest.TestCase):
    def test_compute_task_sample_scores_is_the_same_function_object(self):
        self.assertIs(compute_task_sample_scores, kv_ablation_compute_task_sample_scores)


# --- equal-task aggregation ----------------------------------------------------

class TestEqualTaskWeighting(unittest.TestCase):
    """Delta_family(layer) must be an unweighted mean across the 6 tasks,
    never a pooled/sample-count-weighted mean (lcc's 500 rows must not
    dominate a 200- or 150-row task)."""

    def test_layer_aggregate_is_unweighted_task_average(self):
        tasks = ["trec", "lcc", "passage_retrieval_en", "2wikimqa", "multifieldqa_en", "samsum"]
        # Only lcc differs; if lcc's larger sample count leaked into the
        # aggregation weight, the aggregate would not equal a simple 1/6 mean.
        task_mean_raw = {
            (0, KIVI): {t: 50.0 for t in tasks},
            (0, ROTATION): {t: (90.0 if t == "lcc" else 50.0) for t in tasks},
        }
        task_deltas = compute_task_deltas(task_mean_raw, layers=(0,), tasks=tasks)
        layer_point = compute_layer_aggregate_deltas(task_deltas, layers=(0,), tasks=tasks)
        # unweighted: (+40 on lcc, +0 elsewhere) / 6 = +6.666...
        self.assertAlmostEqual(layer_point[0], 40.0 / 6.0, places=6)

    def test_raw_sample_weighting_would_give_a_different_wrong_answer(self):
        """Explicit negative check: a naive sample-count-weighted average
        (lcc=500 weighted vs. e.g. multifieldqa_en=150) must NOT match the
        equal-task aggregate -- proving equal-task weighting is actually
        doing something, not accidentally equivalent for this fixture."""
        tasks = ["lcc", "multifieldqa_en"]
        counts = {"lcc": 500, "multifieldqa_en": 150}
        task_mean_raw = {
            (0, KIVI): {"lcc": 50.0, "multifieldqa_en": 50.0},
            (0, ROTATION): {"lcc": 90.0, "multifieldqa_en": 50.0},
        }
        task_deltas = compute_task_deltas(task_mean_raw, layers=(0,), tasks=tasks)
        equal_task = compute_layer_aggregate_deltas(task_deltas, layers=(0,), tasks=tasks)[0]
        sample_weighted = sum(task_deltas[(0, t)] * counts[t] for t in tasks) / sum(counts.values())
        self.assertAlmostEqual(equal_task, 20.0, places=6)  # (40 + 0) / 2
        self.assertNotAlmostEqual(equal_task, sample_weighted, places=2)


# --- sign convention -----------------------------------------------------------

class TestSignConvention(unittest.TestCase):
    def test_positive_delta_means_rotation_higher(self):
        task_mean_raw = {(0, KIVI): {"trec": 50.0}, (0, ROTATION): {"trec": 60.0}}
        deltas = compute_task_deltas(task_mean_raw, layers=(0,), tasks=["trec"])
        self.assertGreater(deltas[(0, "trec")], 0)

    def test_negative_delta_means_kivi_higher(self):
        task_mean_raw = {(0, KIVI): {"trec": 60.0}, (0, ROTATION): {"trec": 50.0}}
        deltas = compute_task_deltas(task_mean_raw, layers=(0,), tasks=["trec"])
        self.assertLess(deltas[(0, "trec")], 0)


# --- dataset_index integrity: duplicate / missing / partial rejection ---------

class TestIndexIntegrity(unittest.TestCase):
    def test_exact_set_passes(self):
        rows = [_row(i) for i in range(10)]
        result = validate_index_integrity(rows, 10, "test")
        self.assertEqual(result["status"], "PASS")

    def test_duplicate_index_rejected(self):
        rows = [_row(i) for i in range(9)] + [_row(3)]  # index 3 duplicated, index 9 missing
        with self.assertRaises(I2AnalysisError):
            validate_index_integrity(rows, 10, "test")

    def test_missing_index_rejected(self):
        rows = [_row(i) for i in range(9)]  # only 9 rows, index 9 never appears
        with self.assertRaises(I2AnalysisError):
            validate_index_integrity(rows, 10, "test")

    def test_partial_output_row_count_short_is_rejected(self):
        rows = [_row(i) for i in range(5)]
        with self.assertRaises(I2AnalysisError):
            validate_index_integrity(rows, 10, "test")

    def test_partial_output_row_count_over_is_rejected(self):
        rows = [_row(i) for i in range(11)]
        with self.assertRaises(I2AnalysisError):
            validate_index_integrity(rows, 10, "test")


class TestLoadTaskRowsExplicitIndex(unittest.TestCase):
    def test_load_task_rows_reads_explicit_dataset_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_jsonl(os.path.join(tmp, "trec.jsonl"), [_row(i) for i in range(5)])
            rows = load_task_rows(tmp, "trec")
            self.assertEqual([r["dataset_index"] for r in rows], [0, 1, 2, 3, 4])

    def test_load_task_rows_falls_back_to_position_if_index_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows_no_index = [{"pred": "p", "answers": ["a"], "all_classes": [], "length": 10} for _ in range(3)]
            _write_jsonl(os.path.join(tmp, "trec.jsonl"), rows_no_index)
            rows = load_task_rows(tmp, "trec")
            self.assertEqual([r["dataset_index"] for r in rows], [0, 1, 2])


# --- pairing integrity ----------------------------------------------------------

class TestPairingIntegrity(unittest.TestCase):
    def _all_data(self, layers=(0,), tasks=("trec",), n=5):
        return {
            (layer, family): {task: [_row(i) for i in range(n)] for task in tasks}
            for layer in layers
            for family in (KIVI, ROTATION)
        }

    def test_matching_pairing_passes(self):
        all_data = self._all_data()
        result = validate_pairing(all_data, layers=(0,), tasks=("trec",))
        self.assertEqual(result["status"], "PASS")

    def test_dataset_index_set_mismatch_rejected(self):
        all_data = self._all_data()
        all_data[(0, ROTATION)]["trec"] = [_row(i) for i in range(1, 6)]  # shifted index set
        with self.assertRaises(I2AnalysisError):
            validate_pairing(all_data, layers=(0,), tasks=("trec",))

    def test_field_mismatch_at_same_index_rejected(self):
        all_data = self._all_data()
        mismatched = _row(2)
        mismatched["answers"] = ["different answer"]
        rows = all_data[(0, ROTATION)]["trec"]
        rows[2] = mismatched
        with self.assertRaises(I2AnalysisError):
            validate_pairing(all_data, layers=(0,), tasks=("trec",))

    def test_never_reorders_to_force_a_match(self):
        """A same-multiset-but-different-order rotation file must still be
        accepted (pairing is by dataset_index value, not row order) -- but
        content at each shared index must still match exactly."""
        all_data = self._all_data()
        rows = all_data[(0, ROTATION)]["trec"]
        all_data[(0, ROTATION)]["trec"] = list(reversed(rows))
        result = validate_pairing(all_data, layers=(0,), tasks=("trec",))
        self.assertEqual(result["status"], "PASS")


# --- paired, task-stratified bootstrap ------------------------------------------

class TestBootstrapPairing(unittest.TestCase):
    def test_same_seed_gives_identical_bootstrap(self):
        rng = np.random.default_rng(0)
        sample_scores = {
            (0, KIVI): {"trec": rng.uniform(0, 100, 20)},
            (0, ROTATION): {"trec": rng.uniform(0, 100, 20)},
        }
        conditions = list(sample_scores)
        b1 = bootstrap_task_condition_scores(sample_scores, ["trec"], conditions, 500, seed=42)
        b2 = bootstrap_task_condition_scores(sample_scores, ["trec"], conditions, 500, seed=42)
        np.testing.assert_array_equal(b1[(0, KIVI)]["trec"], b2[(0, KIVI)]["trec"])

    def test_shared_index_array_keeps_delta_paired(self):
        """If kivi and rotation_kivi are resampled with the SAME index
        array (the pairing contract), a per-replicate delta computed from
        already-identical underlying scores must be exactly zero every
        replicate -- proving the two families really do share indices."""
        scores = np.arange(30, dtype=np.float64)
        sample_scores = {(0, KIVI): {"trec": scores}, (0, ROTATION): {"trec": scores.copy()}}
        conditions = list(sample_scores)
        boot = bootstrap_task_condition_scores(sample_scores, ["trec"], conditions, 200, seed=1)
        deltas = boot[(0, ROTATION)]["trec"] - boot[(0, KIVI)]["trec"]
        np.testing.assert_array_equal(deltas, np.zeros_like(deltas))

    def test_layer_aggregate_boot_is_equal_task_mean_of_task_boot(self):
        rng = np.random.default_rng(2)
        tasks = ["trec", "lcc"]
        sample_scores = {
            (0, KIVI): {t: rng.uniform(0, 100, 20) for t in tasks},
            (0, ROTATION): {t: rng.uniform(0, 100, 20) for t in tasks},
        }
        conditions = list(sample_scores)
        boot_task = bootstrap_task_condition_scores(sample_scores, tasks, conditions, 300, seed=7)
        task_delta_boot = bootstrap_task_deltas(boot_task, layers=(0,), tasks=tasks)
        layer_boot = bootstrap_layer_aggregate_deltas(task_delta_boot, layers=(0,), tasks=tasks)
        expected = (task_delta_boot[(0, "trec")] + task_delta_boot[(0, "lcc")]) / 2.0
        np.testing.assert_allclose(layer_boot[0], expected)


class TestFamilyEffectRangeBootstrap(unittest.TestCase):
    def test_range_lets_extremum_layer_vary_per_replicate(self):
        """The range bootstrap must not fix a single (max_layer, min_layer)
        pair chosen from the observed point estimate -- it must recompute
        max/min PER REPLICATE. Construct two layers whose per-replicate
        boot values sometimes cross, and verify the range is never
        negative (a fixed-pair contrast could go negative when the
        replicate's ordering flips)."""
        rng = np.random.default_rng(3)
        layer_boot = {0: rng.normal(0, 1, 5000), 4: rng.normal(0, 1, 5000)}  # heavily overlapping distributions
        range_boot = bootstrap_family_effect_range(layer_boot, layers=(0, 4))
        self.assertTrue(np.all(range_boot >= 0))


# --- strong-crossover / mixed-point-estimate definitions ------------------------

class TestCrossoverDefinitions(unittest.TestCase):
    def test_strong_crossover_requires_one_ci_fully_positive_and_one_fully_negative(self):
        layer_point = {0: 1.0, 4: -1.0}
        layer_ci = {0: (0.5, 1.5), 4: (-1.5, -0.5)}
        pos, neg = strong_crossover_layers(layer_point, layer_ci, layers=(0, 4))
        self.assertEqual(pos, [0])
        self.assertEqual(neg, [4])
        result = classify_interpretation(layer_point, layer_ci, layers=(0, 4))
        self.assertEqual(result["category"], "STRONG_CROSSOVER")
        self.assertTrue(result["strong_crossover"])

    def test_ci_crossing_zero_on_both_sides_is_not_strong_crossover(self):
        """Mixed point-estimate SIGN alone, with both CIs crossing zero,
        must be MIXED_POINT_ESTIMATE_DIRECTION, never STRONG_CROSSOVER."""
        layer_point = {0: 1.0, 4: -1.0}
        layer_ci = {0: (-0.5, 2.0), 4: (-2.0, 0.5)}  # both cross zero
        result = classify_interpretation(layer_point, layer_ci, layers=(0, 4))
        self.assertEqual(result["category"], "MIXED_POINT_ESTIMATE_DIRECTION")
        self.assertFalse(result["strong_crossover"])

    def test_one_sided_ci_alone_without_opposite_sign_is_not_strong_crossover(self):
        # Both layers positive point estimate; only one CI excludes zero.
        layer_point = {0: 1.0, 4: 0.5}
        layer_ci = {0: (0.2, 1.8), 4: (-0.3, 1.3)}
        result = classify_interpretation(layer_point, layer_ci, layers=(0, 4))
        self.assertNotEqual(result["category"], "STRONG_CROSSOVER")
        self.assertEqual(result["category"], "UNIFORM_ROTATION_ADVANTAGE")

    def test_uniform_rotation_advantage(self):
        layer_point = {0: 1.0, 4: 2.0}
        layer_ci = {0: (-0.1, 2.0), 4: (0.5, 3.0)}
        result = classify_interpretation(layer_point, layer_ci, layers=(0, 4))
        self.assertEqual(result["category"], "UNIFORM_ROTATION_ADVANTAGE")

    def test_uniform_kivi_advantage(self):
        layer_point = {0: -1.0, 4: -2.0}
        layer_ci = {0: (-2.0, 0.1), 4: (-3.0, -0.5)}
        result = classify_interpretation(layer_point, layer_ci, layers=(0, 4))
        self.assertEqual(result["category"], "UNIFORM_KIVI_ADVANTAGE")

    def test_no_clear_family_difference_when_a_point_estimate_is_exactly_zero(self):
        layer_point = {0: 0.0, 4: 1.0}
        layer_ci = {0: (-0.5, 0.5), 4: (-0.2, 2.0)}
        result = classify_interpretation(layer_point, layer_ci, layers=(0, 4))
        self.assertEqual(result["category"], "NO_CLEAR_FAMILY_DIFFERENCE")

    def test_all_five_categories_are_the_frozen_vocabulary(self):
        self.assertEqual(
            set(INTERPRETATION_CATEGORIES),
            {"UNIFORM_ROTATION_ADVANTAGE", "UNIFORM_KIVI_ADVANTAGE", "MIXED_POINT_ESTIMATE_DIRECTION",
             "STRONG_CROSSOVER", "NO_CLEAR_FAMILY_DIFFERENCE"},
        )


# --- task-conditioned matrix -----------------------------------------------------

class TestTaskConditionedMatrix(unittest.TestCase):
    def test_matrix_shape_and_values(self):
        tasks = ["trec", "lcc"]
        layers = (0, 4)
        task_deltas = {(l, t): float(l + len(t)) for l in layers for t in tasks}
        matrix = task_conditioned_matrix(task_deltas, layers=layers, tasks=tasks)
        self.assertEqual(set(matrix), set(layers))
        for l in layers:
            self.assertEqual(set(matrix[l]), set(tasks))
            for t in tasks:
                self.assertEqual(matrix[l][t], task_deltas[(l, t)])


# --- C1 must never bias I2 -------------------------------------------------------

class TestC1NeverUsedAsEndpoint(unittest.TestCase):
    def test_c1_note_is_a_plain_string_not_read_by_any_computation(self):
        # The note is a module-level constant only -- assert it is a string
        # (data, not code) and that computing deltas/aggregates/bootstrap
        # never references it or special-cases dataset_index 122 / layer 0.
        self.assertIsInstance(C1_NON_SELECTION_NOTE, str)
        self.assertIn("122", C1_NON_SELECTION_NOTE)
        self.assertIn("NOT an I2 endpoint", C1_NON_SELECTION_NOTE)

    def test_source_never_special_cases_dataset_index_122_in_computation(self):
        import ast
        import analysis.analyze_i2_layer_family_sensitivity as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == 122:
                # The only permitted occurrence of the literal 122 is inside
                # the C1_NON_SELECTION_NOTE string constant itself, which is
                # already asserted above; a numeric 122 anywhere else would
                # indicate C1's sample leaking into a computation.
                self.fail("Literal 122 found outside the C1 provenance note -- must never bias computation.")

    def test_delta_computation_treats_layer_0_like_any_other_layer(self):
        tasks = ["trec"]
        task_mean_raw = {
            (0, KIVI): {"trec": 50.0}, (0, ROTATION): {"trec": 55.0},
            (4, KIVI): {"trec": 50.0}, (4, ROTATION): {"trec": 55.0},
        }
        deltas = compute_task_deltas(task_mean_raw, layers=(0, 4), tasks=tasks)
        self.assertEqual(deltas[(0, "trec")], deltas[(4, "trec")])


if __name__ == "__main__":
    unittest.main()
