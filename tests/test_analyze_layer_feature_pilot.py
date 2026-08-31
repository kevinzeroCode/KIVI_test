"""CPU-only tests for analysis/analyze_layer_feature_pilot.py (Stage G4).

All feature/sensitivity data here is synthetic. No real
outputs/layer_feature_pilot/ data or GPU is touched -- these tests prove
the aggregation, join, and reuse-not-reimplement logic is correct and
deterministic, not that any particular scientific conclusion holds.
"""
import csv
import json
import os
import tempfile
import unittest

from analysis.analyze_layer_feature_pilot import (
    AnalysisError,
    aggregate_task_layer_axis,
    build_layer0_diagnostic_table,
    build_rho_records,
    build_task_layer_matrices,
    discover_single_features_jsonl,
    load_f0_layer0_sensitivity,
    load_stage_e_sensitivity,
    render_report,
    run_analysis,
    variance_nonzero,
)
from analysis.feature_pilot_gate import PRIMARY_STAGE_E_TASKS
from scripts.collect_layer_features import PRIMARY_STAGE_E_LAYERS


def _row(task, sample_idx, layer_idx, axis, relative_l2, p99_abs=1.0, outlier_fraction=0.1, **extra):
    row = {
        "task": task, "sample_idx": sample_idx, "layer_idx": layer_idx, "tensor_axis": axis,
        "relative_l2": relative_l2, "mse": 0.01, "max_abs_error": 0.5,
        "mean": 0.0, "std": 0.4, "variance": 0.16, "max_abs": 3.0,
        "p50_abs": 0.2, "p95_abs": 1.0, "p99_abs": p99_abs, "outlier_fraction": outlier_fraction,
    }
    row.update(extra)
    return row


class AggregateTaskLayerAxisTest(unittest.TestCase):
    def test_exact_four_sample_equal_weight_mean(self):
        rows = [
            _row("t", 0, 0, "key", relative_l2=0.1),
            _row("t", 1, 0, "key", relative_l2=0.2),
            _row("t", 2, 0, "key", relative_l2=0.3),
            _row("t", 3, 0, "key", relative_l2=0.4),
        ]
        agg = aggregate_task_layer_axis(rows)
        entry = agg[("t", 0, "key")]
        self.assertAlmostEqual(entry["mean_relative_l2"], 0.25, places=10)  # simple unweighted mean
        self.assertAlmostEqual(entry["median_relative_l2"], 0.25, places=10)
        self.assertEqual(entry["n_samples"], 4)

    def test_not_weighted_by_token_counts(self):
        # Two samples with wildly different token counts must contribute
        # EQUALLY to the mean -- token fields aren't even read by the
        # aggregation function.
        rows = [
            _row("t", 0, 0, "key", relative_l2=0.0, input_tokens=1, distribution_tokens=1, quantized_tokens=1, residual_tokens=0),
            _row("t", 1, 0, "key", relative_l2=1.0, input_tokens=100000, distribution_tokens=100000, quantized_tokens=99999, residual_tokens=1),
            _row("t", 2, 0, "key", relative_l2=0.0, input_tokens=1, distribution_tokens=1, quantized_tokens=1, residual_tokens=0),
            _row("t", 3, 0, "key", relative_l2=1.0, input_tokens=100000, distribution_tokens=100000, quantized_tokens=99999, residual_tokens=1),
        ]
        entry = aggregate_task_layer_axis(rows)[("t", 0, "key")]
        self.assertAlmostEqual(entry["mean_relative_l2"], 0.5, places=10)  # plain mean, not token-weighted

    def test_wrong_sample_count_raises(self):
        rows = [_row("t", 0, 0, "key", relative_l2=0.1), _row("t", 1, 0, "key", relative_l2=0.2)]
        with self.assertRaises(AnalysisError):
            aggregate_task_layer_axis(rows)

    def test_duplicate_sample_idx_raises(self):
        rows = [_row("t", 0, 0, "key", relative_l2=x) for x in (0.1, 0.2, 0.3, 0.3)]
        rows[3]["sample_idx"] = 2  # duplicate of rows[2]
        with self.assertRaises(AnalysisError):
            aggregate_task_layer_axis(rows)

    def test_deterministic_group_ordering(self):
        rows = []
        for task in ("b_task", "a_task"):
            for i in range(4):
                rows.append(_row(task, i, 5, "key", relative_l2=0.1))
        agg = aggregate_task_layer_axis(rows)
        self.assertEqual(list(agg.keys()), sorted(agg.keys()))

    def test_real_grid_produces_32_observations_per_axis(self):
        rows = []
        for task in PRIMARY_STAGE_E_TASKS:
            for layer in PRIMARY_STAGE_E_LAYERS:
                for axis in ("key", "value"):
                    for sample_idx in range(4):
                        rows.append(_row(task, sample_idx, layer, axis, relative_l2=0.1 * sample_idx))
        agg = aggregate_task_layer_axis(rows)
        key_obs = [k for k in agg if k[2] == "key"]
        value_obs = [k for k in agg if k[2] == "value"]
        self.assertEqual(len(key_obs), 32)
        self.assertEqual(len(value_obs), 32)
        self.assertEqual(len(agg), 64)


class VarianceNonzeroTest(unittest.TestCase):
    def test_constant_is_zero_variance(self):
        self.assertFalse(variance_nonzero([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

    def test_varying_is_nonzero_variance(self):
        self.assertTrue(variance_nonzero([1.0, 2.0, 1.0, 1.0]))


class SensitivityLoadersTest(unittest.TestCase):
    def test_load_stage_e_sensitivity_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sens.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["layer_idx", "axis", "task", "baseline_score", "perturbed_score", "delta"])
                for layer in (0, 4):
                    for axis in ("key", "value"):
                        for task in ("ta", "tb"):
                            w.writerow([layer, axis, task, 10.0, 10.5, 0.5])
            sens = load_stage_e_sensitivity(path, tasks=("ta", "tb"), layers=(0, 4))
            self.assertEqual(sens[(0, "key", "ta")], 0.5)
            self.assertEqual(len(sens), 8)

    def test_load_stage_e_sensitivity_missing_row_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sens.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["layer_idx", "axis", "task", "baseline_score", "perturbed_score", "delta"])
                w.writerow([0, "key", "ta", 10.0, 10.5, 0.5])
            with self.assertRaises(AnalysisError):
                load_stage_e_sensitivity(path, tasks=("ta", "tb"), layers=(0, 4))

    def test_load_f0_layer0_sensitivity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "f0.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["task", "n_samples", "baseline_score", "key_score", "value_score", "key_delta", "value_delta", "axis_difference"])
                w.writerow(["multifieldqa_en", 150, 43.42, 43.31, 43.44, -0.1072, 0.0249, -0.1322])
                w.writerow(["samsum", 200, 40.8, 40.87, 40.74, 0.0646, -0.068, 0.1326])
            sens = load_f0_layer0_sensitivity(path, tasks=("multifieldqa_en", "samsum"))
            self.assertAlmostEqual(sens[("key", "multifieldqa_en")], -0.1072, places=6)
            self.assertAlmostEqual(sens[("value", "samsum")], -0.068, places=6)


class BuildTaskLayerMatricesTest(unittest.TestCase):
    def test_join_matches_task_and_layer(self):
        agg = {("t1", 0, "key"): {"mean_relative_l2": 0.1}, ("t1", 1, "key"): {"mean_relative_l2": 0.2}}
        sens = {(0, "key", "t1"): -0.5, (1, "key", "t1"): -0.9}
        x, y = build_task_layer_matrices(agg, sens, "key", "mean_relative_l2", tasks=("t1",), layers=(0, 1))
        self.assertEqual(x["t1"], {0: 0.1, 1: 0.2})
        self.assertEqual(y["t1"], {0: -0.5, 1: -0.9})

    def test_missing_observation_raises(self):
        agg = {("t1", 0, "key"): {"mean_relative_l2": 0.1}}
        sens = {(0, "key", "t1"): -0.5, (1, "key", "t1"): -0.9}
        with self.assertRaises(AnalysisError):
            build_task_layer_matrices(agg, sens, "key", "mean_relative_l2", tasks=("t1",), layers=(0, 1))


class BuildRhoRecordsTest(unittest.TestCase):
    def test_known_trec_value_undefined_case_reproduced_end_to_end(self):
        # Constant sensitivity across all 8 layers (like real trec/Value) ->
        # raw_rho undefined, gate_rho=0.0, never fabricated as a real number.
        x = {"trec": {l: 0.1 * i for i, l in enumerate(PRIMARY_STAGE_E_LAYERS)}}
        y = {"trec": {l: 0.0 for l in PRIMARY_STAGE_E_LAYERS}}  # constant, like real trec/Value
        records, raw_rho_by_task = build_rho_records(x, y, tasks=("trec",))
        self.assertIsNone(raw_rho_by_task["trec"])
        self.assertIsNone(records["trec"]["raw_rho"])
        self.assertEqual(records["trec"]["gate_rho"], 0.0)
        self.assertFalse(records["trec"]["target_variance_nonzero"])
        self.assertTrue(records["trec"]["feature_variance_nonzero"])

    def test_monotonic_case_gives_expected_sign(self):
        x = {"t": {0: 1, 1: 2, 2: 3, 3: 4}}
        y = {"t": {0: -1, 1: -2, 2: -3, 3: -4}}
        records, _ = build_rho_records(x, y, tasks=("t",))
        self.assertAlmostEqual(records["t"]["raw_rho"], -1.0, places=6)
        self.assertEqual(records["t"]["raw_rho"], records["t"]["gate_rho"])


class BuildLayer0DiagnosticTableTest(unittest.TestCase):
    def test_lcc_from_primary_others_from_diagnostic(self):
        primary_agg = {("lcc", 0, "key"): {"mean_relative_l2": 0.2, "median_relative_l2": 0.2, "mean_p99_abs": 1.5, "mean_outlier_fraction": 0.02}}
        diagnostic_agg = {
            ("multifieldqa_en", 0, "key"): {"mean_relative_l2": 0.19, "median_relative_l2": 0.19, "mean_p99_abs": 1.4, "mean_outlier_fraction": 0.01},
            ("samsum", 0, "key"): {"mean_relative_l2": 0.21, "median_relative_l2": 0.21, "mean_p99_abs": 1.6, "mean_outlier_fraction": 0.03},
        }
        stage_e_sens = {(0, "key", "lcc"): 2.64}
        f0_sens = {("key", "multifieldqa_en"): -0.1072, ("key", "samsum"): 0.0646}
        rows = build_layer0_diagnostic_table(primary_agg, diagnostic_agg, stage_e_sens, f0_sens, "key")
        by_task = {r["task"]: r for r in rows}
        self.assertEqual(by_task["lcc"]["SignedSensitivity"], 2.64)
        self.assertEqual(by_task["lcc"]["mean_relative_l2"], 0.2)
        self.assertEqual(by_task["multifieldqa_en"]["SignedSensitivity"], -0.1072)
        self.assertEqual(by_task["samsum"]["SignedSensitivity"], 0.0646)


class DistributionFeaturesCannotRescueGateTest(unittest.TestCase):
    """Structural guard: the gate is evaluated ONLY on relative_l2's
    raw_rho/gate_rho -- p99_abs/outlier_fraction correlations are computed
    completely separately (distribution_feature_correlations_rows) and are
    never passed into evaluate_feature_pilot_gate."""

    def test_gate_function_signature_has_no_distribution_feature_params(self):
        import inspect
        from analysis.feature_pilot_gate import evaluate_feature_pilot_gate

        sig = inspect.signature(evaluate_feature_pilot_gate)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["key_raw_rho_by_task", "key_layer0_raw_rho", "value_raw_rho_by_task", "value_layer0_raw_rho", "threshold", "tasks"])


class DiscoverSingleFeaturesJsonlTest(unittest.TestCase):
    def test_exactly_one_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run1")
            os.makedirs(run_dir)
            path = os.path.join(run_dir, "features.jsonl")
            with open(path, "w") as f:
                f.write("{}\n")
            found = discover_single_features_jsonl(tmp)
            self.assertEqual(found, path)

    def test_zero_found_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AnalysisError):
                discover_single_features_jsonl(tmp)

    def test_multiple_found_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("run1", "run2"):
                run_dir = os.path.join(tmp, name)
                os.makedirs(run_dir)
                with open(os.path.join(run_dir, "features.jsonl"), "w") as f:
                    f.write("{}\n")
            with self.assertRaises(AnalysisError):
                discover_single_features_jsonl(tmp)


class RunAnalysisEndToEndSyntheticTest(unittest.TestCase):
    """Full pipeline on a small, fully synthetic, self-consistent dataset
    (real task/layer names so the module's fixed constants are satisfied,
    but fabricated numeric values) -- proves deterministic output
    generation without touching any real collected feature data."""

    def _write_jsonl(self, path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def _write_csv(self, path, header, rows):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for row in rows:
                w.writerow(row)

    def test_full_pipeline_deterministic_and_matches_expected_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            primary_root = os.path.join(tmp, "primary")
            diagnostic_root = os.path.join(tmp, "diagnostic")
            os.makedirs(os.path.join(primary_root, "run1"))
            os.makedirs(os.path.join(diagnostic_root, "run1"))

            primary_rows = []
            for task in PRIMARY_STAGE_E_TASKS:
                for layer in PRIMARY_STAGE_E_LAYERS:
                    for axis in ("key", "value"):
                        for sample_idx in range(4):
                            primary_rows.append(_row(task, sample_idx, layer, axis, relative_l2=0.1 + 0.01 * sample_idx))
            self._write_jsonl(os.path.join(primary_root, "run1", "features.jsonl"), primary_rows)

            diagnostic_rows = []
            for task in ("multifieldqa_en", "samsum"):
                for sample_idx in range(4):
                    for axis in ("key", "value"):
                        diagnostic_rows.append(_row(task, sample_idx, 0, axis, relative_l2=0.15 + 0.01 * sample_idx))
            self._write_jsonl(os.path.join(diagnostic_root, "run1", "features.jsonl"), diagnostic_rows)

            stage_e_csv = os.path.join(tmp, "stage_e.csv")
            stage_e_header = ["layer_idx", "axis", "task", "baseline_score", "perturbed_score", "delta"]
            stage_e_rows = []
            for task in PRIMARY_STAGE_E_TASKS:
                for layer in PRIMARY_STAGE_E_LAYERS:
                    for axis in ("key", "value"):
                        stage_e_rows.append([layer, axis, task, 50.0, 50.0 + layer * 0.01, layer * 0.01])
            self._write_csv(stage_e_csv, stage_e_header, stage_e_rows)

            f0_csv = os.path.join(tmp, "f0.csv")
            f0_header = ["task", "n_samples", "baseline_score", "key_score", "value_score", "key_delta", "value_delta", "axis_difference"]
            f0_rows = [
                ["multifieldqa_en", 150, 43.0, 43.1, 42.9, 0.1, -0.1, 0.2],
                ["samsum", 200, 40.0, 40.2, 39.9, 0.2, -0.1, 0.3],
            ]
            self._write_csv(f0_csv, f0_header, f0_rows)

            out1 = os.path.join(tmp, "out1")
            out2 = os.path.join(tmp, "out2")
            summary1 = run_analysis(primary_root, diagnostic_root, stage_e_csv, f0_csv, out1)
            render_report(summary1, out1)
            summary2 = run_analysis(primary_root, diagnostic_root, stage_e_csv, f0_csv, out2)
            render_report(summary2, out2)

            self.assertIn(summary1["gate_result"]["FEATURE_PILOT_STAGE"], ("GO", "NO_GO"))
            self.assertEqual(summary1["gate_result"]["FEATURE_PILOT_STAGE"], summary2["gate_result"]["FEATURE_PILOT_STAGE"])
            self.assertEqual(summary1["task_spearman_relative_l2"], summary2["task_spearman_relative_l2"])

            with open(os.path.join(out1, "task_spearman.csv")) as f1, open(os.path.join(out2, "task_spearman.csv")) as f2:
                self.assertEqual(f1.read(), f2.read())

            for name in ("aggregated_features.csv", "task_spearman.csv", "diagnostic_layer0.csv", "distribution_feature_correlations.csv", "secondary_features.csv", "gate_results.csv", "loto.csv", "summary.json", "report.md"):
                self.assertTrue(os.path.exists(os.path.join(out1, name)), name)


if __name__ == "__main__":
    unittest.main()
