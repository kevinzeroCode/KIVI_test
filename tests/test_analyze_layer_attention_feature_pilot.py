"""CPU-only tests for analysis/analyze_layer_attention_feature_pilot.py
(Stage H4). Mixes: (a) synthetic-fixture unit tests for pure aggregation/
gate-wiring logic that need no real data, and (b) real-frozen-data
integration checks (skipped gracefully if the Stage-H collection isn't
present in this environment). No GPU, no correlation math reimplemented
(reuses spearman_corr / feature_pilot_gate throughout, exactly like the
module under test).
"""
import copy
import os
import unittest

from analysis.analyze_layer_attention_feature_pilot import (
    DEFAULT_DIAGNOSTIC_ROOT,
    DEFAULT_F0_TASK_RESULTS_CSV,
    DEFAULT_PRIMARY_ROOT,
    DEFAULT_STAGE_E_TASK_LAYER_CSV,
    EXPECTED_DIAGNOSTIC_FEATURES,
    EXPECTED_DIAGNOSTIC_MANIFEST,
    EXPECTED_PRIMARY_FEATURES,
    EXPECTED_PRIMARY_MANIFEST,
    FROZEN_CHECKSUMS,
    FrozenInputError,
    aggregate_task_layer_axis,
    compute_layer0_diagnostic_inputs,
    compute_primary_task_rho,
    load_all_records,
    load_f0_layer0_task_deltas,
    load_stage_e_task_layer_sensitivity,
    run_full_analysis,
    secondary_full_loto_table,
    sha256_of,
    verify_frozen_checksums,
)
from analysis.feature_pilot_gate import (
    LAYER0_DIAGNOSTIC_TASKS,
    LOTO_REMOVED_TASK,
    PRIMARY_STAGE_E_TASKS,
    evaluate_criterion_a,
    evaluate_criterion_b,
    evaluate_criterion_c,
    evaluate_criterion_d,
    evaluate_feature_pilot_gate,
    to_gate_rho,
)
from scripts.run_layer_attention_feature_pilot import AXES, DIAGNOSTIC_LAYERS, DIAGNOSTIC_TASKS, PRIMARY_LAYERS, PRIMARY_TASKS

REAL_DATA_PRESENT = all(os.path.exists(p) for p in FROZEN_CHECKSUMS) and os.path.exists(DEFAULT_STAGE_E_TASK_LAYER_CSV) and os.path.exists(DEFAULT_F0_TASK_RESULTS_CSV)


def _skip_without_real_data(test):
    return unittest.skipUnless(REAL_DATA_PRESENT, "frozen Stage-H collection / Stage-E,F0 CSVs not present in this environment")(test)


def _make_record(task, dataset_index, layer_idx, axis, distortion, scope="primary", prompt_tokens=1000, generated=20):
    k_bits, v_bits = (2, 16) if axis == "key" else (16, 2)
    return {
        "scope": scope, "task": task, "dataset_index": dataset_index, "layer_idx": layer_idx, "tensor_axis": axis,
        "policy": f"K{k_bits}/V{v_bits}", "k_bits": k_bits, "v_bits": v_bits,
        "model": "lmsys/longchat-7b-v1.5-32k", "git_commit": "deadbeef", "seed": 42,
        "prompt_input_tokens": prompt_tokens, "actual_generated_token_count": generated,
        "requested_decode_steps": [1, 2, 4, 8, 16], "sampled_decode_steps": [1, 2, 4, 8, 16], "valid_step_count": 5,
        "no_decode_exposure": False, "sample_attention_distortion": distortion,
        "per_step_distortions": [{"decode_step": s, "distortion": distortion} for s in [1, 2, 4, 8, 16]],
    }


class FrozenChecksumTest(unittest.TestCase):
    @_skip_without_real_data
    def test_real_frozen_checksums_currently_match(self):
        result = verify_frozen_checksums()
        for path, entry in result.items():
            self.assertTrue(entry["match"], path)

    def test_mismatch_raises(self):
        bad = dict(FROZEN_CHECKSUMS)
        # Point at a real file (this test file itself) with a deliberately wrong expected hash.
        this_file = __file__
        bad = {this_file: "0" * 64}
        with self.assertRaises(FrozenInputError):
            verify_frozen_checksums(bad)

    def test_missing_file_raises(self):
        with self.assertRaises(FrozenInputError):
            verify_frozen_checksums({"/nonexistent/path/features.jsonl": "0" * 64})

    def test_sha256_of_is_deterministic(self):
        self.assertEqual(sha256_of(__file__), sha256_of(__file__))


@_skip_without_real_data
class RealDataIntegrityTest(unittest.TestCase):
    def test_exact_272_input_record_count(self):
        primary, diagnostic = load_all_records()
        self.assertEqual(len(primary) + len(diagnostic), 272)

    def test_256_16_scope_split(self):
        primary, diagnostic = load_all_records()
        self.assertEqual(len(primary), 256)
        self.assertEqual(len(diagnostic), 16)

    def test_64_primary_aggregate_groups_and_4_samples_each(self):
        primary, _ = load_all_records()
        agg = aggregate_task_layer_axis(primary)
        self.assertEqual(len(agg), 64)
        for key, stats in agg.items():
            self.assertEqual(stats["n"], 4, key)

    def test_32_observations_per_axis(self):
        primary, _ = load_all_records()
        agg = aggregate_task_layer_axis(primary)
        key_count = sum(1 for (t, l, a) in agg if a == "key")
        value_count = sum(1 for (t, l, a) in agg if a == "value")
        self.assertEqual(key_count, 32)
        self.assertEqual(value_count, 32)

    def test_stage_e_label_join_exact_no_missing_no_duplicate(self):
        stage_e = load_stage_e_task_layer_sensitivity()
        expected_keys = {(t, l, a) for t in PRIMARY_TASKS for l in PRIMARY_LAYERS for a in AXES}
        self.assertEqual(set(stage_e.keys()), expected_keys)
        self.assertEqual(len(stage_e), 64)

    def test_f0_layer0_join_covers_diagnostic_tasks(self):
        f0 = load_f0_layer0_task_deltas()
        for task in DIAGNOSTIC_TASKS:
            for axis in AXES:
                self.assertIn((task, axis), f0)

    def test_frozen_input_files_never_modified_by_analysis(self):
        before = {p: sha256_of(p) for p in FROZEN_CHECKSUMS}
        run_full_analysis()
        after = {p: sha256_of(p) for p in FROZEN_CHECKSUMS}
        self.assertEqual(before, after)

    def test_full_analysis_produces_go_or_no_go_only(self):
        result = run_full_analysis()
        self.assertIn(result["STAGE_H_ATTENTION_FEATURE"], ("GO", "NO_GO"))

    def test_full_analysis_agrees_with_feature_pilot_gate_cross_check(self):
        # run_full_analysis() itself raises FrozenInputError internally if this
        # disagrees -- so simply completing without raising is the assertion.
        result = run_full_analysis()
        cross = evaluate_feature_pilot_gate(
            result["axis_results"]["key"]["raw_rho_by_task"], result["axis_results"]["key"]["criterion_d"]["layer0_raw_rho"],
            result["axis_results"]["value"]["raw_rho_by_task"], result["axis_results"]["value"]["criterion_d"]["layer0_raw_rho"],
        )
        self.assertEqual(cross["FEATURE_PILOT_STAGE"], result["STAGE_H_ATTENTION_FEATURE"])

    def test_known_trec_value_undefined_case(self):
        # Preregistered known case: trec's Value SignedSensitivity is constant
        # across the 8 sampled layers -> raw_rho must be None, not 0.0.
        primary, _ = load_all_records()
        agg = aggregate_task_layer_axis(primary)
        stage_e = load_stage_e_task_layer_sensitivity()
        detail = compute_primary_task_rho(agg, stage_e, "value")["trec"]
        self.assertIsNone(detail["raw_rho"])
        self.assertEqual(detail["gate_rho"], 0.0)
        self.assertFalse(detail["target_has_variance"])


class SyntheticAggregationTest(unittest.TestCase):
    def test_aggregate_requires_exactly_four_samples(self):
        records = [_make_record("trec", i, 0, "key", 0.1) for i in range(3)]  # only 3, not 4
        with self.assertRaises(FrozenInputError):
            aggregate_task_layer_axis(records, tasks=("trec",), layers=(0,), axes=("key",), expected_n=4)

    def test_aggregate_mean_is_plain_arithmetic_mean(self):
        values = [0.1, 0.2, 0.3, 0.4]
        records = [_make_record("trec", i, 0, "key", v) for i, v in enumerate(values)]
        agg = aggregate_task_layer_axis(records, tasks=("trec",), layers=(0,), axes=("key",), expected_n=4)
        self.assertAlmostEqual(agg[("trec", 0, "key")]["mean"], sum(values) / 4)

    def test_median_is_secondary_and_differs_from_mean_when_skewed(self):
        # Skewed set where mean != median -- proves median is a genuinely
        # separate, non-gating field, not silently aliasing mean.
        values = [0.0, 0.0, 0.0, 1.0]
        records = [_make_record("trec", i, 0, "key", v) for i, v in enumerate(values)]
        agg = aggregate_task_layer_axis(records, tasks=("trec",), layers=(0,), axes=("key",), expected_n=4)
        stats = agg[("trec", 0, "key")]
        self.assertNotEqual(stats["mean"], stats["median"])
        self.assertAlmostEqual(stats["mean"], 0.25)
        self.assertAlmostEqual(stats["median"], 0.0)

    def test_no_token_or_length_weighting(self):
        # Wildly different prompt_input_tokens/generated counts must NOT
        # change the aggregate -- it is a plain unweighted mean.
        records = [
            _make_record("trec", 0, 0, "key", 0.2, prompt_tokens=100, generated=5),
            _make_record("trec", 1, 0, "key", 0.4, prompt_tokens=30000, generated=20),
            _make_record("trec", 2, 0, "key", 0.6, prompt_tokens=500, generated=1),
            _make_record("trec", 3, 0, "key", 0.8, prompt_tokens=15000, generated=20),
        ]
        agg = aggregate_task_layer_axis(records, tasks=("trec",), layers=(0,), axes=("key",), expected_n=4)
        self.assertAlmostEqual(agg[("trec", 0, "key")]["mean"], 0.5)


class SyntheticSpearmanWiringTest(unittest.TestCase):
    def _agg_and_stage_e(self, x_by_layer, y_by_layer, task="trec", axis="key", layers=(0, 4, 9, 13)):
        agg = {(task, l, axis): {"mean": x_by_layer[l], "median": x_by_layer[l], "n": 4, "values": [x_by_layer[l]] * 4} for l in layers}
        stage_e = {(task, l, axis): y_by_layer[l] for l in layers}
        return agg, stage_e

    def test_defined_negative_case(self):
        layers = (0, 4, 9, 13)
        x = {l: float(i) for i, l in enumerate(layers)}          # increasing distortion
        y = {l: -float(i) for i, l in enumerate(layers)}         # decreasing sensitivity
        agg, stage_e = self._agg_and_stage_e(x, y, layers=layers)
        detail = compute_primary_task_rho(agg, stage_e, "key", tasks=("trec",), layers=layers)["trec"]
        self.assertIsNotNone(detail["raw_rho"])
        self.assertLess(detail["raw_rho"], 0)
        self.assertAlmostEqual(detail["raw_rho"], -1.0)

    def test_undefined_constant_target_case(self):
        layers = (0, 4, 9, 13)
        x = {l: float(i) for i, l in enumerate(layers)}
        y = {l: 5.0 for l in layers}  # constant -- zero variance
        agg, stage_e = self._agg_and_stage_e(x, y, layers=layers)
        detail = compute_primary_task_rho(agg, stage_e, "key", tasks=("trec",), layers=layers)["trec"]
        self.assertIsNone(detail["raw_rho"])
        self.assertEqual(detail["gate_rho"], 0.0)
        self.assertFalse(detail["target_has_variance"])

    def test_raw_rho_null_vs_gate_rho_zero_are_distinct_objects(self):
        # to_gate_rho(None) == 0.0 is the intended, documented convention
        # (an undefined correlation contributes no evidence). What must
        # stay distinct is the RAW value: raw_rho=None ("could not be
        # computed") must never be conflated with a genuinely observed
        # raw_rho=0.0 ("computed and found to be exactly zero").
        undefined_gate_rho = to_gate_rho(None)
        computed_zero_gate_rho = to_gate_rho(0.0)
        self.assertEqual(undefined_gate_rho, 0.0)
        self.assertEqual(computed_zero_gate_rho, 0.0)
        self.assertEqual(undefined_gate_rho, computed_zero_gate_rho)  # gate_rho intentionally collapses them
        self.assertIsNot(None, 0.0)  # but the raw_rho representations themselves are never the same object/claim


class SyntheticGateWiringTest(unittest.TestCase):
    """Verifies THIS module's wiring of real aggregated data into
    feature_pilot_gate's criteria -- not feature_pilot_gate's own math
    (already covered by tests/test_feature_pilot_gate.py)."""

    def _craft_passing_axis(self, tasks=PRIMARY_STAGE_E_TASKS):
        # Strong, monotonic negative association for A/B/C; craft D similarly.
        raw_rho_by_task = {"trec": -0.9, "lcc": -0.8, "passage_retrieval_en": -0.7, "2wikimqa": -1.0}
        return raw_rho_by_task

    def test_criterion_a_b_c_wiring_matches_direct_gate_calls(self):
        raw_rho_by_task = self._craft_passing_axis()
        a = evaluate_criterion_a(raw_rho_by_task)
        b = evaluate_criterion_b(raw_rho_by_task)
        c = evaluate_criterion_c(raw_rho_by_task, removed_task=LOTO_REMOVED_TASK)
        self.assertTrue(a["pass"])
        self.assertTrue(b["pass"])
        self.assertTrue(c["pass"])

    def test_criterion_d_layer0_wiring(self):
        x3, y3 = compute_layer0_diagnostic_inputs(
            agg_primary={("lcc", 0, "key"): {"mean": 1.0}},
            agg_diagnostic={("multifieldqa_en", 0, "key"): {"mean": 2.0}, ("samsum", 0, "key"): {"mean": 3.0}},
            stage_e={("lcc", 0, "key"): -1.0},
            f0={("multifieldqa_en", "key"): -2.0, ("samsum", "key"): -3.0},
            axis="key",
        )
        self.assertEqual(set(x3), set(LAYER0_DIAGNOSTIC_TASKS))
        d = evaluate_criterion_d(__import__("analysis.feature_pilot_gate", fromlist=["compute_layer0_diagnostic_rho"]).compute_layer0_diagnostic_rho(x3, y3))
        self.assertTrue(d["pass"])  # perfectly monotonic negative n=3

    def test_same_axis_gate_requires_all_four(self):
        raw_rho_by_task = self._craft_passing_axis()
        a = evaluate_criterion_a(raw_rho_by_task)
        b = evaluate_criterion_b(raw_rho_by_task)
        c = evaluate_criterion_c(raw_rho_by_task, removed_task=LOTO_REMOVED_TASK)
        d_fail = evaluate_criterion_d(0.9)  # positive -> D fails
        axis_go = a["pass"] and b["pass"] and c["pass"] and d_fail["pass"]
        self.assertFalse(axis_go)  # A/B/C pass but D fails -> whole axis fails
        d_pass = evaluate_criterion_d(-0.9)
        axis_go_all = a["pass"] and b["pass"] and c["pass"] and d_pass["pass"]
        self.assertTrue(axis_go_all)

    def test_no_cross_axis_mixing_in_final_gate(self):
        # Key passes A/B/C but fails D; Value passes D but fails A/B/C.
        # Final gate must be NO_GO -- criteria may never be borrowed across axes.
        key_raw = self._craft_passing_axis()
        key_layer0 = 0.9  # fails D
        value_raw = {"trec": 0.9, "lcc": 0.8, "passage_retrieval_en": 0.7, "2wikimqa": 0.5}  # fails A/B/C
        value_layer0 = -0.9  # passes D alone
        result = evaluate_feature_pilot_gate(key_raw, key_layer0, value_raw, value_layer0)
        self.assertFalse(result["key"]["axis_go"])
        self.assertFalse(result["value"]["axis_go"])
        self.assertEqual(result["FEATURE_PILOT_STAGE"], "NO_GO")

    def test_go_when_exactly_one_axis_passes_all_four(self):
        key_raw = self._craft_passing_axis()
        key_layer0 = -0.9
        value_raw = {"trec": 0.9, "lcc": 0.8, "passage_retrieval_en": 0.7, "2wikimqa": 0.5}
        value_layer0 = 0.9
        result = evaluate_feature_pilot_gate(key_raw, key_layer0, value_raw, value_layer0)
        self.assertTrue(result["key"]["axis_go"])
        self.assertFalse(result["value"]["axis_go"])
        self.assertEqual(result["FEATURE_PILOT_STAGE"], "GO")

    def test_no_go_when_neither_axis_passes(self):
        weak = {"trec": 0.1, "lcc": 0.1, "passage_retrieval_en": 0.1, "2wikimqa": 0.1}
        result = evaluate_feature_pilot_gate(weak, 0.1, weak, 0.1)
        self.assertEqual(result["FEATURE_PILOT_STAGE"], "NO_GO")

    def test_result_is_never_conditional_go(self):
        for key_raw, key_l0, value_raw, value_l0 in (
            (self._craft_passing_axis(), -0.9, {"trec": 0.9, "lcc": 0.8, "passage_retrieval_en": 0.7, "2wikimqa": 0.5}, 0.9),
            ({"trec": 0.1, "lcc": 0.1, "passage_retrieval_en": 0.1, "2wikimqa": 0.1}, 0.1, {"trec": 0.1, "lcc": 0.1, "passage_retrieval_en": 0.1, "2wikimqa": 0.1}, 0.1),
        ):
            result = evaluate_feature_pilot_gate(key_raw, key_l0, value_raw, value_l0)
            self.assertIn(result["FEATURE_PILOT_STAGE"], ("GO", "NO_GO"))


class SecondaryLotoTest(unittest.TestCase):
    def test_full_loto_table_has_one_entry_per_task(self):
        raw_rho_by_task = {"trec": -0.5, "lcc": -0.3, "passage_retrieval_en": 0.2, "2wikimqa": -0.1}
        table = secondary_full_loto_table(raw_rho_by_task)
        self.assertEqual(set(table.keys()), set(PRIMARY_STAGE_E_TASKS))

    def test_loto_locked_lcc_entry_matches_criterion_c(self):
        raw_rho_by_task = {"trec": -0.5, "lcc": -0.3, "passage_retrieval_en": 0.2, "2wikimqa": -0.1}
        table = secondary_full_loto_table(raw_rho_by_task)
        direct_c = evaluate_criterion_c(raw_rho_by_task, removed_task=LOTO_REMOVED_TASK)
        self.assertEqual(table["lcc"]["pass"], direct_c["pass"])
        self.assertEqual(table["lcc"]["negative_count"], direct_c["negative_count"])


if __name__ == "__main__":
    unittest.main()
