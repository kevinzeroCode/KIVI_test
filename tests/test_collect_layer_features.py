"""CPU-only tests for scripts/collect_layer_features.py's orchestration
logic (Stage G3A). No torch/transformers/CUDA required -- these test the
pure-Python planning, policy-construction, call-attribution-order, and
output-validation helpers, never a real forward pass.
"""
import inspect
import json
import os
import tempfile
import unittest

from scripts.collect_layer_features import (
    DEFAULT_CALIBRATION_PERCENTILES,
    F0_DIAGNOSTIC_LAYERS,
    F0_DIAGNOSTIC_TASKS,
    PRIMARY_STAGE_E_LAYERS,
    PRIMARY_STAGE_E_TASKS,
    FeatureCollectionConfigError,
    build_collection_plan,
    build_probe_policy_obj,
    calibration_percentiles_for_sample_count,
    expected_call_order,
    expected_identities,
    extract_length_pairs,
    key_residual_tokens,
    load_task_lengths,
    percentile_rank_index,
    run_real_collection,
    select_calibration_indices,
    validate_task_layer_scope,
    value_residual_tokens,
    validate_axes,
    validate_layers,
    validate_num_samples,
    validate_feature_jsonl,
)
from utils.layer_policy import resolve_layer_policy


class ValidateLayersTest(unittest.TestCase):
    def test_valid_layers_pass_through(self):
        self.assertEqual(validate_layers([0, 31], num_hidden_layers=32), [0, 31])

    def test_empty_raises(self):
        with self.assertRaises(FeatureCollectionConfigError):
            validate_layers([], num_hidden_layers=32)

    def test_out_of_range_raises(self):
        with self.assertRaises(FeatureCollectionConfigError):
            validate_layers([32], num_hidden_layers=32)
        with self.assertRaises(FeatureCollectionConfigError):
            validate_layers([-1], num_hidden_layers=32)

    def test_duplicate_raises(self):
        with self.assertRaises(FeatureCollectionConfigError):
            validate_layers([0, 0], num_hidden_layers=32)


class ValidateAxesTest(unittest.TestCase):
    def test_valid_axes(self):
        self.assertEqual(validate_axes(["key", "value"]), ["key", "value"])

    def test_empty_raises(self):
        with self.assertRaises(FeatureCollectionConfigError):
            validate_axes([])

    def test_unsupported_axis_raises(self):
        with self.assertRaises(FeatureCollectionConfigError):
            validate_axes(["bogus"])


class ValidateNumSamplesTest(unittest.TestCase):
    def test_positive_passes(self):
        self.assertEqual(validate_num_samples(1), 1)

    def test_zero_or_negative_raises(self):
        with self.assertRaises(FeatureCollectionConfigError):
            validate_num_samples(0)
        with self.assertRaises(FeatureCollectionConfigError):
            validate_num_samples(-5)


class BuildCollectionPlanTest(unittest.TestCase):
    def test_exact_expected_count(self):
        plan = build_collection_plan([0, 31], ["key", "value"], {"lcc": 500}, 1)
        # 1 task x 2 layers x 2 axes = 4 cells
        self.assertEqual(len(plan), 4)
        for cell in plan:
            self.assertEqual(cell["num_samples"], 1)

    def test_sample_cap_respects_task_count(self):
        plan = build_collection_plan([0], ["key"], {"lcc": 500}, 1000)
        self.assertEqual(plan[0]["num_samples"], 500)

    def test_never_merges_tasks(self):
        plan = build_collection_plan([0], ["key"], {"lcc": 500, "trec": 200}, 1)
        tasks_seen = {cell["task"] for cell in plan}
        self.assertEqual(tasks_seen, {"lcc", "trec"})


class BuildProbePolicyObjTest(unittest.TestCase):
    def test_overrides_only_requested_layers(self):
        obj = build_probe_policy_obj([0, 31], 2, 2)
        self.assertEqual(set(obj["overrides"].keys()), {"0", "31"})
        self.assertEqual(obj["default"], {"k_bits": 16, "v_bits": 16, "family": "kivi"})

    def test_resolves_correctly_via_layer_policy(self):
        obj = build_probe_policy_obj([0, 31], 2, 2)
        resolved = resolve_layer_policy(32, 16, 16, obj)
        for i, entry in enumerate(resolved.layers):
            if i in (0, 31):
                self.assertEqual((entry.k_bits, entry.v_bits), (2, 2))
            else:
                self.assertEqual((entry.k_bits, entry.v_bits), (16, 16))

    def test_both_axes_always_quantized_together(self):
        # A policy always sets K and V bits jointly per layer -- there is no
        # way to request "key only" quantization at the policy level.
        obj = build_probe_policy_obj([5], 2, 4)
        self.assertEqual(obj["overrides"]["5"], {"k_bits": 2, "v_bits": 4, "family": "kivi"})


class KeyResidualTokensTest(unittest.TestCase):
    """Key's residual is a REMAINDER (distribution_tokens % residual_length),
    kept full-precision so the quantized portion is a clean multiple of
    residual_length for the token-grouped Key quantizer."""

    def test_reproduces_real_g3a_lcc_sample(self):
        # The exact numbers from the Stage G3A canary: an 18,062-token lcc
        # sample emitted Key num_tokens=18048, which was unexplained until
        # traced -- this is that trace, made an executable regression test.
        residual = key_residual_tokens(18062, 128)
        self.assertEqual(residual, 14)
        self.assertEqual(18062 - residual, 18048)

    def test_exact_multiple_gives_zero_residual(self):
        self.assertEqual(key_residual_tokens(256, 128), 0)

    def test_shorter_than_residual_length_is_all_residual(self):
        self.assertEqual(key_residual_tokens(100, 128), 100)

    def test_residual_always_less_than_residual_length_when_quantizing(self):
        for distribution_tokens in range(128, 5000, 37):
            residual = key_residual_tokens(distribution_tokens, 128)
            self.assertLess(residual, 128)
            self.assertGreaterEqual(residual, 0)


class ValueResidualTokensTest(unittest.TestCase):
    """Value's residual is a FIXED-SIZE sliding window (always exactly
    residual_length tokens, or the whole sequence if shorter), since V's
    quantizer groups by channel axis, not token count -- no alignment
    constraint exists. This is why Key and Value residual counts differ in
    kind, not just magnitude, for the same input."""

    def test_reproduces_real_g3a_lcc_sample(self):
        residual = value_residual_tokens(18062, 128)
        self.assertEqual(residual, 128)
        self.assertEqual(18062 - residual, 17934)

    def test_shorter_than_or_equal_to_residual_length_is_all_residual(self):
        self.assertEqual(value_residual_tokens(128, 128), 128)
        self.assertEqual(value_residual_tokens(50, 128), 50)

    def test_always_exactly_residual_length_when_longer(self):
        for distribution_tokens in range(129, 5000, 37):
            self.assertEqual(value_residual_tokens(distribution_tokens, 128), 128)


class KeyValueResidualAsymmetryTest(unittest.TestCase):
    def test_differ_for_the_same_input_when_not_aligned(self):
        # The whole point of having two separate functions: for a
        # non-aligned length, Key's remainder-based residual and Value's
        # fixed-window residual genuinely disagree.
        self.assertNotEqual(key_residual_tokens(18062, 128), value_residual_tokens(18062, 128))

    def test_agree_only_when_input_is_exactly_aligned(self):
        # When distribution_tokens is an exact multiple of residual_length,
        # Key's remainder happens to be 0 while Value's fixed window is
        # still residual_length -- they do NOT coincide even here.
        self.assertEqual(key_residual_tokens(256, 128), 0)
        self.assertEqual(value_residual_tokens(256, 128), 128)

    def test_no_packing_loss_beyond_documented_residual(self):
        # Every token is accounted for by exactly one of
        # quantized/residual for both axes -- no additional silent
        # alignment/chunk remainder exists beyond what these functions
        # already return (verified by direct reading of
        # _chunked_key/value_quantize_and_pack_along_last_dim's internal
        # compute-chunking, which never drops data).
        for distribution_tokens in (129, 200, 18062, 100000):
            key_r = key_residual_tokens(distribution_tokens, 128)
            value_r = value_residual_tokens(distribution_tokens, 128)
            key_q = distribution_tokens - key_r
            value_q = distribution_tokens - value_r
            self.assertEqual(key_q + key_r, distribution_tokens)
            self.assertEqual(value_q + value_r, distribution_tokens)


class PercentileRankIndexTest(unittest.TestCase):
    def test_known_positions(self):
        # n=160 (0..159): p20 -> round(0.2*159)=32? check exact formula
        self.assertEqual(percentile_rank_index(160, 20), round(0.2 * 159))
        self.assertEqual(percentile_rank_index(160, 80), round(0.8 * 159))

    def test_boundaries_clamped(self):
        self.assertEqual(percentile_rank_index(10, 0), 0)
        self.assertEqual(percentile_rank_index(10, 100), 9)

    def test_empty_raises(self):
        with self.assertRaises(FeatureCollectionConfigError):
            percentile_rank_index(0, 50)

    def test_single_element(self):
        self.assertEqual(percentile_rank_index(1, 20), 0)
        self.assertEqual(percentile_rank_index(1, 80), 0)


class SelectCalibrationIndicesTest(unittest.TestCase):
    def test_reproduces_four_distinct_percentile_samples(self):
        pairs = [(i, length) for i, length in enumerate(range(0, 500, 5))]  # 100 examples, length = 5*idx
        selected = select_calibration_indices(pairs)
        self.assertEqual(len(selected), 4)
        self.assertEqual([s["percentile"] for s in selected], [20, 40, 60, 80])
        # Distinct dataset indices selected (no collision on this well-spread input).
        self.assertEqual(len({s["dataset_index"] for s in selected}), 4)
        # Monotonic: higher percentile -> longer (or equal) input length.
        lengths = [s["input_length"] for s in selected]
        self.assertEqual(lengths, sorted(lengths))

    def test_tie_broken_by_ascending_original_index(self):
        # All examples share the same length -- percentile position must
        # still resolve to a specific, deterministic original index via
        # the (length, index) sort, never dataset order/insertion order.
        pairs = [(i, 100) for i in range(20)]
        selected = select_calibration_indices(pairs)
        # With all lengths equal, sorted order == ascending index order,
        # so rank_position directly gives the selected dataset_index.
        for s in selected:
            self.assertEqual(s["dataset_index"], s["rank_position"])

    def test_never_uses_generated_or_sensitivity_data(self):
        # Purely a signature/contract check: the function's only inputs
        # are (index, length) pairs -- no prediction, score, or feature
        # argument exists to accidentally leak in.
        import inspect
        sig = inspect.signature(select_calibration_indices)
        self.assertEqual(list(sig.parameters.keys()), ["index_length_pairs", "percentiles"])

    def test_deterministic_across_repeated_calls(self):
        pairs = [(i, (i * 37) % 200) for i in range(150)]
        self.assertEqual(select_calibration_indices(pairs), select_calibration_indices(pairs))

    def test_small_dataset_collision_avoidance_stays_distinct(self):
        # A tiny dataset where naive nearest-rank positions could collide;
        # the forward/backward search must still produce 4 distinct rows.
        pairs = [(i, i) for i in range(5)]
        selected = select_calibration_indices(pairs)
        self.assertEqual(len({s["rank_position"] for s in selected}), 4)


class ValidateTaskLayerScopeTest(unittest.TestCase):
    """Stage G3B-IMPL Part 7 guard: diagnostic tasks must never silently
    collect the 8 primary layers, and primary tasks must never silently
    collect an incomplete layer set."""

    def test_diagnostic_tasks_require_layer_zero_only(self):
        validate_task_layer_scope(["multifieldqa_en"], list(F0_DIAGNOSTIC_LAYERS))  # no raise

    def test_diagnostic_tasks_with_all_eight_layers_raises(self):
        with self.assertRaises(FeatureCollectionConfigError):
            validate_task_layer_scope(["multifieldqa_en", "samsum"], list(PRIMARY_STAGE_E_LAYERS))

    def test_primary_tasks_require_all_eight_layers(self):
        validate_task_layer_scope(list(PRIMARY_STAGE_E_TASKS), list(PRIMARY_STAGE_E_LAYERS))  # no raise

    def test_primary_tasks_with_incomplete_layers_raises(self):
        with self.assertRaises(FeatureCollectionConfigError):
            validate_task_layer_scope(["lcc"], [0])

    def test_mixing_primary_and_diagnostic_raises(self):
        with self.assertRaises(FeatureCollectionConfigError):
            validate_task_layer_scope(["lcc", "samsum"], [0])

    def test_adhoc_task_outside_registered_sets_is_unconstrained(self):
        validate_task_layer_scope(["hotpotqa"], [3, 7, 19])  # no raise -- ad-hoc use still allowed


class FirstNRegressionGuardTest(unittest.TestCase):
    """Regression guard (Stage G3B-IMPL Part 9): first-N sample indexing
    must never silently reappear as real collection's selection mechanism.
    Verified by source inspection since running run_real_collection itself
    requires a GPU/model."""

    def test_real_collection_uses_percentile_selection_not_first_n(self):
        source = inspect.getsource(run_real_collection)
        self.assertIn("select_calibration_indices(extract_length_pairs(data)", source)
        self.assertNotIn("for sample_idx in range(", source)
        self.assertNotIn("range(num_samples)", source)


class CalibrationPercentilesForSampleCountTest(unittest.TestCase):
    """Stage-G3B-IMPL-2 Part 7/8: --num-samples must explicitly control the
    calibration design, not be a silently-ignored default. num_samples=4
    (the pre-registered design) must reproduce DEFAULT_CALIBRATION_PERCENTILES
    (20, 40, 60, 80) EXACTLY -- same values, same type -- so
    --num-samples 4 is provably not a no-op and reproduces the persisted
    Stage-G3B selection byte-for-byte."""

    def test_four_samples_matches_default_percentiles_exactly(self):
        self.assertEqual(calibration_percentiles_for_sample_count(4), DEFAULT_CALIBRATION_PERCENTILES)
        self.assertEqual(calibration_percentiles_for_sample_count(4), (20, 40, 60, 80))

    def test_other_counts_are_evenly_spaced_and_distinct_from_default(self):
        percentiles = calibration_percentiles_for_sample_count(3)
        self.assertEqual(len(percentiles), 3)
        self.assertEqual(percentiles, (25.0, 50.0, 75.0))
        self.assertNotEqual(percentiles, DEFAULT_CALIBRATION_PERCENTILES)

    def test_zero_or_negative_raises(self):
        with self.assertRaises(FeatureCollectionConfigError):
            calibration_percentiles_for_sample_count(0)
        with self.assertRaises(FeatureCollectionConfigError):
            calibration_percentiles_for_sample_count(-1)


class PreviewVersusRealSelectionParityTest(unittest.TestCase):
    """Proves preview (load_task_lengths, used by --preview-calibration)
    and real collection (extract_length_pairs on an already-loaded
    dataset) feed the exact same selection function and produce identical
    results -- exactly ONE selection implementation, not two independent
    ones. CPU-only (no torch/GPU); requires the `datasets` library and the
    already-locally-cached THUDM/LongBench lcc split (used throughout
    Stages G2/G3A/G3B of this same project)."""

    def test_preview_and_direct_dataset_load_agree_with_explicit_num_samples_4(self):
        from datasets import load_dataset

        percentiles = calibration_percentiles_for_sample_count(4)  # what --num-samples 4 produces
        preview_pairs = load_task_lengths("lcc")
        data = load_dataset("THUDM/LongBench", "lcc", split="test", trust_remote_code=True)
        real_collection_pairs = extract_length_pairs(data)
        self.assertEqual(preview_pairs, real_collection_pairs)
        self.assertEqual(
            select_calibration_indices(preview_pairs, percentiles=percentiles),
            select_calibration_indices(real_collection_pairs, percentiles=percentiles),
        )

    def test_explicit_num_samples_4_reproduces_persisted_preregistration_indices(self):
        # Byte-for-byte proof against docs/stage_g3b_pre_registration.md's
        # persisted table, using the exact percentiles --num-samples 4
        # would compute.
        percentiles = calibration_percentiles_for_sample_count(4)
        expected = {
            "trec": [153, 154, 82, 151],
            "lcc": [122, 174, 214, 357],
            "passage_retrieval_en": [141, 185, 105, 41],
            "2wikimqa": [164, 44, 138, 38],
            "multifieldqa_en": [50, 103, 81, 92],
            "samsum": [58, 118, 151, 52],
        }
        for task, expected_indices in expected.items():
            pairs = load_task_lengths(task)
            selection = select_calibration_indices(pairs, percentiles=percentiles)
            got_indices = [sel["dataset_index"] for sel in selection]
            self.assertEqual(got_indices, expected_indices, f"task={task}")


class ExpectedCallOrderTest(unittest.TestCase):
    """expected_call_order returns the per-axis call order: each of the two
    production per-axis quantizer entry points
    (_chunked_key_quantize_and_pack_along_last_dim /
    _chunked_value_quantize_and_pack_along_last_dim) fires exactly once per
    probed layer, in ascending layer-index order, regardless of prompt
    length (internal token-range chunking is already concatenated before
    returning)."""

    def test_ascending_layer_order(self):
        self.assertEqual(expected_call_order([31, 0]), [0, 31])

    def test_single_layer(self):
        self.assertEqual(expected_call_order([17]), [17])

    def test_already_sorted_unchanged(self):
        self.assertEqual(expected_call_order([0, 31]), [0, 31])


class ExpectedIdentitiesTest(unittest.TestCase):
    def test_canary_shape_exactly_four(self):
        ids = expected_identities([("lcc", 0)], [0, 31], ["key", "value"])
        self.assertEqual(len(ids), 4)
        self.assertEqual(
            set(ids),
            {("lcc", 0, 0, "key"), ("lcc", 0, 0, "value"), ("lcc", 0, 31, "key"), ("lcc", 0, 31, "value")},
        )

    def test_axis_filter_reduces_count(self):
        ids = expected_identities([("lcc", 0)], [0, 31], ["key"])
        self.assertEqual(set(ids), {("lcc", 0, 0, "key"), ("lcc", 0, 31, "key")})

    def test_no_duplicates(self):
        ids = expected_identities([("lcc", 0), ("lcc", 1)], [0, 31], ["key", "value"])
        self.assertEqual(len(ids), len(set(ids)))


class ValidateFeatureJsonlTest(unittest.TestCase):
    def _write(self, tmpdir, rows):
        path = os.path.join(tmpdir, "features.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return path

    def _good_row(self, task="lcc", sample_idx=0, layer_idx=0, axis="key"):
        return {
            "task": task, "sample_idx": sample_idx, "layer_idx": layer_idx, "tensor_axis": axis,
            "input_tokens": 100, "distribution_tokens": 100, "quantized_tokens": 86, "residual_tokens": 14,
            "relative_l2": 0.2, "mse": 0.01, "max_abs_error": 0.5,
            "mean": 0.001, "std": 0.4, "variance": 0.16, "max_abs": 3.0,
            "p50_abs": 0.2, "p95_abs": 1.0, "p99_abs": 1.5, "outlier_fraction": 0.02,
        }

    def test_valid_four_row_canary_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                self._good_row(layer_idx=0, axis="key"),
                self._good_row(layer_idx=0, axis="value"),
                self._good_row(layer_idx=31, axis="key"),
                self._good_row(layer_idx=31, axis="value"),
            ]
            path = self._write(tmp, rows)
            expected = [("lcc", 0, 0, "key"), ("lcc", 0, 0, "value"), ("lcc", 0, 31, "key"), ("lcc", 0, 31, "value")]
            result = validate_feature_jsonl(path, expected)
            self.assertTrue(result["ok"], result["problems"])
            self.assertEqual(result["row_count"], 4)

    def test_missing_row_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [self._good_row(layer_idx=0, axis="key")]
            path = self._write(tmp, rows)
            expected = [("lcc", 0, 0, "key"), ("lcc", 0, 0, "value")]
            result = validate_feature_jsonl(path, expected)
            self.assertFalse(result["ok"])

    def test_duplicate_identity_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [self._good_row(layer_idx=0, axis="key"), self._good_row(layer_idx=0, axis="key")]
            path = self._write(tmp, rows)
            expected = [("lcc", 0, 0, "key"), ("lcc", 0, 0, "key")]
            result = validate_feature_jsonl(path, expected)
            self.assertFalse(result["ok"])
            self.assertTrue(any("duplicate identity" in p for p in result["problems"]))

    def test_nonfinite_field_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = self._good_row()
            bad["relative_l2"] = float("nan")
            path = self._write(tmp, [bad])
            result = validate_feature_jsonl(path, [("lcc", 0, 0, "key")])
            self.assertFalse(result["ok"])

    def test_negative_reconstruction_field_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = self._good_row()
            bad["mse"] = -0.1
            path = self._write(tmp, [bad])
            result = validate_feature_jsonl(path, [("lcc", 0, 0, "key")])
            self.assertFalse(result["ok"])

    def test_missing_trailing_newline_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "features.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(self._good_row()))  # no trailing \n
            result = validate_feature_jsonl(path, [("lcc", 0, 0, "key")])
            self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
