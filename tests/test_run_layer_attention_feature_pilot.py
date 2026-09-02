"""CPU-only tests for scripts/run_layer_attention_feature_pilot.py (Stage
H3). No torch/GPU/model -- the real GPU collection entry point is not
implemented this round (main() refuses to run without --dry-run), so these
tests cover exactly the pure trajectory planning / record schema / resume
logic that exists.
"""
import json
import os
import tempfile
import unittest

from scripts.run_layer_attention_feature_pilot import (
    AXIS_POLICY_BITS,
    DEFAULT_REQUESTED_STEPS,
    DIAGNOSTIC_LAYERS,
    DIAGNOSTIC_TASKS,
    LOCKED_SAMPLE_SELECTION,
    PRIMARY_LAYERS,
    PRIMARY_TASKS,
    PilotConfigError,
    RecordValidationError,
    ResumeError,
    build_trajectory_plan,
    build_trajectory_record,
    group_trajectories_by_policy,
    load_completed_records,
    plan_remaining_trajectories,
    trajectory_identity,
    validate_policy_bits,
    validate_record_shape,
    validate_scope_guard,
)


class LockedSampleSelectionTest(unittest.TestCase):
    def test_exact_stage_g_prompt_indices(self):
        self.assertEqual(LOCKED_SAMPLE_SELECTION["trec"], (153, 154, 82, 151))
        self.assertEqual(LOCKED_SAMPLE_SELECTION["lcc"], (122, 174, 214, 357))
        self.assertEqual(LOCKED_SAMPLE_SELECTION["passage_retrieval_en"], (141, 185, 105, 41))
        self.assertEqual(LOCKED_SAMPLE_SELECTION["2wikimqa"], (164, 44, 138, 38))
        self.assertEqual(LOCKED_SAMPLE_SELECTION["multifieldqa_en"], (50, 103, 81, 92))
        self.assertEqual(LOCKED_SAMPLE_SELECTION["samsum"], (58, 118, 151, 52))

    def test_every_task_has_exactly_four_samples(self):
        for task, samples in LOCKED_SAMPLE_SELECTION.items():
            self.assertEqual(len(samples), 4, task)

    def test_cross_check_against_frozen_stage_g_provenance_if_present(self):
        # Best-effort: if Stage G's real frozen sample_selection.json files
        # are present on disk, cross-check the hardcoded LOCKED table
        # against them exactly. Skips gracefully if not present (this test
        # must not require the real Stage-G run to exist).
        import glob

        primary_files = glob.glob("outputs/layer_feature_pilot/primary/*/sample_selection.json")
        diagnostic_files = glob.glob("outputs/layer_feature_pilot/diagnostic/*/sample_selection.json")
        if not primary_files or not diagnostic_files:
            self.skipTest("frozen Stage-G sample_selection.json not present in this environment")
        seen = {}
        for path in primary_files + diagnostic_files:
            with open(path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            for entry in entries:
                seen.setdefault(entry["task"], []).append(entry["dataset_index"])
        for task, expected in LOCKED_SAMPLE_SELECTION.items():
            self.assertEqual(tuple(seen[task]), expected, task)


class BuildTrajectoryPlanTest(unittest.TestCase):
    def test_exact_totals(self):
        plan = build_trajectory_plan()
        primary = [t for t in plan if t["scope"] == "primary"]
        diagnostic = [t for t in plan if t["scope"] == "diagnostic"]
        self.assertEqual(len(primary), 256)
        self.assertEqual(len(diagnostic), 16)
        self.assertEqual(len(plan), 272)

    def test_no_duplicate_identities(self):
        plan = build_trajectory_plan()
        identities = [trajectory_identity(t) for t in plan]
        self.assertEqual(len(identities), len(set(identities)))

    def test_no_missing_identities(self):
        plan = build_trajectory_plan()
        identities = set(trajectory_identity(t) for t in plan)
        expected = set()
        for task in PRIMARY_TASKS:
            for idx in LOCKED_SAMPLE_SELECTION[task]:
                for layer in PRIMARY_LAYERS:
                    for axis in AXIS_POLICY_BITS:
                        expected.add((task, idx, layer, axis))
        for task in DIAGNOSTIC_TASKS:
            for idx in LOCKED_SAMPLE_SELECTION[task]:
                for layer in DIAGNOSTIC_LAYERS:
                    for axis in AXIS_POLICY_BITS:
                        expected.add((task, idx, layer, axis))
        self.assertEqual(identities, expected)

    def test_key_axis_uses_k2v16(self):
        plan = build_trajectory_plan()
        key_rows = [t for t in plan if t["tensor_axis"] == "key"]
        self.assertTrue(all((t["k_bits"], t["v_bits"]) == (2, 16) for t in key_rows))
        self.assertTrue(all(t["policy"] == "K2/V16" for t in key_rows))

    def test_value_axis_uses_k16v2(self):
        plan = build_trajectory_plan()
        value_rows = [t for t in plan if t["tensor_axis"] == "value"]
        self.assertTrue(all((t["k_bits"], t["v_bits"]) == (16, 2) for t in value_rows))
        self.assertTrue(all(t["policy"] == "K16/V2" for t in value_rows))

    def test_no_joint_k2v2_scientific_policy_ever_appears(self):
        plan = build_trajectory_plan()
        self.assertTrue(all((t["k_bits"], t["v_bits"]) != (2, 2) for t in plan))

    def test_diagnostic_only_ever_layer_zero(self):
        plan = build_trajectory_plan()
        diagnostic_layers = {t["layer_idx"] for t in plan if t["scope"] == "diagnostic"}
        self.assertEqual(diagnostic_layers, {0})

    def test_primary_uses_exactly_eight_layers(self):
        plan = build_trajectory_plan()
        primary_layers = {t["layer_idx"] for t in plan if t["scope"] == "primary"}
        self.assertEqual(primary_layers, set(PRIMARY_LAYERS))

    def test_deterministic_ordering(self):
        self.assertEqual(build_trajectory_plan(), build_trajectory_plan())


class GroupTrajectoriesByPolicyTest(unittest.TestCase):
    def test_sixteen_primary_groups(self):
        groups = group_trajectories_by_policy(build_trajectory_plan())
        primary_groups = [k for k in groups if k[0] == "primary"]
        self.assertEqual(len(primary_groups), 16)

    def test_two_diagnostic_groups(self):
        groups = group_trajectories_by_policy(build_trajectory_plan())
        diagnostic_groups = [k for k in groups if k[0] == "diagnostic"]
        self.assertEqual(len(diagnostic_groups), 2)

    def test_eighteen_total_model_loads(self):
        groups = group_trajectories_by_policy(build_trajectory_plan())
        self.assertEqual(len(groups), 18)

    def test_each_primary_group_has_sixteen_trajectories(self):
        groups = group_trajectories_by_policy(build_trajectory_plan())
        for key, members in groups.items():
            if key[0] == "primary":
                self.assertEqual(len(members), 16, key)

    def test_diagnostic_never_merged_with_primary_layer_zero(self):
        groups = group_trajectories_by_policy(build_trajectory_plan())
        self.assertIn(("primary", 0, "key"), groups)
        self.assertIn(("diagnostic", 0, "key"), groups)
        self.assertNotEqual(groups[("primary", 0, "key")], groups[("diagnostic", 0, "key")])


class ValidateScopeGuardTest(unittest.TestCase):
    def test_diagnostic_layer_zero_passes(self):
        validate_scope_guard("diagnostic", [0])  # no raise

    def test_diagnostic_all_eight_layers_fails_before_model_load(self):
        with self.assertRaises(PilotConfigError):
            validate_scope_guard("diagnostic", PRIMARY_LAYERS)

    def test_primary_eight_layers_passes(self):
        validate_scope_guard("primary", PRIMARY_LAYERS)  # no raise

    def test_primary_incomplete_layers_fails(self):
        with self.assertRaises(PilotConfigError):
            validate_scope_guard("primary", [0])


class ValidatePolicyBitsTest(unittest.TestCase):
    def test_key_k2v16_passes(self):
        validate_policy_bits("key", 2, 16)  # no raise

    def test_value_k16v2_passes(self):
        validate_policy_bits("value", 16, 2)  # no raise

    def test_joint_k2v2_rejected(self):
        with self.assertRaises(PilotConfigError):
            validate_policy_bits("key", 2, 2)
        with self.assertRaises(PilotConfigError):
            validate_policy_bits("value", 2, 2)

    def test_unknown_axis_rejected(self):
        with self.assertRaises(PilotConfigError):
            validate_policy_bits("bogus", 2, 16)


class BuildTrajectoryRecordTest(unittest.TestCase):
    def _trajectory(self, axis="key"):
        k_bits, v_bits = AXIS_POLICY_BITS[axis]
        return {"scope": "primary", "task": "lcc", "dataset_index": 122, "layer_idx": 0, "tensor_axis": axis, "policy": f"K{k_bits}/V{v_bits}", "k_bits": k_bits, "v_bits": v_bits}

    def test_full_steps_available(self):
        rec = build_trajectory_record(
            self._trajectory(), "lmsys/longchat-7b-v1.5-32k", "abc123", 42, 3489, 20,
            {1: 0.1, 2: 0.2, 4: 0.3, 8: 0.4, 16: 0.5},
        )
        validate_record_shape(rec)
        self.assertEqual(rec["sampled_decode_steps"], [1, 2, 4, 8, 16])
        self.assertAlmostEqual(rec["sample_attention_distortion"], 0.3, places=10)
        self.assertFalse(rec["no_decode_exposure"])

    def test_partial_early_eos(self):
        rec = build_trajectory_record(
            self._trajectory(), "m", "c", 42, 100, 5, {1: 0.2, 2: 0.6},
        )
        validate_record_shape(rec)
        self.assertEqual(rec["sampled_decode_steps"], [1, 2])
        self.assertAlmostEqual(rec["sample_attention_distortion"], 0.4, places=10)
        self.assertFalse(rec["no_decode_exposure"])

    def test_zero_decode_exposure(self):
        rec = build_trajectory_record(self._trajectory(), "m", "c", 42, 50, 1, {})
        validate_record_shape(rec)
        self.assertTrue(rec["no_decode_exposure"])
        self.assertEqual(rec["sample_attention_distortion"], 0.0)
        self.assertEqual(rec["valid_step_count"], 0)

    def test_unavailable_step_never_imputed(self):
        rec = build_trajectory_record(self._trajectory(), "m", "c", 42, 100, 9, {1: 1.0, 2: 1.0, 8: 1.0, 16: 1.0})
        self.assertEqual(rec["sampled_decode_steps"], [1, 2, 8, 16])
        self.assertAlmostEqual(rec["sample_attention_distortion"], 1.0, places=10)

    def test_per_step_mean_reproduces_exactly(self):
        rec = build_trajectory_record(self._trajectory(), "m", "c", 42, 100, 5, {1: 0.11, 4: 0.37})
        recomputed = sum(s["distortion"] for s in rec["per_step_distortions"]) / len(rec["per_step_distortions"])
        self.assertAlmostEqual(recomputed, rec["sample_attention_distortion"], places=12)

    def test_requested_steps_locked(self):
        rec = build_trajectory_record(self._trajectory(), "m", "c", 42, 100, 5, {1: 0.1})
        self.assertEqual(rec["requested_decode_steps"], list(DEFAULT_REQUESTED_STEPS))
        self.assertEqual(list(DEFAULT_REQUESTED_STEPS), [1, 2, 4, 8, 16])

    def test_value_axis_record_policy(self):
        rec = build_trajectory_record(self._trajectory("value"), "m", "c", 42, 100, 5, {1: 0.1})
        self.assertEqual((rec["k_bits"], rec["v_bits"]), (16, 2))


class ValidateRecordShapeTest(unittest.TestCase):
    def _good_record(self):
        return build_trajectory_record(
            {"scope": "primary", "task": "lcc", "dataset_index": 122, "layer_idx": 0, "tensor_axis": "key", "policy": "K2/V16", "k_bits": 2, "v_bits": 16},
            "m", "c", 42, 100, 5, {1: 0.1, 2: 0.2},
        )

    def test_valid_record_passes(self):
        self.assertTrue(validate_record_shape(self._good_record()))

    def test_missing_field_rejected(self):
        rec = self._good_record()
        del rec["policy"]
        with self.assertRaises(RecordValidationError):
            validate_record_shape(rec)

    def test_nonfinite_distortion_rejected(self):
        rec = self._good_record()
        rec["per_step_distortions"][0]["distortion"] = float("nan")
        with self.assertRaises(RecordValidationError):
            validate_record_shape(rec)

    def test_negative_distortion_rejected(self):
        rec = self._good_record()
        rec["per_step_distortions"][0]["distortion"] = -0.1
        with self.assertRaises(RecordValidationError):
            validate_record_shape(rec)

    def test_inconsistent_mean_rejected(self):
        rec = self._good_record()
        rec["sample_attention_distortion"] = 999.0
        with self.assertRaises(RecordValidationError):
            validate_record_shape(rec)

    def test_policy_axis_mismatch_rejected(self):
        rec = self._good_record()
        rec["k_bits"] = 16
        rec["v_bits"] = 16
        with self.assertRaises(RecordValidationError):
            validate_record_shape(rec)


class LoadCompletedRecordsAndResumeTest(unittest.TestCase):
    def _write_jsonl(self, path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def _record(self, task="lcc", idx=122, layer=0, axis="key", git_commit="c1"):
        traj = {"scope": "primary", "task": task, "dataset_index": idx, "layer_idx": layer, "tensor_axis": axis, "policy": "K2/V16" if axis == "key" else "K16/V2", "k_bits": 2 if axis == "key" else 16, "v_bits": 16 if axis == "key" else 2}
        return build_trajectory_record(traj, "m", git_commit, 42, 100, 5, {1: 0.1, 2: 0.2})

    def test_no_file_returns_empty(self):
        completed = load_completed_records("/nonexistent/path/features.jsonl")
        self.assertEqual(completed, {})

    def test_valid_completed_records_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "features.jsonl")
            self._write_jsonl(path, [self._record(idx=122), self._record(idx=174)])
            completed = load_completed_records(path)
            self.assertEqual(len(completed), 2)
            self.assertIn(("lcc", 122, 0, "key"), completed)

    def test_malformed_json_line_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "features.jsonl")
            with open(path, "w") as f:
                f.write("{not valid json\n")
            with self.assertRaises(ResumeError):
                load_completed_records(path)

    def test_duplicate_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "features.jsonl")
            self._write_jsonl(path, [self._record(idx=122), self._record(idx=122)])
            with self.assertRaises(ResumeError):
                load_completed_records(path)

    def test_git_commit_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "features.jsonl")
            self._write_jsonl(path, [self._record(git_commit="old_commit")])
            with self.assertRaises(ResumeError):
                load_completed_records(path, expected_git_commit="new_commit")

    def test_matching_git_commit_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "features.jsonl")
            self._write_jsonl(path, [self._record(git_commit="c1")])
            completed = load_completed_records(path, expected_git_commit="c1")
            self.assertEqual(len(completed), 1)

    def test_plan_remaining_excludes_completed(self):
        plan = build_trajectory_plan()[:5]
        completed_ids = {trajectory_identity(plan[0]), trajectory_identity(plan[2])}
        remaining = plan_remaining_trajectories(plan, completed_ids)
        self.assertEqual(len(remaining), 3)
        self.assertNotIn(trajectory_identity(plan[0]), [trajectory_identity(t) for t in remaining])

    def test_plan_remaining_all_when_nothing_completed(self):
        plan = build_trajectory_plan()[:10]
        remaining = plan_remaining_trajectories(plan, set())
        self.assertEqual(len(remaining), 10)


if __name__ == "__main__":
    unittest.main()
