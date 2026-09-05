"""CPU-only tests for scripts/run_layer_attention_feature_pilot.py (Stage
H3A: pure trajectory planning / record schema / resume logic; Stage H3B:
the real-collection ORCHESTRATOR -- exercised here via dependency
injection / fake backends only, never CUDA/torch/a real model. The
GPU-touching default backends (default_load_model_fn,
default_run_trajectory_fn) are H2-reused and are NOT invoked by anything
in this file.
"""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import scripts.run_layer_attention_feature_pilot as _mod
from scripts.run_layer_attention_feature_pilot import (
    AXIS_POLICY_BITS,
    DEFAULT_GROUP_SIZE,
    DEFAULT_MAX_NEW_TOKENS_HORIZON,
    DEFAULT_MODEL_NAME,
    DEFAULT_REQUESTED_STEPS,
    DEFAULT_RESIDUAL_LENGTH,
    DEFAULT_SEED,
    DIAGNOSTIC_LAYERS,
    DIAGNOSTIC_TASKS,
    LOCKED_SAMPLE_SELECTION,
    PRIMARY_LAYERS,
    PRIMARY_TASKS,
    PilotConfigError,
    PilotLockError,
    RecordValidationError,
    ResumeError,
    build_trajectory_plan,
    build_trajectory_record,
    filter_plan,
    group_trajectories_by_policy,
    load_completed_records,
    parse_args,
    plan_remaining_trajectories,
    run_real_collection,
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


class FilterPlanTest(unittest.TestCase):
    def test_scope_filter(self):
        primary = filter_plan(build_trajectory_plan(), scope="primary")
        self.assertEqual(len(primary), 256)
        self.assertTrue(all(t["scope"] == "primary" for t in primary))

    def test_layer_filter_spans_both_scopes(self):
        layer0 = filter_plan(build_trajectory_plan(), layer=0)
        self.assertEqual(len(layer0), 48)  # primary layer0: 4 tasks x 4 samples x 2 axes = 32; diagnostic layer0: 2x4x2 = 16
        self.assertTrue(all(t["layer_idx"] == 0 for t in layer0))

    def test_axis_filter(self):
        key_only = filter_plan(build_trajectory_plan(), axis="key")
        self.assertEqual(len(key_only), 136)
        self.assertTrue(all(t["tensor_axis"] == "key" for t in key_only))

    def test_task_filter(self):
        trec_only = filter_plan(build_trajectory_plan(), task="trec")
        self.assertEqual(len(trec_only), 64)

    def test_dataset_index_filter(self):
        one_prompt = filter_plan(build_trajectory_plan(), task="lcc", dataset_index=122)
        self.assertEqual(len(one_prompt), 16)  # 8 layers x 2 axes
        self.assertTrue(all(t["dataset_index"] == 122 for t in one_prompt))

    def test_max_trajectories_truncates_without_reordering(self):
        plan = build_trajectory_plan()
        first_five = filter_plan(plan, max_trajectories=5)
        self.assertEqual(first_five, plan[:5])

    def test_full_selector_combo_yields_exactly_one_trajectory(self):
        one = filter_plan(build_trajectory_plan(), scope="primary", layer=0, axis="key", task="trec", dataset_index=153)
        self.assertEqual(len(one), 1)
        self.assertEqual(trajectory_identity(one[0]), ("trec", 153, 0, "key"))

    def test_empty_result_when_no_match(self):
        none = filter_plan(build_trajectory_plan(), task="trec", dataset_index=999999)
        self.assertEqual(none, [])


class ParseArgsSelectorTest(unittest.TestCase):
    def test_defaults_are_none_and_flags_false(self):
        args = parse_args([])
        self.assertFalse(args.dry_run)
        self.assertFalse(args.run)
        for field in ("layer", "axis", "task", "dataset_index", "max_trajectories", "scope"):
            self.assertIsNone(getattr(args, field))

    def test_run_flag_and_all_selectors_parse(self):
        args = parse_args([
            "--run", "--scope", "primary", "--layer", "4", "--axis", "key",
            "--task", "trec", "--dataset-index", "153", "--max-trajectories", "1",
        ])
        self.assertTrue(args.run)
        self.assertFalse(args.dry_run)
        self.assertEqual(args.scope, "primary")
        self.assertEqual(args.layer, 4)
        self.assertEqual(args.axis, "key")
        self.assertEqual(args.task, "trec")
        self.assertEqual(args.dataset_index, 153)
        self.assertEqual(args.max_trajectories, 1)

    def test_invalid_axis_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(["--axis", "both"])


def _make_args(output_root, scope="diagnostic", run_label="testrun", **overrides):
    ns = SimpleNamespace(
        scope=scope, output_root=output_root, run_label=run_label,
        model_name_or_path=DEFAULT_MODEL_NAME, cache_dir="./cached_models",
        group_size=DEFAULT_GROUP_SIZE, residual_length=DEFAULT_RESIDUAL_LENGTH,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS_HORIZON, seed=DEFAULT_SEED,
        layer=None, axis=None, task=None, dataset_index=None, max_trajectories=None,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _ok_git_status_fn():
    return ""


def _ok_git_commit_fn():
    return "deadbeef"


def _ok_boot_id_fn():
    return "boot-1"


def _ok_conflict_check_fn():
    return []


def _ok_gpu_preflight_fn():
    return {"checked": True, "total_used_mib": 0, "processes": [], "warning": None}


class _FakeBackend:
    """Dependency-injected fake backend (Part 16): stands in for the real
    GPU-touching load_model_fn/release_model_fn/run_trajectory_fn so
    run_real_collection's ORCHESTRATION (fixed-policy grouping, fresh
    per-trajectory state, record writing, resume, manifest lifecycle,
    fail-closed failure handling) is fully CPU-testable without CUDA.
    """

    def __init__(self, fail_on_identity=None):
        self.load_calls = []
        self.release_calls = []
        self.trajectory_calls = []  # list of (identity, load_id)
        self.fail_on_identity = fail_on_identity

    def load_model_fn(self, model_name_or_path, cache_dir, layer_idx, k_bits, v_bits, seed, group_size, residual_length):
        handle = {"layer_idx": layer_idx, "k_bits": k_bits, "v_bits": v_bits, "load_id": len(self.load_calls)}
        self.load_calls.append(handle)
        return handle

    def release_model_fn(self, model_handle):
        self.release_calls.append(model_handle)

    def run_trajectory_fn(self, model_handle, trajectory, dataset_cache, max_new_tokens, group_size):
        identity = trajectory_identity(trajectory)
        self.trajectory_calls.append((identity, model_handle["load_id"]))
        if self.fail_on_identity is not None and identity == self.fail_on_identity:
            raise RuntimeError("synthetic failure")
        return {"prompt_input_tokens": 100, "actual_generated_token_count": 5, "step_distortions": {1: 0.1, 2: 0.2}}


def _run(args, backend, **fn_overrides):
    kwargs = dict(
        load_model_fn=backend.load_model_fn, release_model_fn=backend.release_model_fn,
        run_trajectory_fn=backend.run_trajectory_fn, git_status_fn=_ok_git_status_fn,
        git_commit_fn=_ok_git_commit_fn, boot_id_fn=_ok_boot_id_fn,
        conflict_check_fn=_ok_conflict_check_fn, gpu_preflight_fn=_ok_gpu_preflight_fn,
    )
    kwargs.update(fn_overrides)
    return run_real_collection(args, **kwargs)


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class RunRealCollectionFakeBackendTest(unittest.TestCase):
    def test_single_group_one_load_all_members_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _make_args(tmp, scope="diagnostic", layer=0, axis="key", task="multifieldqa_en")
            backend = _FakeBackend()
            rc = _run(args, backend)
            self.assertEqual(rc, 0)
            self.assertEqual(len(backend.load_calls), 1)
            self.assertEqual(len(backend.release_calls), 1)
            self.assertEqual(len(backend.trajectory_calls), 4)
            records = _read_jsonl(os.path.join(tmp, "testrun", "features.jsonl"))
            self.assertEqual(len(records), 4)
            manifest = _read_manifest(os.path.join(tmp, "testrun", "manifest.json"))
            self.assertEqual(manifest["state"], "COMPLETE")
            self.assertEqual(manifest["completed_count"], 4)
            self.assertEqual(manifest["planned_count"], 4)
            self.assertEqual(manifest["collection_head"], "deadbeef")

    def test_next_policy_group_requires_a_new_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _make_args(tmp, scope="diagnostic", layer=0, task="multifieldqa_en")  # both axes -> 2 groups
            backend = _FakeBackend()
            rc = _run(args, backend)
            self.assertEqual(rc, 0)
            self.assertEqual(len(backend.load_calls), 2)
            self.assertEqual(len(backend.release_calls), 2)
            self.assertEqual(len(backend.trajectory_calls), 8)
            # Every trajectory in a given group used that group's own load_id -- never a stale/reused handle.
            by_load_id = {}
            for identity, load_id in backend.trajectory_calls:
                by_load_id.setdefault(load_id, set()).add(identity[3])  # axis
            self.assertEqual(set(by_load_id.keys()), {0, 1})
            for axes in by_load_id.values():
                self.assertEqual(len(axes), 1)  # each load_id serves exactly one axis

    def test_records_written_exactly_once_per_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _make_args(tmp, scope="diagnostic", layer=0, axis="value", task="samsum")
            backend = _FakeBackend()
            _run(args, backend)
            identities = [c[0] for c in backend.trajectory_calls]
            self.assertEqual(len(identities), len(set(identities)))
            records = _read_jsonl(os.path.join(tmp, "testrun", "features.jsonl"))
            record_identities = [(r["task"], r["dataset_index"], r["layer_idx"], r["tensor_axis"]) for r in records]
            self.assertEqual(len(record_identities), len(set(record_identities)))

    def test_resume_skips_completed_group_with_zero_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _make_args(tmp, scope="diagnostic", layer=0, axis="key", task="multifieldqa_en")
            _run(args, _FakeBackend())
            backend2 = _FakeBackend()
            rc2 = _run(args, backend2)
            self.assertEqual(rc2, 0)
            self.assertEqual(len(backend2.load_calls), 0)
            self.assertEqual(len(backend2.trajectory_calls), 0)
            records = _read_jsonl(os.path.join(tmp, "testrun", "features.jsonl"))
            self.assertEqual(len(records), 4)  # not duplicated
            manifest = _read_manifest(os.path.join(tmp, "testrun", "manifest.json"))
            self.assertEqual(manifest["skipped_resumed_count"], 4)

    def test_failure_preserves_earlier_records_and_marks_manifest_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = filter_plan(build_trajectory_plan(), scope="diagnostic", layer=0, axis="key", task="multifieldqa_en")
            fail_identity = trajectory_identity(plan[2])
            args = _make_args(tmp, scope="diagnostic", layer=0, axis="key", task="multifieldqa_en")
            backend = _FakeBackend(fail_on_identity=fail_identity)
            with self.assertRaises(RuntimeError):
                _run(args, backend)
            records = _read_jsonl(os.path.join(tmp, "testrun", "features.jsonl"))
            self.assertEqual(len(records), 2)  # the two before the failure
            manifest = _read_manifest(os.path.join(tmp, "testrun", "manifest.json"))
            self.assertEqual(manifest["state"], "FAILED")
            self.assertEqual(tuple(manifest["failure_identity"]), fail_identity)
            self.assertIn("RuntimeError", manifest["exit_status"])
            self.assertEqual(len(backend.release_calls), 1)  # release still ran (finally)

    def test_resume_after_failure_completes_remaining_without_duplicating(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = filter_plan(build_trajectory_plan(), scope="diagnostic", layer=0, axis="key", task="multifieldqa_en")
            fail_identity = trajectory_identity(plan[2])
            args = _make_args(tmp, scope="diagnostic", layer=0, axis="key", task="multifieldqa_en")
            with self.assertRaises(RuntimeError):
                _run(args, _FakeBackend(fail_on_identity=fail_identity))
            backend2 = _FakeBackend()
            rc2 = _run(args, backend2)
            self.assertEqual(rc2, 0)
            self.assertEqual(len(backend2.trajectory_calls), 2)  # only the never-completed 2
            records = _read_jsonl(os.path.join(tmp, "testrun", "features.jsonl"))
            self.assertEqual(len(records), 4)
            record_identities = {(r["task"], r["dataset_index"], r["layer_idx"], r["tensor_axis"]) for r in records}
            self.assertEqual(len(record_identities), 4)
            manifest = _read_manifest(os.path.join(tmp, "testrun", "manifest.json"))
            self.assertEqual(manifest["state"], "COMPLETE")

    def test_scope_selector_preserves_correct_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _make_args(tmp, scope="primary", layer=0, axis="key", task="trec")
            backend = _FakeBackend()
            _run(args, backend)
            self.assertTrue(all(identity[0] == "trec" for identity, _ in backend.trajectory_calls))
            records = _read_jsonl(os.path.join(tmp, "testrun", "features.jsonl"))
            self.assertTrue(all(r["scope"] == "primary" for r in records))

    def test_max_trajectories_one_exercises_same_path_as_full_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _make_args(tmp, scope="primary", layer=0, axis="key", task="trec", max_trajectories=1)
            backend = _FakeBackend()
            rc = _run(args, backend)
            self.assertEqual(rc, 0)
            self.assertEqual(len(backend.load_calls), 1)
            self.assertEqual(len(backend.trajectory_calls), 1)
            manifest = _read_manifest(os.path.join(tmp, "testrun", "manifest.json"))
            self.assertEqual(manifest["planned_count"], 1)
            self.assertEqual(manifest["state"], "COMPLETE")

    def test_manifest_transitions_running_then_complete(self):
        states_seen = []
        real_write_manifest = _mod.write_manifest

        def spy(path, manifest):
            states_seen.append(manifest["state"])
            real_write_manifest(path, manifest)

        with tempfile.TemporaryDirectory() as tmp:
            args = _make_args(tmp, scope="diagnostic", layer=0, axis="key", task="multifieldqa_en")
            with patch("scripts.run_layer_attention_feature_pilot.write_manifest", side_effect=spy):
                _run(args, _FakeBackend())
        self.assertEqual(states_seen[0], "RUNNING")
        self.assertEqual(states_seen[-1], "COMPLETE")

    def test_manifest_transitions_running_then_failed(self):
        states_seen = []
        real_write_manifest = _mod.write_manifest

        def spy(path, manifest):
            states_seen.append(manifest["state"])
            real_write_manifest(path, manifest)

        with tempfile.TemporaryDirectory() as tmp:
            plan = filter_plan(build_trajectory_plan(), scope="diagnostic", layer=0, axis="key", task="multifieldqa_en")
            fail_identity = trajectory_identity(plan[0])
            args = _make_args(tmp, scope="diagnostic", layer=0, axis="key", task="multifieldqa_en")
            with patch("scripts.run_layer_attention_feature_pilot.write_manifest", side_effect=spy):
                with self.assertRaises(RuntimeError):
                    _run(args, _FakeBackend(fail_on_identity=fail_identity))
        self.assertEqual(states_seen[0], "RUNNING")
        self.assertEqual(states_seen[-1], "FAILED")


class RunRealCollectionRefusalTest(unittest.TestCase):
    def test_dirty_git_refuses_before_any_backend_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _make_args(tmp, scope="diagnostic")
            backend = _FakeBackend()
            with self.assertRaises(PilotConfigError):
                _run(args, backend, git_status_fn=lambda: " M some_file.py\n")
            self.assertEqual(len(backend.load_calls), 0)

    def test_busy_gpu_refuses_before_any_backend_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _make_args(tmp, scope="diagnostic")
            backend = _FakeBackend()
            busy = lambda: {"checked": True, "total_used_mib": 99999, "processes": [], "warning": "GPU busy"}
            with self.assertRaises(PilotConfigError):
                _run(args, backend, gpu_preflight_fn=busy)
            self.assertEqual(len(backend.load_calls), 0)

    def test_conflicting_process_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _make_args(tmp, scope="diagnostic")
            backend = _FakeBackend()
            with self.assertRaises(PilotLockError):
                _run(args, backend, conflict_check_fn=lambda: ["fake pid 1 running run_layer_attention_feature_pilot.py"])
            self.assertEqual(len(backend.load_calls), 0)

    def test_missing_scope_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _make_args(tmp, scope=None)
            with self.assertRaises(PilotConfigError):
                _run(args, _FakeBackend())

    def test_empty_selector_result_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _make_args(tmp, scope="primary", task="trec", dataset_index=999999)
            with self.assertRaises(PilotConfigError):
                _run(args, _FakeBackend())

    def test_resume_with_mismatched_commit_refuses_before_backend_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "testrun")
            os.makedirs(run_dir, exist_ok=True)
            plan = filter_plan(build_trajectory_plan(), scope="diagnostic", layer=0, axis="key", task="multifieldqa_en")
            bad_record = build_trajectory_record(plan[0], DEFAULT_MODEL_NAME, "some-other-commit", DEFAULT_SEED, 100, 5, {1: 0.1})
            with open(os.path.join(run_dir, "features.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps(bad_record) + "\n")
            args = _make_args(tmp, scope="diagnostic", layer=0, axis="key", task="multifieldqa_en")
            backend = _FakeBackend()
            with self.assertRaises(ResumeError):
                _run(args, backend)
            self.assertEqual(len(backend.load_calls), 0)

    def test_resume_with_malformed_json_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "testrun")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "features.jsonl"), "w", encoding="utf-8") as f:
                f.write("{not valid json\n")
            args = _make_args(tmp, scope="diagnostic", layer=0, axis="key", task="multifieldqa_en")
            backend = _FakeBackend()
            with self.assertRaises(ResumeError):
                _run(args, backend)
            self.assertEqual(len(backend.load_calls), 0)


class NoCorrelationOrGateInCollectionPathTest(unittest.TestCase):
    def test_source_never_mentions_correlation_or_gate_computation(self):
        with open(_mod.__file__, "r", encoding="utf-8") as f:
            source = f.read().lower()
        for token in ("spearman", "scipy.stats", "gate_rho", "raw_rho", "abcd_gate"):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
