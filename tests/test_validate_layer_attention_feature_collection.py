"""CPU-only tests for analysis/validate_layer_attention_feature_collection.py
(Stage H3 validator). Synthetic fixtures only -- no real Stage-H data
exists yet. No correlation, no gate -- this script (and these tests) never
compute one.
"""
import json
import os
import tempfile
import unittest

from analysis.validate_layer_attention_feature_collection import validate_collection, validate_partial_collection
from scripts.run_layer_attention_feature_pilot import build_trajectory_plan, build_trajectory_record, filter_plan, trajectory_identity


def _record_for(trajectory, git_commit="c1"):
    return build_trajectory_record(trajectory, "lmsys/longchat-7b-v1.5-32k", git_commit, 42, 100, 5, {1: 0.1, 2: 0.2})


def _write_run(root, run_label, records, manifest_commit=None, manifest_extra=None):
    """`manifest_commit` maps to the REAL H3B manifest field name
    (collection_head), not the pre-H3B placeholder `git_commit` key this
    helper used to write (which never matched any real manifest)."""
    run_dir = os.path.join(root, run_label)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "features.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    if manifest_commit is not None or manifest_extra is not None:
        manifest = {}
        if manifest_commit is not None:
            manifest["collection_head"] = manifest_commit
        if manifest_extra is not None:
            manifest.update(manifest_extra)
        with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)
    return run_dir


class ValidateCollectionSyntheticTest(unittest.TestCase):
    def test_empty_roots_report_not_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = validate_collection(os.path.join(tmp, "primary"), os.path.join(tmp, "diagnostic"))
            self.assertFalse(result["ok"])
            self.assertEqual(result["primary_record_count"], 0)
            self.assertEqual(result["diagnostic_record_count"], 0)

    def test_full_complete_synthetic_collection_passes(self):
        plan = build_trajectory_plan()
        primary_plan = [t for t in plan if t["scope"] == "primary"]
        diagnostic_plan = [t for t in plan if t["scope"] == "diagnostic"]

        with tempfile.TemporaryDirectory() as tmp:
            primary_root = os.path.join(tmp, "primary")
            diagnostic_root = os.path.join(tmp, "diagnostic")
            _write_run(primary_root, "run1", [_record_for(t) for t in primary_plan], manifest_commit="c1")
            _write_run(diagnostic_root, "run1", [_record_for(t) for t in diagnostic_plan], manifest_commit="c1")

            result = validate_collection(primary_root, diagnostic_root, expected_commit="c1")
            self.assertTrue(result["ok"], result["problems"])
            self.assertEqual(result["primary_record_count"], 256)
            self.assertEqual(result["diagnostic_record_count"], 16)
            self.assertEqual(result["total_record_count"], 272)
            self.assertEqual(result["unique_prompt_identities"], 24)
            self.assertEqual(result["unique_scientific_identities"], 272)

    def test_missing_records_detected(self):
        plan = build_trajectory_plan()
        primary_plan = [t for t in plan if t["scope"] == "primary"][:-1]  # drop one
        diagnostic_plan = [t for t in plan if t["scope"] == "diagnostic"]

        with tempfile.TemporaryDirectory() as tmp:
            primary_root = os.path.join(tmp, "primary")
            diagnostic_root = os.path.join(tmp, "diagnostic")
            _write_run(primary_root, "run1", [_record_for(t) for t in primary_plan])
            _write_run(diagnostic_root, "run1", [_record_for(t) for t in diagnostic_plan])

            result = validate_collection(primary_root, diagnostic_root)
            self.assertFalse(result["ok"])
            self.assertTrue(any("missing" in p for p in result["problems"]))

    def test_duplicate_identity_across_run_dirs_detected(self):
        plan = build_trajectory_plan()
        primary_plan = [t for t in plan if t["scope"] == "primary"]
        diagnostic_plan = [t for t in plan if t["scope"] == "diagnostic"]

        with tempfile.TemporaryDirectory() as tmp:
            primary_root = os.path.join(tmp, "primary")
            diagnostic_root = os.path.join(tmp, "diagnostic")
            _write_run(primary_root, "run1", [_record_for(t) for t in primary_plan])
            # duplicate the FIRST primary trajectory in a second run dir
            _write_run(primary_root, "run2", [_record_for(primary_plan[0])])
            _write_run(diagnostic_root, "run1", [_record_for(t) for t in diagnostic_plan])

            result = validate_collection(primary_root, diagnostic_root)
            self.assertFalse(result["ok"])
            self.assertTrue(any("duplicate identity" in p for p in result["problems"]))

    def test_manifest_commit_mismatch_detected(self):
        plan = build_trajectory_plan()
        primary_plan = [t for t in plan if t["scope"] == "primary"]
        diagnostic_plan = [t for t in plan if t["scope"] == "diagnostic"]

        with tempfile.TemporaryDirectory() as tmp:
            primary_root = os.path.join(tmp, "primary")
            diagnostic_root = os.path.join(tmp, "diagnostic")
            _write_run(primary_root, "run1", [_record_for(t, git_commit="c1") for t in primary_plan], manifest_commit="c1")
            _write_run(diagnostic_root, "run1", [_record_for(t, git_commit="c1") for t in diagnostic_plan], manifest_commit="c1")

            result = validate_collection(primary_root, diagnostic_root, expected_commit="different_commit")
            self.assertFalse(result["ok"])
            self.assertTrue(any("collection_head" in p for p in result["problems"]))

    def test_malformed_record_rejected(self):
        plan = build_trajectory_plan()
        primary_plan = [t for t in plan if t["scope"] == "primary"]
        diagnostic_plan = [t for t in plan if t["scope"] == "diagnostic"]

        with tempfile.TemporaryDirectory() as tmp:
            primary_root = os.path.join(tmp, "primary")
            diagnostic_root = os.path.join(tmp, "diagnostic")
            records = [_record_for(t) for t in primary_plan]
            records[0]["sample_attention_distortion"] = 999.0  # break internal consistency
            _write_run(primary_root, "run1", records)
            _write_run(diagnostic_root, "run1", [_record_for(t) for t in diagnostic_plan])

            result = validate_collection(primary_root, diagnostic_root)
            self.assertFalse(result["ok"])
            self.assertLess(result["per_record_structurally_valid"], result["total_record_count"])

    def test_no_correlation_or_gate_keys_present(self):
        # Structural guard: the validator's own output must never contain
        # a Spearman/gate-shaped key -- it is read-only integrity only.
        with tempfile.TemporaryDirectory() as tmp:
            result = validate_collection(os.path.join(tmp, "primary"), os.path.join(tmp, "diagnostic"))
            forbidden_substrings = ("spearman", "rho", "gate", "FEATURE_PILOT_STAGE", "correlation")
            result_str = json.dumps(result).lower()
            for token in forbidden_substrings:
                self.assertNotIn(token.lower(), result_str)


class ValidatePartialCollectionTest(unittest.TestCase):
    """Stage H3C-CANARY (Part 2): the full validator's 256/16/272 assumption
    is wrong for a selector-limited canary -- these tests exercise the
    separate validate_partial_collection() path against a synthetic
    single-trajectory (and multi-trajectory) canary instead.
    """

    def _one_trajectory_plan(self, task="lcc", dataset_index=122, layer=0, axis="key"):
        return filter_plan(build_trajectory_plan(), task=task, dataset_index=dataset_index, layer=layer, axis=axis)

    def test_empty_expected_plan_raises(self):
        with self.assertRaises(ValueError):
            validate_partial_collection("/nonexistent", "primary", [])

    def test_single_trajectory_complete_and_valid_passes(self):
        plan = self._one_trajectory_plan()
        self.assertEqual(len(plan), 1)
        with tempfile.TemporaryDirectory() as tmp:
            _write_run(
                tmp, "h3c_key_l0_lcc122", [_record_for(plan[0], git_commit="deadbeef")],
                manifest_commit="deadbeef", manifest_extra={"planned_count": 1, "completed_count": 1, "state": "COMPLETE"},
            )
            result = validate_partial_collection(tmp, "primary", plan, expected_commit="deadbeef")
            self.assertTrue(result["ok"], result["problems"])
            self.assertEqual(result["record_count"], 1)
            self.assertEqual(result["expected_count"], 1)
            self.assertEqual(result["unique_scientific_identities"], 1)
            self.assertEqual(result["manifest_states"], ["COMPLETE"])

    def test_wrong_identity_is_rejected_as_extra_and_missing(self):
        plan = self._one_trajectory_plan()  # lcc/122/0/key
        other = filter_plan(build_trajectory_plan(), task="lcc", dataset_index=174, layer=0, axis="key")
        with tempfile.TemporaryDirectory() as tmp:
            _write_run(
                tmp, "run1", [_record_for(other[0], git_commit="deadbeef")],
                manifest_commit="deadbeef", manifest_extra={"planned_count": 1, "completed_count": 1, "state": "COMPLETE"},
            )
            result = validate_partial_collection(tmp, "primary", plan, expected_commit="deadbeef")
            self.assertFalse(result["ok"])
            self.assertTrue(any("missing" in p for p in result["problems"]))
            self.assertTrue(any("unexpected identities" in p for p in result["problems"]))

    def test_manifest_not_complete_is_rejected(self):
        plan = self._one_trajectory_plan()
        with tempfile.TemporaryDirectory() as tmp:
            _write_run(
                tmp, "run1", [_record_for(plan[0], git_commit="deadbeef")],
                manifest_commit="deadbeef", manifest_extra={"planned_count": 1, "completed_count": 0, "state": "RUNNING"},
            )
            result = validate_partial_collection(tmp, "primary", plan, expected_commit="deadbeef")
            self.assertFalse(result["ok"])
            self.assertTrue(any("state=" in p for p in result["problems"]))
            self.assertTrue(any("completed_count" in p for p in result["problems"]))

    def test_missing_manifest_is_rejected(self):
        plan = self._one_trajectory_plan()
        with tempfile.TemporaryDirectory() as tmp:
            _write_run(tmp, "run1", [_record_for(plan[0], git_commit="deadbeef")])  # no manifest
            result = validate_partial_collection(tmp, "primary", plan, expected_commit="deadbeef")
            self.assertFalse(result["ok"])
            self.assertTrue(any("no manifest.json found" in p for p in result["problems"]))

    def test_duplicate_record_for_same_identity_is_rejected(self):
        plan = self._one_trajectory_plan()
        with tempfile.TemporaryDirectory() as tmp:
            _write_run(
                tmp, "run1", [_record_for(plan[0], git_commit="deadbeef"), _record_for(plan[0], git_commit="deadbeef")],
                manifest_commit="deadbeef", manifest_extra={"planned_count": 1, "completed_count": 1, "state": "COMPLETE"},
            )
            result = validate_partial_collection(tmp, "primary", plan, expected_commit="deadbeef")
            self.assertFalse(result["ok"])
            self.assertTrue(any("duplicate identity" in p for p in result["problems"]))

    def test_commit_mismatch_is_rejected(self):
        plan = self._one_trajectory_plan()
        with tempfile.TemporaryDirectory() as tmp:
            _write_run(
                tmp, "run1", [_record_for(plan[0], git_commit="deadbeef")],
                manifest_commit="deadbeef", manifest_extra={"planned_count": 1, "completed_count": 1, "state": "COMPLETE"},
            )
            result = validate_partial_collection(tmp, "primary", plan, expected_commit="a-different-commit")
            self.assertFalse(result["ok"])
            self.assertTrue(any("collection_head" in p for p in result["problems"]))

    def test_malformed_record_rejected_in_partial_mode(self):
        plan = self._one_trajectory_plan()
        with tempfile.TemporaryDirectory() as tmp:
            record = _record_for(plan[0], git_commit="deadbeef")
            record["sample_attention_distortion"] = 999.0  # break internal consistency
            _write_run(
                tmp, "run1", [record],
                manifest_commit="deadbeef", manifest_extra={"planned_count": 1, "completed_count": 1, "state": "COMPLETE"},
            )
            result = validate_partial_collection(tmp, "primary", plan, expected_commit="deadbeef")
            self.assertFalse(result["ok"])
            self.assertLess(result["per_record_structurally_valid"], result["record_count"])

    def test_three_trajectory_canary_plan_passes(self):
        # Mirrors the exact Stage H3C-CANARY scope: Key + Value on lcc/122/layer0,
        # plus samsum/58/layer0/value -- three trajectories across two scopes,
        # validated independently per canary root (as the real canaries will be).
        key_plan = filter_plan(build_trajectory_plan(), task="lcc", dataset_index=122, layer=0, axis="key")
        value_plan = filter_plan(build_trajectory_plan(), task="lcc", dataset_index=122, layer=0, axis="value")
        samsum_plan = filter_plan(build_trajectory_plan(), task="samsum", dataset_index=58, layer=0, axis="value")
        with tempfile.TemporaryDirectory() as tmp:
            key_root = os.path.join(tmp, "h3c_key_l0_lcc122")
            value_root = os.path.join(tmp, "h3c_value_l0_lcc122")
            samsum_root = os.path.join(tmp, "h3c_value_l0_samsum58")
            _write_run(key_root, "run1", [_record_for(key_plan[0], git_commit="deadbeef")],
                       manifest_commit="deadbeef", manifest_extra={"planned_count": 1, "completed_count": 1, "state": "COMPLETE"})
            _write_run(value_root, "run1", [_record_for(value_plan[0], git_commit="deadbeef")],
                       manifest_commit="deadbeef", manifest_extra={"planned_count": 1, "completed_count": 1, "state": "COMPLETE"})
            _write_run(samsum_root, "run1", [_record_for(samsum_plan[0], git_commit="deadbeef")],
                       manifest_commit="deadbeef", manifest_extra={"planned_count": 1, "completed_count": 1, "state": "COMPLETE"})
            for root, plan, scope in ((key_root, key_plan, "primary"), (value_root, value_plan, "primary"), (samsum_root, samsum_plan, "diagnostic")):
                result = validate_partial_collection(root, scope, plan, expected_commit="deadbeef")
                self.assertTrue(result["ok"], (root, result["problems"]))


if __name__ == "__main__":
    unittest.main()
