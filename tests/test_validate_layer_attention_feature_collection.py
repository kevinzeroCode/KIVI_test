"""CPU-only tests for analysis/validate_layer_attention_feature_collection.py
(Stage H3 validator). Synthetic fixtures only -- no real Stage-H data
exists yet. No correlation, no gate -- this script (and these tests) never
compute one.
"""
import json
import os
import tempfile
import unittest

from analysis.validate_layer_attention_feature_collection import validate_collection
from scripts.run_layer_attention_feature_pilot import build_trajectory_plan, build_trajectory_record


def _record_for(trajectory, git_commit="c1"):
    return build_trajectory_record(trajectory, "lmsys/longchat-7b-v1.5-32k", git_commit, 42, 100, 5, {1: 0.1, 2: 0.2})


def _write_run(root, run_label, records, manifest_commit=None):
    run_dir = os.path.join(root, run_label)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "features.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    if manifest_commit is not None:
        with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"git_commit": manifest_commit}, f)
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
            self.assertTrue(any("git_commit" in p for p in result["problems"]))

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


if __name__ == "__main__":
    unittest.main()
