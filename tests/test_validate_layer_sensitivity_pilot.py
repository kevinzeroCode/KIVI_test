"""CPU-only tests for scripts/validate_layer_sensitivity_pilot.py.

Everything here operates on small synthetic temp-dir fixtures -- no GPU, no
real pilot output, no model/dataset download. The real
outputs/layer_sensitivity_pilot/ tree is validated by actually running the
script (see the Stage-D post-run validation report), not by these tests.

Run with:
    ./.venv/bin/python -m unittest tests.test_validate_layer_sensitivity_pilot -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from utils.pilot_policy import PILOT_TASK_COUNTS, write_pilot_policies  # noqa: E402

import validate_layer_sensitivity_pilot as val  # noqa: E402


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _row(i, answers=None, all_classes=None, length=10):
    return {"pred": f"p{i}", "answers": answers or ["a"], "all_classes": all_classes or [], "length": length}


class PilotFixture:
    """Builds a minimal-but-complete synthetic pilot tree (1 condition:
    layer00_key) plus a matching FP16 baseline dir, so validate_* functions
    can be exercised against something structurally real."""

    def __init__(self, tmp):
        self.tmp = tmp
        self.policies_dir = os.path.join(tmp, "policies")
        self.pilot_root = os.path.join(tmp, "pilot")
        self.baseline_dir = os.path.join(tmp, "baseline")
        write_pilot_policies(self.policies_dir)
        os.makedirs(self.pilot_root)
        os.makedirs(self.baseline_dir)

        self.dirname = None  # discovered after first inventory call

    def write_baseline(self, task_counts=PILOT_TASK_COUNTS):
        for task, n in task_counts.items():
            _write_jsonl(os.path.join(self.baseline_dir, f"{task}.jsonl"), [_row(i) for i in range(n)])

    def write_condition(self, condition_id, layer_idx, axis, policy_hash, task_counts=PILOT_TASK_COUNTS, complete=True):
        dirname = f"{condition_id}_{policy_hash}"
        condition_dir = os.path.join(self.pilot_root, dirname)
        os.makedirs(condition_dir, exist_ok=True)
        run_config = {
            "model_name_or_path": "lmsys/longchat-7b-v1.5-32k",
            "condition_id": condition_id,
            "layer_idx": layer_idx,
            "axis": axis,
            "k_bits": 2 if axis == "key" else 16,
            "v_bits": 16 if axis == "key" else 2,
            "policy_path": os.path.join(self.policies_dir, f"{condition_id}_{'k2' if axis == 'key' else 'v2'}.json"),
            "policy_hash": policy_hash,
            "max_length": 31500,
            "group_size": 32,
            "residual_length": 128,
            "seed": 42,
            "tasks": list(task_counts),
            "expected_task_counts": dict(task_counts),
            "pilot": "layer_sensitivity_pilot_stage_d0",
        }
        with open(os.path.join(condition_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(run_config, f)
        for task, n in task_counts.items():
            _write_jsonl(os.path.join(condition_dir, f"{task}.jsonl"), [_row(i) for i in range(n)])
        manifest = {
            "condition_id": condition_id,
            "start_time": "2026-08-19T00:00:00+08:00",
            "end_time": "2026-08-19T01:00:00+08:00",
            "boot_id_start": "boot-a",
            "boot_id_end": "boot-a",
            "boot_id_stable": True,
            "exit_code": 0,
            "nonfinite_logits_seen": False,
            "git_commit": "deadbeef",
            "kernel_log_note": "blocked",
        }
        with open(os.path.join(condition_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        with open(os.path.join(condition_dir, "host_monitor.log"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"timestamp": "t0", "gpu": "40, 1 %, 10 W"}) + "\n")

        pilot_manifest_path = os.path.join(self.pilot_root, "pilot_manifest.json")
        if os.path.exists(pilot_manifest_path):
            with open(pilot_manifest_path, "r", encoding="utf-8") as f:
                pm = json.load(f)
        else:
            pm = {"conditions": []}
        pm["conditions"] = [c for c in pm["conditions"] if c["condition_id"] != condition_id]
        pm["conditions"].append({"condition_id": condition_id, "status": "complete" if complete else "running"})
        with open(pilot_manifest_path, "w", encoding="utf-8") as f:
            json.dump(pm, f)

        return dirname


class TestGlobalOutputInventory(unittest.TestCase):
    def test_missing_all_conditions_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = PilotFixture(tmp)
            inventory, expected = val.inventory_conditions(fx.pilot_root, fx.policies_dir)
            self.assertEqual(inventory["expected_condition_count"], 16)
            self.assertEqual(inventory["actual_condition_dir_count"], 0)
            self.assertEqual(len(inventory["missing_conditions"]), 16)
            self.assertEqual(len(expected), 16)

    def test_unexpected_directory_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = PilotFixture(tmp)
            os.makedirs(os.path.join(fx.pilot_root, "layer99_key_deadbeefcafe"))
            inventory, _ = val.inventory_conditions(fx.pilot_root, fx.policies_dir)
            self.assertIn("layer99_key_deadbeefcafe", inventory["unexpected_directories"])

    def test_all_present_reports_zero_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = PilotFixture(tmp)
            fx.write_condition("layer00_key", 0, "key", "4c4b21298993")
            inventory, expected = val.inventory_conditions(fx.pilot_root, fx.policies_dir)
            self.assertEqual(len(inventory["missing_conditions"]), 15)  # only wrote 1 of 16


class TestStrictJsonlValidation(unittest.TestCase):
    def test_exact_count_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = PilotFixture(tmp)
            fx.write_condition("layer00_key", 0, "key", "4c4b21298993")
            _, expected = val.inventory_conditions(fx.pilot_root, fx.policies_dir)
            expected_one = {k: v for k, v in expected.items() if v["condition_id"] == "layer00_key"}
            result = val.validate_all_jsonl(fx.pilot_root, expected_one)
            self.assertEqual(result["n_files_pass"], result["n_files_checked"])
            self.assertEqual(result["total_rows"], sum(PILOT_TASK_COUNTS.values()))

    def test_over_count_fails_not_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = PilotFixture(tmp)
            dirname = fx.write_condition("layer00_key", 0, "key", "4c4b21298993")
            # Corrupt trec.jsonl to have 201 rows (over-count).
            path = os.path.join(fx.pilot_root, dirname, "trec.jsonl")
            _write_jsonl(path, [_row(i) for i in range(201)])
            _, expected = val.inventory_conditions(fx.pilot_root, fx.policies_dir)
            expected_one = {k: v for k, v in expected.items() if v["condition_id"] == "layer00_key"}
            result = val.validate_all_jsonl(fx.pilot_root, expected_one)
            trec_result = next(r for r in result["results"] if r["task"] == "trec")
            self.assertFalse(trec_result["ok"])
            self.assertLess(result["n_files_pass"], result["n_files_checked"])

    def test_leftover_partial_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = PilotFixture(tmp)
            dirname = fx.write_condition("layer00_key", 0, "key", "4c4b21298993")
            open(os.path.join(fx.pilot_root, dirname, "trec.jsonl.partial"), "w").close()
            _, expected = val.inventory_conditions(fx.pilot_root, fx.policies_dir)
            expected_one = {k: v for k, v in expected.items() if v["condition_id"] == "layer00_key"}
            result = val.validate_all_jsonl(fx.pilot_root, expected_one)
            self.assertEqual(result["n_partial_remaining"], 1)


class TestPairingValidation(unittest.TestCase):
    def test_identical_metadata_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = PilotFixture(tmp)
            fx.write_baseline()
            fx.write_condition("layer00_key", 0, "key", "4c4b21298993")
            _, expected = val.inventory_conditions(fx.pilot_root, fx.policies_dir)
            expected_one = {k: v for k, v in expected.items() if v["condition_id"] == "layer00_key"}
            result = val.validate_pairing_all(fx.pilot_root, expected_one, fx.baseline_dir)
            self.assertEqual(result["n_files_paired"], result["n_files_total"])

    def test_mismatched_metadata_fails_and_never_reorders(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = PilotFixture(tmp)
            fx.write_baseline()
            dirname = fx.write_condition("layer00_key", 0, "key", "4c4b21298993")
            path = os.path.join(fx.pilot_root, dirname, "trec.jsonl")
            rows = [_row(i) for i in range(200)]
            rows[5]["answers"] = ["totally different"]
            _write_jsonl(path, rows)
            _, expected = val.inventory_conditions(fx.pilot_root, fx.policies_dir)
            expected_one = {k: v for k, v in expected.items() if v["condition_id"] == "layer00_key"}
            result = val.validate_pairing_all(fx.pilot_root, expected_one, fx.baseline_dir)
            trec_result = next(r for r in result["results"] if r["task"] == "trec")
            self.assertFalse(trec_result["paired"])
            self.assertEqual(trec_result["n_mismatches"], 1)
            self.assertEqual(trec_result["first_mismatches"][0]["row"], 5)


class TestConditionPolicyValidation(unittest.TestCase):
    def test_correct_policy_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = PilotFixture(tmp)
            fx.write_condition("layer00_key", 0, "key", "4c4b21298993")
            _, expected = val.inventory_conditions(fx.pilot_root, fx.policies_dir)
            expected_one = {k: v for k, v in expected.items() if v["condition_id"] == "layer00_key"}
            results = val.validate_condition_policies(fx.pilot_root, expected_one)
            self.assertTrue(results[0]["ok"])

    def test_wrong_max_length_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = PilotFixture(tmp)
            dirname = fx.write_condition("layer00_key", 0, "key", "4c4b21298993")
            rc_path = os.path.join(fx.pilot_root, dirname, "run_config.json")
            rc = json.load(open(rc_path))
            rc["max_length"] = 4096
            json.dump(rc, open(rc_path, "w"))
            _, expected = val.inventory_conditions(fx.pilot_root, fx.policies_dir)
            expected_one = {k: v for k, v in expected.items() if v["condition_id"] == "layer00_key"}
            results = val.validate_condition_policies(fx.pilot_root, expected_one)
            self.assertFalse(results[0]["ok"])
            self.assertTrue(any("max_length" in p for p in results[0]["problems"]))

    def test_missing_run_config_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = PilotFixture(tmp)
            dirname = fx.write_condition("layer00_key", 0, "key", "4c4b21298993")
            os.remove(os.path.join(fx.pilot_root, dirname, "run_config.json"))
            _, expected = val.inventory_conditions(fx.pilot_root, fx.policies_dir)
            expected_one = {k: v for k, v in expected.items() if v["condition_id"] == "layer00_key"}
            results = val.validate_condition_policies(fx.pilot_root, expected_one)
            self.assertFalse(results[0]["ok"])


class TestManifestValidation(unittest.TestCase):
    def test_complete_condition_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = PilotFixture(tmp)
            fx.write_condition("layer00_key", 0, "key", "4c4b21298993")
            _, expected = val.inventory_conditions(fx.pilot_root, fx.policies_dir)
            expected_one = {k: v for k, v in expected.items() if v["condition_id"] == "layer00_key"}
            top, results = val.validate_manifests(fx.pilot_root, expected_one)
            self.assertTrue(results[0]["ok"])

    def test_nonzero_exit_code_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = PilotFixture(tmp)
            dirname = fx.write_condition("layer00_key", 0, "key", "4c4b21298993")
            m_path = os.path.join(fx.pilot_root, dirname, "manifest.json")
            m = json.load(open(m_path))
            m["exit_code"] = 1
            json.dump(m, open(m_path, "w"))
            _, expected = val.inventory_conditions(fx.pilot_root, fx.policies_dir)
            expected_one = {k: v for k, v in expected.items() if v["condition_id"] == "layer00_key"}
            _, results = val.validate_manifests(fx.pilot_root, expected_one)
            self.assertFalse(results[0]["ok"])

    def test_boot_id_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = PilotFixture(tmp)
            dirname = fx.write_condition("layer00_key", 0, "key", "4c4b21298993")
            m_path = os.path.join(fx.pilot_root, dirname, "manifest.json")
            m = json.load(open(m_path))
            m["boot_id_end"] = "boot-b"
            json.dump(m, open(m_path, "w"))
            _, expected = val.inventory_conditions(fx.pilot_root, fx.policies_dir)
            expected_one = {k: v for k, v in expected.items() if v["condition_id"] == "layer00_key"}
            _, results = val.validate_manifests(fx.pilot_root, expected_one)
            self.assertFalse(results[0]["ok"])

    def test_pilot_manifest_not_complete_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = PilotFixture(tmp)
            fx.write_condition("layer00_key", 0, "key", "4c4b21298993", complete=False)
            _, expected = val.inventory_conditions(fx.pilot_root, fx.policies_dir)
            expected_one = {k: v for k, v in expected.items() if v["condition_id"] == "layer00_key"}
            _, results = val.validate_manifests(fx.pilot_root, expected_one)
            self.assertFalse(results[0]["ok"])


class TestBootIdCheck(unittest.TestCase):
    def test_single_boot_id_no_reboot(self):
        results = [{"boot_id_start": "b1", "boot_id_end": "b1"} for _ in range(16)]
        check = val.check_boot_ids(results)
        self.assertEqual(check["unique_boot_ids_observed"], ["b1"])
        self.assertFalse(check["crossed_reboot"])
        self.assertEqual(check["n_conditions_boot_id_matched"], 16)

    def test_multiple_boot_ids_detected(self):
        results = [{"boot_id_start": "b1", "boot_id_end": "b1"}] * 10 + [{"boot_id_start": "b2", "boot_id_end": "b2"}] * 6
        check = val.check_boot_ids(results)
        self.assertTrue(check["crossed_reboot"])
        self.assertEqual(len(check["unique_boot_ids_observed"]), 2)

    def test_always_reports_kernel_log_limitation(self):
        check = val.check_boot_ids([])
        self.assertIn("permission-blocked", check["kernel_log_limitation"])


class TestMasterLogScan(unittest.TestCase):
    def test_clean_log_no_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "log.txt")
            with open(log_path, "w") as f:
                f.write("Loading checkpoint shards: 100%\n")
                f.write("Token indices sequence length is longer than the specified maximum\n")
            result = val.scan_master_log(log_path)
            self.assertEqual(result["n_flagged_lines"], 0)

    def test_real_traceback_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "log.txt")
            with open(log_path, "w") as f:
                f.write("Traceback (most recent call last):\n")
                f.write("RuntimeError: CUDA out of memory\n")
            result = val.scan_master_log(log_path)
            self.assertGreater(result["n_flagged_lines"], 0)

    def test_benign_reminder_not_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "log.txt")
            with open(log_path, "w") as f:
                f.write("you may observe exceptions, performance degradation, or nothing at all.\n")
            result = val.scan_master_log(log_path)
            self.assertEqual(result["n_flagged_lines"], 0)


class TestCanaryReuseCheck(unittest.TestCase):
    def test_file_predating_condition_start_flagged_as_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = PilotFixture(tmp)
            dirname = fx.write_condition("layer00_key", 0, "key", "4c4b21298993")
            trec_path = os.path.join(fx.pilot_root, dirname, "trec.jsonl")
            old_time = 1755000000  # well before the fixture's 2026-08-19 manifest start_time
            os.utime(trec_path, (old_time, old_time))
            _, expected = val.inventory_conditions(fx.pilot_root, fx.policies_dir)
            result = val.check_canary_reuse(fx.pilot_root, expected)
            self.assertTrue(result["file_predates_full_run_start_evidence_of_reuse"])
            self.assertTrue(result["trec_jsonl_row_count_matches_no_duplication"])


if __name__ == "__main__":
    unittest.main()
