"""CPU-only tests for Stage D0: the layer-sensitivity pilot infrastructure
(utils/pilot_policy.py, scripts/run_layer_sensitivity_pilot.py).

No GPU, no model/dataset download: everything here operates on temp
directories and synthetic fixtures. run_condition() itself (the GPU
generation path) is never imported/exercised here.

Run with:
    ./.venv/bin/python -m unittest tests.test_layer_sensitivity_pilot -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from utils.pilot_policy import (  # noqa: E402
    PILOT_AXES,
    PILOT_LAYERS,
    PILOT_TASK_COUNTS,
    PilotPolicyError,
    all_pilot_specs,
    discover_and_validate_pilot_policies,
    output_dir_name,
    select_task_counts,
    total_example_count,
    write_pilot_policies,
)

import run_layer_sensitivity_pilot as driver  # noqa: E402


# --- condition construction --------------------------------------------------

class TestConditionConstruction(unittest.TestCase):
    def test_exactly_16_conditions(self):
        self.assertEqual(len(all_pilot_specs()), 16)

    def test_exact_layer_list(self):
        self.assertEqual(PILOT_LAYERS, (0, 4, 9, 13, 18, 22, 27, 31))

    def test_exact_two_axes(self):
        self.assertEqual(PILOT_AXES, ("key", "value"))

    def test_deterministic_order_layers_outer_axes_inner(self):
        specs = all_pilot_specs()
        expected_order = []
        for layer in PILOT_LAYERS:
            for axis in PILOT_AXES:
                expected_order.append(f"layer{layer:02d}_{axis}")
        self.assertEqual([s["condition_id"] for s in specs], expected_order)

    def test_order_reproducible_across_calls(self):
        self.assertEqual(
            [s["condition_id"] for s in all_pilot_specs()],
            [s["condition_id"] for s in all_pilot_specs()],
        )


class TestSparsePolicyContents(unittest.TestCase):
    def test_key_probe_policy_shape(self):
        spec = next(s for s in all_pilot_specs() if s["condition_id"] == "layer09_key")
        obj = spec["policy_obj"]
        self.assertEqual(obj["default"], {"k_bits": 16, "v_bits": 16, "family": "kivi"})
        self.assertEqual(set(obj["overrides"]), {"9"})
        self.assertEqual(obj["overrides"]["9"], {"k_bits": 2, "v_bits": 16, "family": "kivi"})

    def test_value_probe_policy_shape(self):
        spec = next(s for s in all_pilot_specs() if s["condition_id"] == "layer22_value")
        obj = spec["policy_obj"]
        self.assertEqual(obj["default"], {"k_bits": 16, "v_bits": 16, "family": "kivi"})
        self.assertEqual(obj["overrides"]["22"], {"k_bits": 16, "v_bits": 2, "family": "kivi"})

    def test_no_k4_or_v4_anywhere(self):
        for spec in all_pilot_specs():
            for cell in [spec["policy_obj"]["default"]] + list(spec["policy_obj"]["overrides"].values()):
                self.assertIn(cell["k_bits"], (2, 16))
                self.assertIn(cell["v_bits"], (2, 16))
                self.assertNotEqual(cell["k_bits"], 4)
                self.assertNotEqual(cell["v_bits"], 4)


# --- policy generation / discovery / hashing --------------------------------

class TestPolicyGenerationAndDiscovery(unittest.TestCase):
    def test_write_then_discover_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            written, unchanged = write_pilot_policies(tmp)
            self.assertEqual(len(written), 16)
            self.assertEqual(len(unchanged), 0)

            conditions = discover_and_validate_pilot_policies(tmp)
            self.assertEqual(len(conditions), 16)

    def test_rewriting_identical_content_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_pilot_policies(tmp)
            written2, unchanged2 = write_pilot_policies(tmp)
            self.assertEqual(len(written2), 0)
            self.assertEqual(len(unchanged2), 16)

    def test_drifted_existing_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_pilot_policies(tmp)
            path = Path(tmp) / "layer00_key_k2.json"
            obj = json.loads(path.read_text())
            obj["default"]["k_bits"] = 4  # hand-tamper
            path.write_text(json.dumps(obj))
            with self.assertRaises(PilotPolicyError):
                write_pilot_policies(tmp)

    def test_missing_policy_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_pilot_policies(tmp)
            os.remove(os.path.join(tmp, "layer31_value_v2.json"))
            with self.assertRaises(PilotPolicyError):
                discover_and_validate_pilot_policies(tmp)

    def test_all_16_hashes_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_pilot_policies(tmp)
            conditions = discover_and_validate_pilot_policies(tmp)
            hashes = [c["resolved_policy_hash"] for c in conditions]
            self.assertEqual(len(hashes), len(set(hashes)))


# --- output directory non-collision -----------------------------------------

class TestOutputDirNonCollision(unittest.TestCase):
    def test_all_16_output_dir_names_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_pilot_policies(tmp)
            conditions = discover_and_validate_pilot_policies(tmp)
            names = [output_dir_name(c) for c in conditions]
            self.assertEqual(len(names), len(set(names)))

    def test_output_dir_name_includes_condition_id_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_pilot_policies(tmp)
            conditions = discover_and_validate_pilot_policies(tmp)
            c = conditions[0]
            name = output_dir_name(c)
            self.assertTrue(name.startswith(c["condition_id"] + "_"))
            self.assertTrue(name.endswith(c["resolved_policy_hash"]))


# --- task counts / totals ----------------------------------------------------

class TestTaskCountsAndTotals(unittest.TestCase):
    def test_expected_task_counts(self):
        self.assertEqual(dict(PILOT_TASK_COUNTS), {
            "trec": 200, "lcc": 500, "passage_retrieval_en": 200, "2wikimqa": 200,
        })

    def test_total_example_count_is_17600(self):
        self.assertEqual(total_example_count(), 17600)

    def test_dataset_evaluations_is_64(self):
        self.assertEqual(len(all_pilot_specs()) * len(PILOT_TASK_COUNTS), 64)


# --- --tasks actually restricts generation (Stage D1 regression: --tasks was
# parsed by argparse but silently ignored by both run_condition() and the
# dry-run/manifest reporting, which always used the full 4-task set) --------

class TestTasksFlagIsRespected(unittest.TestCase):
    def test_select_task_counts_restricts_to_requested_subset(self):
        result = select_task_counts(["trec"])
        self.assertEqual(dict(result), {"trec": 200})

    def test_select_task_counts_preserves_canonical_order_not_arg_order(self):
        # Requested out of order; result must follow PILOT_TASK_COUNTS order.
        result = select_task_counts(["2wikimqa", "trec"])
        self.assertEqual(list(result), ["trec", "2wikimqa"])

    def test_select_task_counts_rejects_unknown_task(self):
        with self.assertRaises(PilotPolicyError):
            select_task_counts(["not_a_real_task"])

    def test_run_condition_signature_accepts_task_counts_and_uses_it(self):
        # Static/structural check (no GPU): run_condition must accept a
        # task_counts parameter and must not hardcode the module-level
        # PILOT_TASK_COUNTS inside its body.
        import inspect

        import run_layer_sensitivity_pilot as drv

        sig = inspect.signature(drv.run_condition)
        self.assertIn("task_counts", sig.parameters)
        source = inspect.getsource(drv.run_condition)
        self.assertNotIn("PILOT_TASK_COUNTS.items()", source)
        self.assertIn("task_counts.items()", source)

    def test_dry_run_total_examples_reflects_single_task_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "--dry-run", "--tasks", "trec",
                "--policies-dir", os.path.join(tmp, "policies"),
                "--output-root", os.path.join(tmp, "out"),
            ]
            with mock.patch.object(sys, "argv", ["run_layer_sensitivity_pilot.py"] + argv):
                with mock.patch("builtins.print") as mock_print:
                    rc = driver.main()
            self.assertEqual(rc, 0)
            printed = "\n".join(str(c.args[0]) for c in mock_print.call_args_list)
            self.assertIn("Total examples: 16 x 200 = 3200", printed)

    def test_initial_manifest_records_only_selected_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_pilot_policies(os.path.join(tmp, "policies"))
            conditions = discover_and_validate_pilot_policies(os.path.join(tmp, "policies"))
            for c in conditions:
                c["output_dir"] = os.path.join(tmp, "out", output_dir_name(c))

            class _Args:
                pass

            manifest = driver.build_initial_pilot_manifest(conditions, _Args(), select_task_counts(["trec", "lcc"]))
            self.assertEqual(manifest["tasks"], ["trec", "lcc"])
            self.assertEqual(manifest["total_examples"], 16 * (200 + 500))
            for c in manifest["conditions"]:
                self.assertEqual(c["tasks"], ["trec", "lcc"])


# --- resume semantics ---------------------------------------------------------

def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class TestResumeSemantics(unittest.TestCase):
    def _row(self, i):
        return {"pred": f"p{i}", "answers": ["a"], "all_classes": [], "length": 10}

    def test_final_file_exact_count_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "trec.jsonl")
            _write_jsonl(out_path, [self._row(i) for i in range(200)])
            action, active_path, done = driver.resolve_pilot_task_resume_plan("trec", out_path, out_path + ".partial", 200)
            self.assertEqual(action, "skip")
            self.assertEqual(done, 200)

    def test_final_file_over_count_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "trec.jsonl")
            _write_jsonl(out_path, [self._row(i) for i in range(201)])  # one too many
            with self.assertRaises(driver.PilotResumeError):
                driver.resolve_pilot_task_resume_plan("trec", out_path, out_path + ".partial", 200)

    def test_final_file_under_count_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "trec.jsonl")
            _write_jsonl(out_path, [self._row(i) for i in range(150)])  # too few for a *final* file
            with self.assertRaises(driver.PilotResumeError):
                driver.resolve_pilot_task_resume_plan("trec", out_path, out_path + ".partial", 200)

    def test_clean_partial_resumes_at_exact_valid_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial_path = os.path.join(tmp, "trec.jsonl.partial")
            out_path = os.path.join(tmp, "trec.jsonl")
            _write_jsonl(partial_path, [self._row(i) for i in range(37)])
            action, active_path, done = driver.resolve_pilot_task_resume_plan("trec", out_path, partial_path, 200)
            self.assertEqual(action, "resume")
            self.assertEqual(done, 37)
            self.assertEqual(active_path, partial_path)

    def test_partial_at_exact_expected_is_ready_to_finalize(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial_path = os.path.join(tmp, "trec.jsonl.partial")
            out_path = os.path.join(tmp, "trec.jsonl")
            _write_jsonl(partial_path, [self._row(i) for i in range(200)])
            action, active_path, done = driver.resolve_pilot_task_resume_plan("trec", out_path, partial_path, 200)
            self.assertEqual(action, "finalize")
            driver.finalize_task_if_ready(action, active_path, out_path, 200)
            self.assertTrue(os.path.exists(out_path))
            self.assertFalse(os.path.exists(partial_path))

    def test_corrupt_partial_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial_path = os.path.join(tmp, "trec.jsonl.partial")
            out_path = os.path.join(tmp, "trec.jsonl")
            with open(partial_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(self._row(0)) + "\n")
                f.write("{not valid json\n")
            with self.assertRaises(driver.PilotResumeError):
                driver.resolve_pilot_task_resume_plan("trec", out_path, partial_path, 200)

    def test_no_files_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "trec.jsonl")
            partial_path = os.path.join(tmp, "trec.jsonl.partial")
            action, active_path, done = driver.resolve_pilot_task_resume_plan("trec", out_path, partial_path, 200)
            self.assertEqual(action, "start")
            self.assertEqual(done, 0)


class TestRunConfigMismatchFailsClosed(unittest.TestCase):
    def _condition(self, layer_idx=9, axis="key", policy_hash="abc123"):
        return {
            "condition_id": f"layer{layer_idx:02d}_{axis}",
            "layer_idx": layer_idx,
            "axis": axis,
            "k_bits": 2 if axis == "key" else 16,
            "v_bits": 16 if axis == "key" else 2,
            "policy_path": "some/path.json",
            "resolved_policy_hash": policy_hash,
        }

    def test_matching_config_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            condition_dir = os.path.join(tmp, "cond")
            cond = self._condition()
            run_config = driver.build_condition_run_config(cond, "lmsys/longchat-7b-v1.5-32k", 31500, 32, 128, 42)
            status1 = driver.prepare_condition_directory(condition_dir, run_config)
            self.assertEqual(status1, "created")
            status2 = driver.prepare_condition_directory(condition_dir, run_config)
            self.assertEqual(status2, "validated")

    def test_different_policy_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            condition_dir = os.path.join(tmp, "cond")
            cond_a = self._condition(policy_hash="hash_a")
            cond_b = self._condition(policy_hash="hash_b")
            run_config_a = driver.build_condition_run_config(cond_a, "lmsys/longchat-7b-v1.5-32k", 31500, 32, 128, 42)
            run_config_b = driver.build_condition_run_config(cond_b, "lmsys/longchat-7b-v1.5-32k", 31500, 32, 128, 42)
            driver.prepare_condition_directory(condition_dir, run_config_a)
            with self.assertRaises(driver.PilotConfigError):
                driver.prepare_condition_directory(condition_dir, run_config_b)

    def test_dir_without_run_config_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            condition_dir = os.path.join(tmp, "cond")
            os.makedirs(condition_dir)
            cond = self._condition()
            run_config = driver.build_condition_run_config(cond, "lmsys/longchat-7b-v1.5-32k", 31500, 32, 128, 42)
            with self.assertRaises(driver.PilotConfigError):
                driver.prepare_condition_directory(condition_dir, run_config)


# --- manifest atomic update/recovery ------------------------------------------

class TestManifestAtomicUpdate(unittest.TestCase):
    def test_write_then_read_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pilot_manifest.json")
            manifest = {"conditions": [{"condition_id": "layer00_key", "status": "pending"}]}
            driver.write_pilot_manifest_atomic(path, manifest)
            reread = driver.read_pilot_manifest(path)
            self.assertEqual(reread, manifest)

    def test_update_status_updates_correct_condition_only(self):
        manifest = {
            "conditions": [
                {"condition_id": "layer00_key", "status": "pending"},
                {"condition_id": "layer00_value", "status": "pending"},
            ]
        }
        driver.update_condition_status(manifest, "layer00_key", "running")
        self.assertEqual(manifest["conditions"][0]["status"], "running")
        self.assertEqual(manifest["conditions"][1]["status"], "pending")

    def test_update_unknown_condition_fails_closed(self):
        manifest = {"conditions": [{"condition_id": "layer00_key", "status": "pending"}]}
        with self.assertRaises(driver.PilotConfigError):
            driver.update_condition_status(manifest, "layer99_key", "running")

    def test_write_leaves_no_tmp_file_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pilot_manifest.json")
            driver.write_pilot_manifest_atomic(path, {"conditions": []})
            files = os.listdir(tmp)
            self.assertEqual(files, ["pilot_manifest.json"])

    def test_original_manifest_survives_a_failed_mid_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pilot_manifest.json")
            driver.write_pilot_manifest_atomic(path, {"conditions": [], "version": 1})
            # Simulate a crash mid-write: os.replace never happens because
            # json.dump raises partway through serialization.
            with mock.patch("json.dump", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    driver.write_pilot_manifest_atomic(path, {"conditions": [], "version": 2})
            # Original file must be untouched -- no partial/corrupt manifest.
            self.assertEqual(driver.read_pilot_manifest(path), {"conditions": [], "version": 1})


# --- dry-run does not launch generation ---------------------------------------

class TestDryRunDoesNotImportTorch(unittest.TestCase):
    def test_dry_run_main_does_not_newly_import_torch(self):
        # Other test modules in the same process may have already imported
        # torch/transformers before this test runs -- that's fine and
        # expected when run alongside GPU-aware suites. What must hold is
        # that *this dry-run call itself* never triggers a new import of
        # either, so we snapshot before and assert no new entry appears.
        modules_before = set(sys.modules)
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--dry-run", "--policies-dir", os.path.join(tmp, "policies"), "--output-root", os.path.join(tmp, "out")]
            with mock.patch.object(sys, "argv", ["run_layer_sensitivity_pilot.py"] + argv):
                rc = driver.main()
            self.assertEqual(rc, 0)
        newly_imported = set(sys.modules) - modules_before
        self.assertNotIn("torch", newly_imported)
        self.assertNotIn("transformers", newly_imported)


class TestConflictingProcessDetection(unittest.TestCase):
    def test_ignores_wrapper_shell_mentioning_pattern(self):
        fake_output = (
            "12345 /bin/bash -c source snapshot.sh && eval 'echo layer_policy_smoke.py'\n"
        ).encode()
        with mock.patch("subprocess.check_output", return_value=fake_output):
            conflicts = driver.check_no_conflicting_process(self_pid=1)
        self.assertEqual(conflicts, [])

    def test_detects_genuine_python_invocation(self):
        fake_output = (
            "12345 ./.venv/bin/python scripts/layer_policy_smoke.py --num_samples 5\n"
        ).encode()
        with mock.patch("subprocess.check_output", return_value=fake_output):
            conflicts = driver.check_no_conflicting_process(self_pid=1)
        self.assertEqual(len(conflicts), 1)

    def test_excludes_self_pid(self):
        fake_output = (
            "777 ./.venv/bin/python scripts/run_layer_sensitivity_pilot.py --dry-run\n"
        ).encode()
        with mock.patch("subprocess.check_output", return_value=fake_output):
            conflicts = driver.check_no_conflicting_process(self_pid=777)
        self.assertEqual(conflicts, [])

    def test_no_pgrep_match_returns_empty(self):
        import subprocess as sp
        with mock.patch("subprocess.check_output", side_effect=sp.CalledProcessError(1, "pgrep")):
            conflicts = driver.check_no_conflicting_process(self_pid=1)
        self.assertEqual(conflicts, [])


# --- single-instance lock behavior --------------------------------------------

class TestSingleInstanceLock(unittest.TestCase):
    def test_second_lock_attempt_fails_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = os.path.join(tmp, ".pilot.lock")
            fh1 = driver.acquire_pilot_lock(lock_path)
            try:
                with self.assertRaises(driver.PilotLockError):
                    driver.acquire_pilot_lock(lock_path)
            finally:
                driver.release_pilot_lock(fh1)

    def test_lock_released_allows_reacquisition(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = os.path.join(tmp, ".pilot.lock")
            fh1 = driver.acquire_pilot_lock(lock_path)
            driver.release_pilot_lock(fh1)
            fh2 = driver.acquire_pilot_lock(lock_path)
            driver.release_pilot_lock(fh2)  # must not raise


# --- NaN/Inf instrumentation (Stage D1 gap found after the canary: run_condition
# had no direct NaN/Inf check, unlike scripts/layer_policy_smoke.py) ---------

class TestNanInfInstrumentation(unittest.TestCase):
    def test_run_condition_registers_lm_head_hook_and_records_flag(self):
        import inspect

        import run_layer_sensitivity_pilot as drv

        source = inspect.getsource(drv.run_condition)
        self.assertIn("register_forward_hook", source)
        self.assertIn("nonfinite_logits_seen", source)
        self.assertIn("torch.isfinite", source)
        # Hook must be removed unconditionally (finally block), not only on
        # the success path, and must not crash if model construction itself
        # failed before the hook was ever registered.
        self.assertIn("hook_handle = None", source)
        self.assertIn("if hook_handle is not None", source)


# --- resume dry-run report (Part E: dry-run must prove a completed
# condition/task is recognized as SKIP, using the exact-count rule) ---------

class TestResumeReport(unittest.TestCase):
    def test_completed_task_reports_skip_at_exact_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_pilot_policies(os.path.join(tmp, "policies"))
            conditions = discover_and_validate_pilot_policies(os.path.join(tmp, "policies"), layers=(0,), axes=("key",))
            for c in conditions:
                c["output_dir"] = os.path.join(tmp, "out", output_dir_name(c))
            condition_dir = conditions[0]["output_dir"]
            os.makedirs(condition_dir)
            _write_jsonl(os.path.join(condition_dir, "trec.jsonl"), [{"pred": "x", "answers": ["a"], "all_classes": [], "length": 1} for _ in range(200)])

            report = driver.compute_resume_report(conditions, select_task_counts(["trec"]), os.path.join(tmp, "out"))
            self.assertEqual(len(report), 1)
            self.assertEqual(report[0], {"condition_id": "layer00_key", "task": "trec", "action": "skip", "done": 200, "expected": 200})

    def test_over_count_reports_error_not_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_pilot_policies(os.path.join(tmp, "policies"))
            conditions = discover_and_validate_pilot_policies(os.path.join(tmp, "policies"), layers=(0,), axes=("key",))
            for c in conditions:
                c["output_dir"] = os.path.join(tmp, "out", output_dir_name(c))
            condition_dir = conditions[0]["output_dir"]
            os.makedirs(condition_dir)
            _write_jsonl(os.path.join(condition_dir, "trec.jsonl"), [{"pred": "x", "answers": ["a"], "all_classes": [], "length": 1} for _ in range(201)])

            report = driver.compute_resume_report(conditions, select_task_counts(["trec"]), os.path.join(tmp, "out"))
            self.assertEqual(report[0]["action"], "ERROR")

    def test_no_existing_output_reports_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_pilot_policies(os.path.join(tmp, "policies"))
            conditions = discover_and_validate_pilot_policies(os.path.join(tmp, "policies"), layers=(0,), axes=("key",))
            for c in conditions:
                c["output_dir"] = os.path.join(tmp, "out", output_dir_name(c))
            report = driver.compute_resume_report(conditions, select_task_counts(["trec"]), os.path.join(tmp, "out"))
            self.assertEqual(report[0]["action"], "start")


# --- baseline audit ------------------------------------------------------------

class TestBaselineAudit(unittest.TestCase):
    def test_reusable_true_when_all_tasks_present_and_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            for task, n in PILOT_TASK_COUNTS.items():
                _write_jsonl(
                    os.path.join(tmp, f"{task}.jsonl"),
                    [{"pred": "x", "answers": ["a"], "all_classes": [], "length": 1} for _ in range(n)],
                )
            audit = driver.audit_fp16_baseline_reuse(tmp, PILOT_TASK_COUNTS)
            self.assertTrue(audit["reusable"])

    def test_reusable_false_when_a_task_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = dict(PILOT_TASK_COUNTS)
            tasks.pop("lcc")
            for task, n in tasks.items():
                _write_jsonl(
                    os.path.join(tmp, f"{task}.jsonl"),
                    [{"pred": "x", "answers": ["a"], "all_classes": [], "length": 1} for _ in range(n)],
                )
            audit = driver.audit_fp16_baseline_reuse(tmp, PILOT_TASK_COUNTS)
            self.assertFalse(audit["reusable"])


if __name__ == "__main__":
    unittest.main()
