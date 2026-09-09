"""CPU-only tests for Stage I2A: the layer x quantizer-family sensitivity
experiment infrastructure (utils/i2_layer_family_conditions.py,
scripts/run_i2_layer_family_sensitivity.py).

No GPU, no model/dataset download: everything here operates on temp
directories and synthetic fixtures. run_condition() and
run_i2b_canary_condition() (the GPU paths) are never imported/exercised
here.

Run with:
    ./.venv/bin/python -m unittest tests.test_i2_layer_family_sensitivity -v
"""
import argparse
import inspect
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from utils.i2_layer_family_conditions import (  # noqa: E402
    I2_FAMILIES,
    I2_GROUP_SIZE,
    I2_K_BITS,
    I2_LAYERS,
    I2_RESIDUAL_LENGTH,
    I2_TASK_COUNTS,
    I2_V_BITS,
    TOTAL_CONDITIONS,
    TOTAL_PLANNED_ROWS,
    I2ConditionError,
    all_i2_specs,
    build_policy_obj,
    condition_id,
    discover_and_validate_i2_policies,
    output_dir_name,
    total_example_count,
    write_i2_policies,
)

import run_i2_layer_family_sensitivity as driver  # noqa: E402


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            json.dump(r, f)
            f.write("\n")


# --- condition construction --------------------------------------------------

class TestConditionConstruction(unittest.TestCase):
    def test_exactly_16_conditions(self):
        self.assertEqual(len(all_i2_specs()), 16)
        self.assertEqual(TOTAL_CONDITIONS, 16)

    def test_exact_layer_list(self):
        self.assertEqual(I2_LAYERS, (0, 4, 9, 13, 18, 22, 27, 31))

    def test_exact_two_families(self):
        self.assertEqual(I2_FAMILIES, ("kivi", "rotation_kivi"))

    def test_fixed_k2_v16(self):
        self.assertEqual(I2_K_BITS, 2)
        self.assertEqual(I2_V_BITS, 16)
        self.assertEqual(I2_GROUP_SIZE, 32)
        self.assertEqual(I2_RESIDUAL_LENGTH, 128)

    def test_deterministic_order_layers_outer_families_inner(self):
        specs = all_i2_specs()
        expected_order = [condition_id(l, f) for l in I2_LAYERS for f in I2_FAMILIES]
        self.assertEqual([s["condition_id"] for s in specs], expected_order)

    def test_order_reproducible_across_calls(self):
        self.assertEqual(
            [s["condition_id"] for s in all_i2_specs()],
            [s["condition_id"] for s in all_i2_specs()],
        )

    def test_condition_uniqueness(self):
        ids = [s["condition_id"] for s in all_i2_specs()]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 16)


class TestOneTargetLayerPolicyConstruction(unittest.TestCase):
    def test_kivi_target_policy_shape(self):
        obj = build_policy_obj(9, "kivi")
        self.assertEqual(obj["default"], {"k_bits": 16, "v_bits": 16, "family": "kivi"})
        self.assertEqual(set(obj["overrides"]), {"9"})
        self.assertEqual(obj["overrides"]["9"], {"k_bits": 2, "v_bits": 16, "family": "kivi"})

    def test_rotation_target_policy_shape(self):
        obj = build_policy_obj(22, "rotation_kivi")
        self.assertEqual(obj["default"], {"k_bits": 16, "v_bits": 16, "family": "kivi"})
        self.assertEqual(obj["overrides"]["22"], {"k_bits": 2, "v_bits": 16, "family": "rotation_kivi"})

    def test_all_non_target_layers_are_k16_v16_kivi(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_i2_policies(tmp)
            conditions = discover_and_validate_i2_policies(tmp)
            for cond in conditions:
                resolved = cond["resolved_layer_policy"]
                for i, entry in enumerate(resolved.layers):
                    if i == cond["layer_idx"]:
                        continue
                    self.assertEqual((entry.k_bits, entry.v_bits, entry.family), (16, 16, "kivi"))

    def test_target_layer_matches_condition_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_i2_policies(tmp)
            conditions = discover_and_validate_i2_policies(tmp)
            for cond in conditions:
                entry = cond["resolved_layer_policy"].layers[cond["layer_idx"]]
                self.assertEqual((entry.k_bits, entry.v_bits, entry.family), (2, 16, cond["family"]))

    def test_no_k4_or_v_quantization_anywhere(self):
        for spec in all_i2_specs():
            for cell in [spec["policy_obj"]["default"]] + list(spec["policy_obj"]["overrides"].values()):
                self.assertIn(cell["k_bits"], (2, 16))
                self.assertEqual(cell["v_bits"], 16)
                self.assertNotEqual(cell["k_bits"], 4)


# --- policy generation / discovery / hashing --------------------------------

class TestPolicyGenerationAndDiscovery(unittest.TestCase):
    def test_write_then_discover_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            written, unchanged = write_i2_policies(tmp)
            self.assertEqual(len(written), 16)
            self.assertEqual(len(unchanged), 0)
            conditions = discover_and_validate_i2_policies(tmp)
            self.assertEqual(len(conditions), 16)

    def test_rewriting_identical_content_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_i2_policies(tmp)
            written2, unchanged2 = write_i2_policies(tmp)
            self.assertEqual(len(written2), 0)
            self.assertEqual(len(unchanged2), 16)

    def test_drifted_existing_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_i2_policies(tmp)
            path = Path(tmp) / "layer00_kivi.json"
            obj = json.loads(path.read_text())
            obj["default"]["k_bits"] = 4
            path.write_text(json.dumps(obj))
            with self.assertRaises(I2ConditionError):
                write_i2_policies(tmp)

    def test_missing_policy_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_i2_policies(tmp)
            os.remove(os.path.join(tmp, "layer31_rotation_kivi.json"))
            with self.assertRaises(I2ConditionError):
                discover_and_validate_i2_policies(tmp)

    def test_all_16_hashes_unique_ie_condition_uniqueness(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_i2_policies(tmp)
            conditions = discover_and_validate_i2_policies(tmp)
            hashes = [c["resolved_policy_hash"] for c in conditions]
            self.assertEqual(len(hashes), len(set(hashes)))
            self.assertEqual(len(hashes), 16)

    def test_stable_policy_hash_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_i2_policies(tmp)
            c1 = discover_and_validate_i2_policies(tmp)
            c2 = discover_and_validate_i2_policies(tmp)
            self.assertEqual(
                [c["resolved_policy_hash"] for c in c1],
                [c["resolved_policy_hash"] for c in c2],
            )

    def test_corrupted_policy_shape_fails_closed(self):
        """A hand-edited policy that no longer matches the one-target-layer
        contract (e.g. a second layer perturbed) must be rejected at
        discovery time, not silently accepted."""
        with tempfile.TemporaryDirectory() as tmp:
            write_i2_policies(tmp)
            path = Path(tmp) / "layer00_kivi.json"
            obj = json.loads(path.read_text())
            obj["overrides"]["5"] = {"k_bits": 2, "v_bits": 16, "family": "kivi"}
            path.write_text(json.dumps(obj))
            with self.assertRaises(I2ConditionError):
                discover_and_validate_i2_policies(tmp)


# --- output directory non-collision -----------------------------------------

class TestOutputDirNonCollision(unittest.TestCase):
    def test_all_16_output_dir_names_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_i2_policies(tmp)
            conditions = discover_and_validate_i2_policies(tmp)
            names = [output_dir_name(c) for c in conditions]
            self.assertEqual(len(names), len(set(names)))

    def test_output_dir_name_includes_condition_id_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_i2_policies(tmp)
            conditions = discover_and_validate_i2_policies(tmp)
            c = conditions[0]
            name = output_dir_name(c)
            self.assertTrue(name.startswith(c["condition_id"] + "_"))
            self.assertTrue(name.endswith(c["resolved_policy_hash"]))


# --- task counts / totals ----------------------------------------------------

class TestTaskCountsAndTotals(unittest.TestCase):
    def test_exact_six_tasks(self):
        self.assertEqual(
            list(I2_TASK_COUNTS),
            ["trec", "lcc", "passage_retrieval_en", "2wikimqa", "multifieldqa_en", "samsum"],
        )

    def test_exact_expected_counts(self):
        self.assertEqual(dict(I2_TASK_COUNTS), {
            "trec": 200, "lcc": 500, "passage_retrieval_en": 200,
            "2wikimqa": 200, "multifieldqa_en": 150, "samsum": 200,
        })

    def test_total_planned_rows_is_23200(self):
        self.assertEqual(TOTAL_PLANNED_ROWS, 23200)
        self.assertEqual(total_example_count(), 23200)

    def test_total_rows_equals_conditions_times_samples_per_condition(self):
        self.assertEqual(TOTAL_CONDITIONS * sum(I2_TASK_COUNTS.values()), 23200)


# --- output-root isolation ----------------------------------------------------

class TestOutputRootIsolation(unittest.TestCase):
    def test_default_output_root_is_dedicated(self):
        self.assertTrue(driver.DEFAULT_OUTPUT_ROOT.endswith("outputs/i2_layer_family_sensitivity"))
        for forbidden in ("layer_sensitivity_pilot", "layer_attention_feature_pilot", "i1_rotation_kivi_canary", "/pred"):
            self.assertNotIn(forbidden, driver.DEFAULT_OUTPUT_ROOT)

    def test_default_policies_dir_is_dedicated(self):
        self.assertTrue(driver.DEFAULT_POLICIES_DIR.endswith("i2_layer_family_sensitivity"))


# --- resume semantics (exact-count-only contract) ----------------------------

class TestResumeSemantics(unittest.TestCase):
    def _row(self, i):
        return {"dataset_index": i, "pred": f"p{i}", "answers": ["a"], "all_classes": [], "length": 10}

    def test_final_file_exact_count_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "trec.jsonl")
            _write_jsonl(out_path, [self._row(i) for i in range(200)])
            action, active_path, done = driver.resolve_i2_task_resume_plan("trec", out_path, out_path + ".partial", 200)
            self.assertEqual(action, "skip")
            self.assertEqual(done, 200)

    def test_final_file_over_count_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "trec.jsonl")
            _write_jsonl(out_path, [self._row(i) for i in range(201)])
            with self.assertRaises(driver.I2ResumeError):
                driver.resolve_i2_task_resume_plan("trec", out_path, out_path + ".partial", 200)

    def test_final_file_under_count_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "trec.jsonl")
            _write_jsonl(out_path, [self._row(i) for i in range(150)])
            with self.assertRaises(driver.I2ResumeError):
                driver.resolve_i2_task_resume_plan("trec", out_path, out_path + ".partial", 200)

    def test_clean_partial_resumes_at_exact_valid_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial_path = os.path.join(tmp, "trec.jsonl.partial")
            out_path = os.path.join(tmp, "trec.jsonl")
            _write_jsonl(partial_path, [self._row(i) for i in range(37)])
            action, active_path, done = driver.resolve_i2_task_resume_plan("trec", out_path, partial_path, 200)
            self.assertEqual(action, "resume")
            self.assertEqual(done, 37)

    def test_partial_at_exact_expected_is_ready_to_finalize(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial_path = os.path.join(tmp, "trec.jsonl.partial")
            out_path = os.path.join(tmp, "trec.jsonl")
            _write_jsonl(partial_path, [self._row(i) for i in range(200)])
            action, active_path, done = driver.resolve_i2_task_resume_plan("trec", out_path, partial_path, 200)
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
            with self.assertRaises(driver.I2ResumeError):
                driver.resolve_i2_task_resume_plan("trec", out_path, partial_path, 200)

    def test_no_files_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "trec.jsonl")
            partial_path = os.path.join(tmp, "trec.jsonl.partial")
            action, active_path, done = driver.resolve_i2_task_resume_plan("trec", out_path, partial_path, 200)
            self.assertEqual(action, "start")
            self.assertEqual(done, 0)


# --- run_config mismatch fails closed -----------------------------------------

class TestRunConfigMismatchFailsClosed(unittest.TestCase):
    def _condition(self, layer_idx=9, family="kivi", policy_hash="abc123"):
        return {
            "condition_id": f"layer{layer_idx:02d}_{family}",
            "layer_idx": layer_idx,
            "family": family,
            "k_bits": 2,
            "v_bits": 16,
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
            with self.assertRaises(driver.I2ConfigError):
                driver.prepare_condition_directory(condition_dir, run_config_b)

    def test_dir_without_run_config_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            condition_dir = os.path.join(tmp, "cond")
            os.makedirs(condition_dir)
            cond = self._condition()
            run_config = driver.build_condition_run_config(cond, "lmsys/longchat-7b-v1.5-32k", 31500, 32, 128, 42)
            with self.assertRaises(driver.I2ConfigError):
                driver.prepare_condition_directory(condition_dir, run_config)


# --- manifest atomic update ----------------------------------------------------

class TestManifestAtomicUpdate(unittest.TestCase):
    def test_write_then_read_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "manifest.json")
            manifest = {"conditions": [{"condition_id": "layer00_kivi", "status": "pending"}]}
            driver.write_i2_manifest_atomic(path, manifest)
            reread = driver.read_i2_manifest(path)
            self.assertEqual(reread, manifest)

    def test_update_status_updates_correct_condition_only(self):
        manifest = {
            "conditions": [
                {"condition_id": "layer00_kivi", "status": "pending"},
                {"condition_id": "layer00_rotation_kivi", "status": "pending"},
            ]
        }
        driver.update_condition_status(manifest, "layer00_kivi", "running")
        self.assertEqual(manifest["conditions"][0]["status"], "running")
        self.assertEqual(manifest["conditions"][1]["status"], "pending")

    def test_update_unknown_condition_fails_closed(self):
        manifest = {"conditions": [{"condition_id": "layer00_kivi", "status": "pending"}]}
        with self.assertRaises(driver.I2ConfigError):
            driver.update_condition_status(manifest, "layer99_kivi", "running")

    def test_write_leaves_no_tmp_file_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "manifest.json")
            driver.write_i2_manifest_atomic(path, {"conditions": []})
            self.assertEqual(os.listdir(tmp), ["manifest.json"])


# --- no GPU on import / dry-run / tests ---------------------------------------

class TestNoGpuExecutionDuringImportOrDryRun(unittest.TestCase):
    def test_module_import_does_not_import_torch_in_a_fresh_process(self):
        # Run in a fresh subprocess (rather than checking sys.modules in
        # this test process) because other test modules sharing this
        # process may have already imported torch/transformers for
        # unrelated reasons -- a fresh interpreter is the only way to prove
        # importing these I2A modules alone never pulls either in.
        import subprocess

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        code = (
            "import sys; "
            "import utils.i2_layer_family_conditions; "
            "import importlib; "
            "sys.path.insert(0, 'scripts'); "
            "importlib.import_module('run_i2_layer_family_sensitivity'); "
            "assert 'torch' not in sys.modules, 'torch was imported'; "
            "assert 'transformers' not in sys.modules, 'transformers was imported'; "
            "print('OK')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("OK", result.stdout)

    def test_dry_run_generation_mode_does_not_newly_import_torch(self):
        modules_before = set(sys.modules)
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--dry-run", "--policies-dir", os.path.join(tmp, "policies"), "--output-root", os.path.join(tmp, "out")]
            with mock.patch.object(sys, "argv", ["run_i2_layer_family_sensitivity.py"] + argv):
                rc = driver.main()
            self.assertEqual(rc, 0)
        newly_imported = set(sys.modules) - modules_before
        self.assertNotIn("torch", newly_imported)
        self.assertNotIn("transformers", newly_imported)
        self.assertNotIn("datasets", newly_imported)

    def test_dry_run_canary_mode_does_not_newly_import_torch(self):
        modules_before = set(sys.modules)
        argv = ["--mode", "canary", "--dry-run"]
        with mock.patch.object(sys, "argv", ["run_i2_layer_family_sensitivity.py"] + argv):
            rc = driver.main()
        self.assertEqual(rc, 0)
        newly_imported = set(sys.modules) - modules_before
        self.assertNotIn("torch", newly_imported)

    def test_bare_run_flag_without_mode_refuses_not_executes(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--run", "--policies-dir", os.path.join(tmp, "policies"), "--output-root", os.path.join(tmp, "out")]
            with mock.patch.object(sys, "argv", ["run_i2_layer_family_sensitivity.py"] + argv):
                with self.assertRaises(driver.I2ConfigError):
                    driver.main()

    def test_canary_run_flag_reaches_preflight_not_a_bare_refusal(self):
        # Superseded by Stage I2B: --mode canary --run no longer refuses
        # unconditionally -- it now reaches i2b_preflight_checks(), which
        # itself refuses on a dirty tree/unknown HEAD/conflicting
        # process/busy GPU (see TestI2BPreflight) rather than always
        # raising I2CanaryError regardless of environment. This is a smoke
        # test that main() actually calls into i2b_preflight_checks (does
        # NOT assert a torch-free property -- run_i2b_canary does a
        # deferred `import torch`, guarded here by mocking preflight to
        # fail before that import is ever reached).
        argv = ["--mode", "canary", "--run"]
        with mock.patch.object(sys, "argv", ["run_i2_layer_family_sensitivity.py"] + argv):
            with mock.patch.object(driver, "i2b_preflight_checks", side_effect=driver.I2ConfigError("forced")) as mocked:
                with self.assertRaises(driver.I2ConfigError):
                    driver.main()
        mocked.assert_called_once()


# --- conflicting-process detection / single-instance lock (copied contract) --

class TestConflictingProcessDetection(unittest.TestCase):
    def test_ignores_wrapper_shell_mentioning_pattern(self):
        fake_output = (
            "12345 /bin/bash -c source snapshot.sh && eval 'echo run_i2_layer_family_sensitivity.py'\n"
        ).encode()
        with mock.patch("subprocess.check_output", return_value=fake_output):
            conflicts = driver.check_no_conflicting_process(self_pid=1)
        self.assertEqual(conflicts, [])

    def test_detects_genuine_python_invocation(self):
        fake_output = (
            "12345 ./.venv/bin/python scripts/run_i2_layer_family_sensitivity.py --run\n"
        ).encode()
        with mock.patch("subprocess.check_output", return_value=fake_output):
            conflicts = driver.check_no_conflicting_process(self_pid=1)
        self.assertEqual(len(conflicts), 1)

    def test_excludes_self_pid(self):
        fake_output = (
            "777 ./.venv/bin/python scripts/run_i2_layer_family_sensitivity.py --dry-run\n"
        ).encode()
        with mock.patch("subprocess.check_output", return_value=fake_output):
            conflicts = driver.check_no_conflicting_process(self_pid=777)
        self.assertEqual(conflicts, [])

    def test_no_pgrep_match_returns_empty(self):
        import subprocess as sp
        with mock.patch("subprocess.check_output", side_effect=sp.CalledProcessError(1, "pgrep")):
            conflicts = driver.check_no_conflicting_process(self_pid=1)
        self.assertEqual(conflicts, [])


class TestSingleInstanceLock(unittest.TestCase):
    def test_second_lock_attempt_fails_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = os.path.join(tmp, ".i2.lock")
            fh1 = driver.acquire_i2_lock(lock_path)
            try:
                with self.assertRaises(driver.I2LockError):
                    driver.acquire_i2_lock(lock_path)
            finally:
                driver.release_i2_lock(fh1)


# --- I2B canary configuration --------------------------------------------------

class TestI2BCanaryConfiguration(unittest.TestCase):
    def test_canary_layers_are_subset_of_i2_layers(self):
        for l in driver.CANARY_LAYERS:
            self.assertIn(l, I2_LAYERS)

    def test_canary_layers_exactly_0_18_31(self):
        self.assertEqual(driver.CANARY_LAYERS, (0, 18, 31))

    def test_canary_families_match_i2_families(self):
        self.assertEqual(tuple(driver.CANARY_FAMILIES), I2_FAMILIES)

    def test_canary_uses_k2_v16(self):
        self.assertEqual(driver.I2_K_BITS, 2)
        self.assertEqual(driver.I2_V_BITS, 16)

    def test_canary_max_new_tokens_exceeds_residual_length(self):
        # Deliberately long enough to observe a real Key-cache rollover
        # (bulk cadence ~= residual_length), unlike the short H2/C1 canaries.
        self.assertGreater(driver.CANARY_MAX_NEW_TOKENS, driver.I2_RESIDUAL_LENGTH)

    def test_six_canary_conditions_3_layers_x_2_families(self):
        conditions = driver.i2b_canary_conditions()
        self.assertEqual(len(conditions), 6)
        pairs = {(c["layer_idx"], c["family"]) for c in conditions}
        self.assertEqual(len(pairs), 6)

    def test_canary_conditions_reject_a_layer_outside_i2_layers(self):
        with self.assertRaises(driver.I2CanaryError):
            driver.i2b_canary_conditions(layers=(0, 99))

    def test_canary_dataset_index_is_an_already_fixed_calibration_sample(self):
        # Reuses the same already-vetted lcc calibration index the H2/I1
        # canaries use -- not a newly invented sample.
        self.assertEqual(driver.CANARY_DATASET_INDEX, 122)
        self.assertEqual(driver.CANARY_TASK, "lcc")


class TestI2BCanaryGate(unittest.TestCase):
    def _base_result(self, **overrides):
        result = {
            "layer_idx": 0, "family": "kivi",
            "policy_resolved_correctly": True, "quantized_prefix_observed": True,
            "cache_shapes_valid": True, "generation_finite": True, "hook_neutral": True,
            "ran_without_error": True,
        }
        result.update(overrides)
        return result

    def test_all_true_passes(self):
        gated = driver.compute_i2b_canary_gate(self._base_result())
        self.assertTrue(gated["condition_pass"])

    def test_missing_quantized_prefix_fails_not_silently_passed(self):
        gated = driver.compute_i2b_canary_gate(self._base_result(quantized_prefix_observed=False))
        self.assertFalse(gated["condition_pass"])
        self.assertFalse(gated["quantized_prefix_observed"])

    def test_any_single_false_criterion_fails_the_condition(self):
        for key in ("policy_resolved_correctly", "cache_shapes_valid", "generation_finite", "hook_neutral", "ran_without_error"):
            gated = driver.compute_i2b_canary_gate(self._base_result(**{key: False}))
            self.assertFalse(gated["condition_pass"], f"expected condition_pass=False when {key}=False")


# --- I2B: quantized-Key cache structural shape validation ---------------------

def _fake_tensor(*shape):
    return SimpleNamespace(shape=shape)


def _cache_state(**kwargs):
    fields = [
        "key_quant_trans", "key_full", "key_scale_trans", "key_mn_trans",
        "value_quant", "value_full", "value_scale", "value_mn", "kv_seq_len",
    ]
    defaults = {f: None for f in fields}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestQuantizedKeyCacheShapeValidation(unittest.TestCase):
    """Section 9: must not merely check non-None -- must verify structural
    (leading-dimension) agreement per the real
    triton_quantize_and_pack_along_last_dim contract, and that every
    present tensor has strictly positive dimensions."""

    def test_no_quantized_prefix_yet_is_trivially_valid(self):
        self.assertTrue(driver._validate_quantized_key_cache_shapes(_cache_state()))

    def test_matching_leading_dims_across_quant_scale_mn_is_valid(self):
        state = _cache_state(
            key_quant_trans=_fake_tensor(1, 32, 128, 8),
            key_scale_trans=_fake_tensor(1, 32, 128, 4),
            key_mn_trans=_fake_tensor(1, 32, 128, 4),
        )
        self.assertTrue(driver._validate_quantized_key_cache_shapes(state))

    def test_mismatched_leading_dims_is_invalid(self):
        state = _cache_state(
            key_quant_trans=_fake_tensor(1, 32, 128, 8),
            key_scale_trans=_fake_tensor(1, 16, 128, 4),  # wrong num_heads vs. quant_trans
            key_mn_trans=_fake_tensor(1, 32, 128, 4),
        )
        self.assertFalse(driver._validate_quantized_key_cache_shapes(state))

    def test_quantized_prefix_without_scale_is_invalid(self):
        state = _cache_state(key_quant_trans=_fake_tensor(1, 32, 128, 8), key_mn_trans=_fake_tensor(1, 32, 128, 4))
        self.assertFalse(driver._validate_quantized_key_cache_shapes(state))

    def test_quantized_prefix_without_mn_is_invalid(self):
        state = _cache_state(key_quant_trans=_fake_tensor(1, 32, 128, 8), key_scale_trans=_fake_tensor(1, 32, 128, 4))
        self.assertFalse(driver._validate_quantized_key_cache_shapes(state))

    def test_zero_length_dimension_is_invalid(self):
        state = _cache_state(
            key_quant_trans=_fake_tensor(1, 32, 128, 0),
            key_scale_trans=_fake_tensor(1, 32, 128, 4),
            key_mn_trans=_fake_tensor(1, 32, 128, 4),
        )
        self.assertFalse(driver._validate_quantized_key_cache_shapes(state))

    def test_key_full_none_is_valid_when_prefill_exactly_fills_quantized_prefix(self):
        # models/llama_kivi.py's prefill branch legitimately sets
        # key_states_full = None when the prefill length is an exact
        # multiple of residual_length -- requiring key_full unconditionally
        # would invent a stricter cache format than production guarantees.
        state = _cache_state(
            key_quant_trans=_fake_tensor(1, 32, 128, 8),
            key_scale_trans=_fake_tensor(1, 32, 128, 4),
            key_mn_trans=_fake_tensor(1, 32, 128, 4),
            key_full=None,
        )
        self.assertTrue(driver._validate_quantized_key_cache_shapes(state))

    def test_key_full_present_but_malformed_is_invalid(self):
        state = _cache_state(
            key_quant_trans=_fake_tensor(1, 32, 128, 8),
            key_scale_trans=_fake_tensor(1, 32, 128, 4),
            key_mn_trans=_fake_tensor(1, 32, 128, 4),
            key_full=_fake_tensor(1, 32, 0, 128),  # zero-length residual dim
        )
        self.assertFalse(driver._validate_quantized_key_cache_shapes(state))


# --- I2B: finite-check hook control logic (Section 12) -------------------------

class TestFiniteCheckHookControlLogic(unittest.TestCase):
    def test_finite_output_leaves_flag_false(self):
        import torch

        hook, flags = driver._make_finite_check_hook()
        hook(module=None, inp=None, output=torch.tensor([1.0, 2.0, 3.0]))
        self.assertFalse(flags["seen_nonfinite"])

    def test_nan_output_flips_flag_true(self):
        import torch

        hook, flags = driver._make_finite_check_hook()
        hook(module=None, inp=None, output=torch.tensor([1.0, float("nan")]))
        self.assertTrue(flags["seen_nonfinite"])

    def test_inf_output_flips_flag_true(self):
        import torch

        hook, flags = driver._make_finite_check_hook()
        hook(module=None, inp=None, output=torch.tensor([1.0, float("inf")]))
        self.assertTrue(flags["seen_nonfinite"])

    def test_flag_latches_true_across_subsequent_finite_calls(self):
        import torch

        hook, flags = driver._make_finite_check_hook()
        hook(module=None, inp=None, output=torch.tensor([float("nan")]))
        hook(module=None, inp=None, output=torch.tensor([1.0]))  # a later, perfectly finite call
        self.assertTrue(flags["seen_nonfinite"])  # must not un-latch

    def test_hook_is_neutral_does_not_modify_real_module_output(self):
        """Proves the hook cannot bias hook-neutrality: registering it on a
        real nn.Module must leave that module's returned output tensor
        byte-identical to the unhooked call."""
        import torch
        import torch.nn as nn

        torch.manual_seed(0)
        layer = nn.Linear(4, 4)
        x = torch.randn(1, 4)
        baseline = layer(x).clone()

        hook, flags = driver._make_finite_check_hook()
        handle = layer.register_forward_hook(hook)
        try:
            hooked = layer(x)
        finally:
            handle.remove()

        torch.testing.assert_close(hooked, baseline)
        self.assertFalse(flags["seen_nonfinite"])

    def test_independent_flags_per_hook_instance(self):
        import torch

        hook_a, flags_a = driver._make_finite_check_hook()
        hook_b, flags_b = driver._make_finite_check_hook()
        hook_a(module=None, inp=None, output=torch.tensor([float("nan")]))
        self.assertTrue(flags_a["seen_nonfinite"])
        self.assertFalse(flags_b["seen_nonfinite"])


# --- I2B: preflight (Section 5) -------------------------------------------------

class TestI2BPreflight(unittest.TestCase):
    def test_dirty_git_tree_refuses(self):
        with mock.patch.object(driver, "get_git_status_short", return_value=" M some_file.py\n"):
            with self.assertRaises(driver.I2ConfigError):
                driver.i2b_preflight_checks()

    def test_unknown_git_status_refuses(self):
        with mock.patch.object(driver, "get_git_status_short", return_value=None):
            with self.assertRaises(driver.I2ConfigError):
                driver.i2b_preflight_checks()

    def test_unknown_head_refuses(self):
        with mock.patch.object(driver, "get_git_status_short", return_value=""):
            with mock.patch.object(driver, "get_git_commit", return_value=None):
                with self.assertRaises(driver.I2ConfigError):
                    driver.i2b_preflight_checks()

    def test_conflicting_process_refuses(self):
        with mock.patch.object(driver, "get_git_status_short", return_value=""):
            with mock.patch.object(driver, "get_git_commit", return_value="deadbeef"):
                with mock.patch.object(driver, "check_no_conflicting_process", return_value=["12345 python foo.py"]):
                    with self.assertRaises(driver.I2ConflictError):
                        driver.i2b_preflight_checks()

    def test_gpu_busy_refuses(self):
        with mock.patch.object(driver, "get_git_status_short", return_value=""):
            with mock.patch.object(driver, "get_git_commit", return_value="deadbeef"):
                with mock.patch.object(driver, "check_no_conflicting_process", return_value=[]):
                    with mock.patch.object(driver, "gpu_preflight", return_value={"checked": True, "warning": "busy"}):
                        with self.assertRaises(driver.I2ConfigError):
                            driver.i2b_preflight_checks()

    def test_all_clear_passes_and_returns_head_and_preflight(self):
        with mock.patch.object(driver, "get_git_status_short", return_value=""):
            with mock.patch.object(driver, "get_git_commit", return_value="deadbeef"):
                with mock.patch.object(driver, "check_no_conflicting_process", return_value=[]):
                    with mock.patch.object(driver, "gpu_preflight", return_value={"checked": True, "warning": None}):
                        head, preflight = driver.i2b_preflight_checks()
        self.assertEqual(head, "deadbeef")
        self.assertEqual(preflight["warning"], None)


# --- I2B: CLI wiring / orchestration (mocked GPU) -------------------------------

class _FakeCuda:
    def is_available(self):
        return True

    def get_device_name(self, i=0):
        return "FakeGPU"

    def empty_cache(self):
        pass


class _FakeTorch:
    def __init__(self):
        self.cuda = _FakeCuda()


def _make_condition_result(layer_idx, family, **overrides):
    result = {
        "layer_idx": layer_idx, "family": family,
        "policy_resolved_correctly": True, "quantized_prefix_observed": True,
        "cache_shapes_valid": True, "generation_finite": True, "hook_neutral": True,
        "ran_without_error": True, "error": None, "generated_token_count": 140,
        "cache_observations": [],
    }
    result.update(overrides)
    return result


class TestI2BCliWiring(unittest.TestCase):
    def test_mode_canary_run_reaches_run_i2b_canary(self):
        argv = ["--mode", "canary", "--run"]
        with mock.patch.object(sys, "argv", ["run_i2_layer_family_sensitivity.py"] + argv):
            with mock.patch.object(driver, "run_i2b_canary", return_value=0) as mocked:
                rc = driver.main()
        mocked.assert_called_once()
        self.assertEqual(rc, 0)

    def test_mode_generation_run_still_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--run", "--policies-dir", os.path.join(tmp, "policies"), "--output-root", os.path.join(tmp, "out")]
            with mock.patch.object(sys, "argv", ["run_i2_layer_family_sensitivity.py"] + argv):
                with mock.patch.object(driver, "run_i2b_canary") as mocked:
                    with self.assertRaises(driver.I2ConfigError):
                        driver.main()
            mocked.assert_not_called()

    def test_mode_generation_run_explicit_still_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--mode", "generation", "--run", "--policies-dir", os.path.join(tmp, "policies"), "--output-root", os.path.join(tmp, "out")]
            with mock.patch.object(sys, "argv", ["run_i2_layer_family_sensitivity.py"] + argv):
                with self.assertRaises(driver.I2ConfigError):
                    driver.main()

    def test_canary_dry_run_never_calls_run_i2b_canary(self):
        argv = ["--mode", "canary", "--dry-run"]
        with mock.patch.object(sys, "argv", ["run_i2_layer_family_sensitivity.py"] + argv):
            with mock.patch.object(driver, "run_i2b_canary") as mocked:
                rc = driver.main()
        mocked.assert_not_called()
        self.assertEqual(rc, 0)

    def test_canary_mode_default_no_run_flag_never_calls_run_i2b_canary(self):
        argv = ["--mode", "canary"]
        with mock.patch.object(sys, "argv", ["run_i2_layer_family_sensitivity.py"] + argv):
            with mock.patch.object(driver, "run_i2b_canary") as mocked:
                rc = driver.main()
        mocked.assert_not_called()
        self.assertEqual(rc, 0)


class TestI2BOrchestration(unittest.TestCase):
    def _run(self, tmp, condition_side_effect):
        args = argparse.Namespace(model_name_or_path="fake-model", cache_dir="fake-cache")
        fake_torch = _FakeTorch()
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            with mock.patch.object(driver, "i2b_preflight_checks", return_value=("deadbeefcafe", {"checked": True, "warning": None})):
                with mock.patch.object(driver, "I2B_OUTPUT_ROOT", tmp):
                    with mock.patch.object(driver, "run_i2b_canary_condition", side_effect=condition_side_effect):
                        rc = driver.run_i2b_canary(args)
        return rc

    def test_exactly_six_conditions_executed_in_deterministic_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected_order = [(c["layer_idx"], c["family"]) for c in driver.i2b_canary_conditions()]
            seen_order = []

            def fake_condition(layer_idx, family, model_name_or_path, cache_dir):
                seen_order.append((layer_idx, family))
                return _make_condition_result(layer_idx, family)

            self._run(tmp, fake_condition)
            self.assertEqual(seen_order, expected_order)
            self.assertEqual(len(seen_order), 6)

    def test_all_six_pass_gives_exit_0_and_overall_pass_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            conditions = driver.i2b_canary_conditions()
            side_effect = [_make_condition_result(c["layer_idx"], c["family"]) for c in conditions]
            rc = self._run(tmp, side_effect)
            self.assertEqual(rc, 0)

            run_dirs = os.listdir(tmp)
            self.assertEqual(len(run_dirs), 1)
            run_dir = os.path.join(tmp, run_dirs[0])
            with open(os.path.join(run_dir, "canary_summary.json")) as f:
                summary = json.load(f)
            self.assertTrue(summary["overall_pass"])
            self.assertEqual(len(summary["conditions"]), 6)
            self.assertTrue(os.path.exists(os.path.join(run_dir, "canary.lock")))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "host_monitor.log")))

    def test_one_failed_condition_fails_the_aggregate_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            conditions = driver.i2b_canary_conditions()
            side_effect = [_make_condition_result(c["layer_idx"], c["family"]) for c in conditions]
            side_effect[2]["quantized_prefix_observed"] = False  # load-bearing criterion fails for one condition
            rc = self._run(tmp, side_effect)
            run_dirs = os.listdir(tmp)
            with open(os.path.join(tmp, run_dirs[0], "canary_summary.json")) as f:
                summary = json.load(f)
            self.assertFalse(summary["overall_pass"])
            passes = [c["condition_pass"] for c in summary["conditions"]]
            self.assertEqual(passes.count(True), 5)
            self.assertEqual(passes.count(False), 1)

    def test_provenance_fields_present_in_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            conditions = driver.i2b_canary_conditions()
            side_effect = [_make_condition_result(c["layer_idx"], c["family"]) for c in conditions]
            self._run(tmp, side_effect)
            run_dirs = os.listdir(tmp)
            with open(os.path.join(tmp, run_dirs[0], "canary_summary.json")) as f:
                summary = json.load(f)
            for field in (
                "collection_head", "boot_id_start", "boot_id_end", "boot_id_stable",
                "gpu_before", "gpu_after", "cuda_device", "triton_ptxas_path",
                "canary_task", "canary_dataset_index", "canary_seed",
                "group_size", "residual_length", "max_new_tokens", "exit_code",
            ):
                self.assertIn(field, summary, f"missing provenance field {field!r}")

    def test_output_isolated_under_canary_subtree(self):
        self.assertTrue(driver.I2B_OUTPUT_ROOT.endswith(os.path.join("i2_layer_family_sensitivity", "canary")))
        self.assertNotIn("i1_rotation_kivi_canary", driver.I2B_OUTPUT_ROOT)
        self.assertNotIn("layer_sensitivity_pilot", driver.I2B_OUTPUT_ROOT)
        self.assertNotIn("layer_attention_feature_pilot", driver.I2B_OUTPUT_ROOT)


# --- I2C0: pinned dataset revision ---------------------------------------------

class TestFormalDatasetRevision(unittest.TestCase):
    def test_exact_revision_constant(self):
        self.assertEqual(driver.I2_DATASET_REVISION, "5e628be450b7e67fb7ae6e201bd6d8f7056f7672")

    def test_run_condition_defaults_to_pinned_revision(self):
        sig = inspect.signature(driver.run_condition)
        self.assertEqual(sig.parameters["dataset_revision"].default, driver.I2_DATASET_REVISION)

    def test_run_condition_source_passes_revision_to_load_dataset(self):
        # run_condition is a GPU path (deferred torch/datasets/pred_long_bench
        # imports) that cannot be exercised without a real model; verify by
        # source inspection that its load_dataset() call forwards the
        # dataset_revision parameter rather than omitting it.
        source = inspect.getsource(driver.run_condition)
        self.assertIn('load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True, revision=dataset_revision)', source)

    def test_run_config_includes_dataset_revision(self):
        cond = {
            "condition_id": "layer00_kivi", "layer_idx": 0, "family": "kivi", "k_bits": 2, "v_bits": 16,
            "policy_path": "p.json", "resolved_policy_hash": "abc123",
        }
        run_config = driver.build_condition_run_config(cond, "lmsys/longchat-7b-v1.5-32k", 31500, 32, 128, 42)
        self.assertEqual(run_config["dataset_revision"], driver.I2_DATASET_REVISION)

    def test_dataset_revision_is_a_resume_core_key(self):
        self.assertIn("dataset_revision", driver.I2_CONDITION_CORE_KEYS)

    def test_resume_rejects_different_dataset_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            condition_dir = os.path.join(tmp, "cond")
            cond = {
                "condition_id": "layer00_kivi", "layer_idx": 0, "family": "kivi", "k_bits": 2, "v_bits": 16,
                "policy_path": "p.json", "resolved_policy_hash": "abc123",
            }
            rc_a = driver.build_condition_run_config(cond, "lmsys/longchat-7b-v1.5-32k", 31500, 32, 128, 42, dataset_revision="oldrev")
            rc_b = driver.build_condition_run_config(cond, "lmsys/longchat-7b-v1.5-32k", 31500, 32, 128, 42, dataset_revision="newrev")
            driver.prepare_condition_directory(condition_dir, rc_a)
            with self.assertRaises(driver.I2ConfigError):
                driver.prepare_condition_directory(condition_dir, rc_b)

    def test_resume_rejects_missing_dataset_revision_in_legacy_run_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            condition_dir = os.path.join(tmp, "cond")
            os.makedirs(condition_dir)
            legacy_run_config = {
                "model_name_or_path": "lmsys/longchat-7b-v1.5-32k", "condition_id": "layer00_kivi",
                "layer_idx": 0, "family": "kivi", "k_bits": 2, "v_bits": 16, "policy_path": "p.json",
                "policy_hash": "abc123", "max_length": 31500, "group_size": 32, "residual_length": 128, "seed": 42,
                # deliberately no "dataset_revision" key -- a pre-I2C0 legacy run_config.json
            }
            with open(os.path.join(condition_dir, "run_config.json"), "w", encoding="utf-8") as f:
                json.dump(legacy_run_config, f)
            cond = {
                "condition_id": "layer00_kivi", "layer_idx": 0, "family": "kivi", "k_bits": 2, "v_bits": 16,
                "policy_path": "p.json", "resolved_policy_hash": "abc123",
            }
            new_run_config = driver.build_condition_run_config(cond, "lmsys/longchat-7b-v1.5-32k", 31500, 32, 128, 42)
            with self.assertRaises(driver.I2ConfigError):
                driver.prepare_condition_directory(condition_dir, new_run_config)


# --- I2C0: fail-closed dataset identity preflight -------------------------------

class _FakeLongBenchDataset(list):
    pass


def _fake_datasets_module(rows_by_task, raise_for_task=None):
    module = SimpleNamespace()

    def _load_dataset(repo, task, split=None, trust_remote_code=None, revision=None):
        if raise_for_task is not None and task == raise_for_task:
            raise RuntimeError(f"could not resolve revision {revision!r} for task {task!r}")
        return _FakeLongBenchDataset(range(rows_by_task[task]))

    module.load_dataset = _load_dataset
    return module


class TestDatasetIdentityPreflight(unittest.TestCase):
    def test_matching_counts_passes_and_reports_all_six_tasks(self):
        fake_module = _fake_datasets_module(dict(driver.I2_TASK_COUNTS))
        with mock.patch.dict(sys.modules, {"datasets": fake_module}):
            report = driver.i2c_dataset_identity_preflight()
        self.assertEqual(len(report), 6)
        for row in report:
            self.assertEqual(row["requested_revision"], driver.I2_DATASET_REVISION)
            self.assertEqual(row["available_row_count"], row["planned_row_count"])

    def test_mismatched_count_fails_closed(self):
        counts = dict(driver.I2_TASK_COUNTS)
        counts["trec"] = counts["trec"] - 1  # off by one
        fake_module = _fake_datasets_module(counts)
        with mock.patch.dict(sys.modules, {"datasets": fake_module}):
            with self.assertRaises(driver.I2ConfigError):
                driver.i2c_dataset_identity_preflight()

    def test_unresolvable_revision_fails_closed_never_falls_back(self):
        fake_module = _fake_datasets_module(dict(driver.I2_TASK_COUNTS), raise_for_task="lcc")
        with mock.patch.dict(sys.modules, {"datasets": fake_module}):
            with self.assertRaises(driver.I2ConfigError):
                driver.i2c_dataset_identity_preflight()


# --- I2C0: formal argument lock --------------------------------------------------

def _formal_args(**overrides):
    base = dict(
        model_name_or_path="lmsys/longchat-7b-v1.5-32k",
        cache_dir="./cached_models",
        mode="generation",
        layers=list(driver.I2_LAYERS),
        families=list(driver.I2_FAMILIES),
        dry_run=False,
        run=True,
        device=0,
        max_length=31500,
        group_size=driver.I2_GROUP_SIZE,
        residual_length=driver.I2_RESIDUAL_LENGTH,
        seed=42,
        output_root=driver.DEFAULT_OUTPUT_ROOT,
        policies_dir=driver.DEFAULT_POLICIES_DIR,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestFormalArgumentLock(unittest.TestCase):
    def test_exact_universe_passes(self):
        driver.validate_formal_i2c_arguments(_formal_args())  # must not raise

    def test_reversed_layer_order_still_passes_normalized(self):
        driver.validate_formal_i2c_arguments(_formal_args(layers=list(reversed(driver.I2_LAYERS))))

    def test_wrong_model_fails(self):
        with self.assertRaises(driver.I2ConfigError):
            driver.validate_formal_i2c_arguments(_formal_args(model_name_or_path="some-other-model"))

    def test_extra_layer_fails(self):
        with self.assertRaises(driver.I2ConfigError):
            driver.validate_formal_i2c_arguments(_formal_args(layers=list(driver.I2_LAYERS) + [5]))

    def test_missing_layer_fails(self):
        with self.assertRaises(driver.I2ConfigError):
            driver.validate_formal_i2c_arguments(_formal_args(layers=list(driver.I2_LAYERS)[:-1]))

    def test_subset_layers_does_not_masquerade_as_i2c(self):
        with self.assertRaises(driver.I2ConfigError):
            driver.validate_formal_i2c_arguments(_formal_args(layers=[0, 18, 31]))  # e.g. the I2B canary's layer subset

    def test_single_family_fails(self):
        with self.assertRaises(driver.I2ConfigError):
            driver.validate_formal_i2c_arguments(_formal_args(families=["kivi"]))

    def test_wrong_max_length_fails(self):
        with self.assertRaises(driver.I2ConfigError):
            driver.validate_formal_i2c_arguments(_formal_args(max_length=8192))

    def test_wrong_group_size_fails(self):
        with self.assertRaises(driver.I2ConfigError):
            driver.validate_formal_i2c_arguments(_formal_args(group_size=64))

    def test_wrong_residual_length_fails(self):
        with self.assertRaises(driver.I2ConfigError):
            driver.validate_formal_i2c_arguments(_formal_args(residual_length=256))

    def test_wrong_seed_fails(self):
        with self.assertRaises(driver.I2ConfigError):
            driver.validate_formal_i2c_arguments(_formal_args(seed=0))

    def test_wrong_output_root_fails(self):
        with self.assertRaises(driver.I2ConfigError):
            driver.validate_formal_i2c_arguments(_formal_args(output_root="/tmp/not_the_formal_root"))

    def test_wrong_policies_dir_fails(self):
        with self.assertRaises(driver.I2ConfigError):
            driver.validate_formal_i2c_arguments(_formal_args(policies_dir="/tmp/not_the_formal_policies_dir"))


# --- I2C0: current-HEAD I2B prerequisite -----------------------------------------

def _write_canary_summary(canary_root, run_label, **fields):
    d = os.path.join(canary_root, run_label)
    os.makedirs(d, exist_ok=True)
    summary = {"collection_head": "headA", "overall_pass": True, "conditions": [{"condition_pass": True} for _ in range(6)]}
    summary.update(fields)
    with open(os.path.join(d, "canary_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f)
    return d


class TestI2BPrerequisiteForI2C(unittest.TestCase):
    def test_no_canary_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "does_not_exist")
            with self.assertRaises(driver.I2ConfigError):
                driver.find_passing_i2b_canary_for_head("HEAD1", canary_root=missing)

    def test_empty_canary_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(driver.I2ConfigError):
                driver.find_passing_i2b_canary_for_head("HEAD1", canary_root=tmp)

    def test_stale_head_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_canary_summary(tmp, "20260101T000000000000Z", collection_head="OLD_HEAD")
            with self.assertRaises(driver.I2ConfigError):
                driver.find_passing_i2b_canary_for_head("NEW_HEAD", canary_root=tmp)

    def test_five_of_six_conditions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            conditions = [{"condition_pass": True}] * 5 + [{"condition_pass": False}]
            _write_canary_summary(tmp, "20260101T000000000000Z", collection_head="HEAD1", conditions=conditions, overall_pass=False)
            with self.assertRaises(driver.I2ConfigError):
                driver.find_passing_i2b_canary_for_head("HEAD1", canary_root=tmp)

    def test_overall_pass_false_rejected_even_with_six_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_canary_summary(tmp, "20260101T000000000000Z", collection_head="HEAD1", overall_pass=False)
            with self.assertRaises(driver.I2ConfigError):
                driver.find_passing_i2b_canary_for_head("HEAD1", canary_root=tmp)

    def test_six_of_six_current_head_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_canary_summary(tmp, "20260101T000000000000Z", collection_head="HEAD1")
            result = driver.find_passing_i2b_canary_for_head("HEAD1", canary_root=tmp)
            self.assertTrue(result["overall_pass"])
            self.assertEqual(len(result["conditions"]), 6)

    def test_returns_most_recent_matching_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_canary_summary(tmp, "20260101T000000000000Z", collection_head="HEAD1", marker="old")
            _write_canary_summary(tmp, "20260102T000000000000Z", collection_head="HEAD1", marker="new")
            result = driver.find_passing_i2b_canary_for_head("HEAD1", canary_root=tmp)
            self.assertEqual(result["marker"], "new")

    def test_unreadable_summary_json_is_skipped_not_crashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_dir = os.path.join(tmp, "20260101T000000000000Z")
            os.makedirs(bad_dir)
            with open(os.path.join(bad_dir, "canary_summary.json"), "w", encoding="utf-8") as f:
                f.write("{not valid json")
            with self.assertRaises(driver.I2ConfigError):
                driver.find_passing_i2b_canary_for_head("HEAD1", canary_root=tmp)


# --- I2C0: top-level formal manifest ---------------------------------------------

class TestI2CManifest(unittest.TestCase):
    def _conditions(self):
        return [
            {"condition_id": "layer00_kivi", "layer_idx": 0, "family": "kivi", "k_bits": 2, "v_bits": 16,
             "policy_path": "p0.json", "resolved_policy_hash": "hash0", "output_dir": "/tmp/out0"},
            {"condition_id": "layer00_rotation_kivi", "layer_idx": 0, "family": "rotation_kivi", "k_bits": 2, "v_bits": 16,
             "policy_path": "p1.json", "resolved_policy_hash": "hash1", "output_dir": "/tmp/out1"},
        ]

    def test_manifest_records_required_provenance_fields(self):
        manifest = driver.build_initial_i2c_manifest(
            self._conditions(), _formal_args(), "deadbeefcafe", {"checked": True, "warning": None}, [{"task": "trec"}]
        )
        for field in (
            "experiment", "collection_head", "dataset_revision", "dataset_identity_preflight",
            "model_name_or_path", "seed", "max_length", "group_size", "residual_length",
            "tasks", "expected_task_counts", "total_planned_rows", "num_conditions",
            "boot_id_start", "gpu_preflight", "triton_ptxas_path", "start_time", "last_update_time", "conditions",
        ):
            self.assertIn(field, manifest, f"missing manifest field {field!r}")
        self.assertEqual(manifest["dataset_revision"], driver.I2_DATASET_REVISION)
        self.assertEqual(len(manifest["conditions"]), 2)
        self.assertTrue(all(c["status"] == "pending" for c in manifest["conditions"]))

    def test_record_condition_result_updates_correct_condition_only(self):
        manifest = driver.build_initial_i2c_manifest(
            self._conditions(), _formal_args(), "deadbeefcafe", {"checked": True, "warning": None}, []
        )
        driver.record_i2c_condition_result(manifest, "layer00_kivi", status="running")
        statuses = {c["condition_id"]: c["status"] for c in manifest["conditions"]}
        self.assertEqual(statuses["layer00_kivi"], "running")
        self.assertEqual(statuses["layer00_rotation_kivi"], "pending")

    def test_record_condition_result_bumps_last_update_time(self):
        manifest = driver.build_initial_i2c_manifest(
            self._conditions(), _formal_args(), "deadbeefcafe", {"checked": True, "warning": None}, []
        )
        original = manifest["last_update_time"]
        import time

        time.sleep(0.01)
        driver.record_i2c_condition_result(manifest, "layer00_kivi", status="complete", exit_code=0, nonfinite_logits_seen=False)
        self.assertNotEqual(manifest["last_update_time"], original)
        updated = next(c for c in manifest["conditions"] if c["condition_id"] == "layer00_kivi")
        self.assertEqual(updated["exit_code"], 0)
        self.assertEqual(updated["nonfinite_logits_seen"], False)

    def test_record_unknown_condition_fails_closed(self):
        manifest = driver.build_initial_i2c_manifest(
            self._conditions(), _formal_args(), "deadbeefcafe", {"checked": True, "warning": None}, []
        )
        with self.assertRaises(driver.I2ConfigError):
            driver.record_i2c_condition_result(manifest, "layer99_kivi", status="running")


# --- I2C0: condition-completion validation (Section 14) -------------------------

class TestConditionCompletionValidation(unittest.TestCase):
    def _write_complete_condition(self, condition_dir):
        os.makedirs(condition_dir, exist_ok=True)
        for task, n in driver.I2_TASK_COUNTS.items():
            _write_jsonl(os.path.join(condition_dir, f"{task}.jsonl"), [{"dataset_index": i, "pred": "p"} for i in range(n)])
        with open(os.path.join(condition_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"exit_code": 0, "nonfinite_logits_seen": False}, f)

    def test_complete_condition_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_complete_condition(tmp)
            is_complete, report = driver.validate_condition_completion(tmp)
            self.assertTrue(is_complete)
            self.assertEqual(report["problems"], [])
            self.assertEqual(report["total_rows"], sum(driver.I2_TASK_COUNTS.values()))

    def test_missing_task_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_complete_condition(tmp)
            os.remove(os.path.join(tmp, "samsum.jsonl"))
            is_complete, report = driver.validate_condition_completion(tmp)
            self.assertFalse(is_complete)

    def test_wrong_row_count_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_complete_condition(tmp)
            _write_jsonl(os.path.join(tmp, "trec.jsonl"), [{"dataset_index": i, "pred": "p"} for i in range(199)])  # one short
            is_complete, report = driver.validate_condition_completion(tmp)
            self.assertFalse(is_complete)

    def test_leftover_partial_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_complete_condition(tmp)
            _write_jsonl(os.path.join(tmp, "trec.jsonl.partial"), [{"dataset_index": 0, "pred": "p"}])
            is_complete, report = driver.validate_condition_completion(tmp)
            self.assertFalse(is_complete)

    def test_missing_condition_manifest_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_complete_condition(tmp)
            os.remove(os.path.join(tmp, "manifest.json"))
            is_complete, report = driver.validate_condition_completion(tmp)
            self.assertFalse(is_complete)

    def test_nonzero_exit_code_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_complete_condition(tmp)
            with open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump({"exit_code": 1, "nonfinite_logits_seen": False}, f)
            is_complete, report = driver.validate_condition_completion(tmp)
            self.assertFalse(is_complete)

    def test_nonfinite_true_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_complete_condition(tmp)
            with open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump({"exit_code": 0, "nonfinite_logits_seen": True}, f)
            is_complete, report = driver.validate_condition_completion(tmp)
            self.assertFalse(is_complete)


class TestAggregateCompletion(unittest.TestCase):
    def _write_complete_condition(self, condition_dir):
        os.makedirs(condition_dir, exist_ok=True)
        for task, n in driver.I2_TASK_COUNTS.items():
            _write_jsonl(os.path.join(condition_dir, f"{task}.jsonl"), [{"dataset_index": i, "pred": "p"} for i in range(n)])
        with open(os.path.join(condition_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"exit_code": 0, "nonfinite_logits_seen": False}, f)

    def _conditions(self, n=16):
        specs = driver.i2b_canary_conditions(layers=driver.I2_LAYERS, families=driver.I2_FAMILIES)[:n]
        return [{"condition_id": f"cond{i}", "resolved_policy_hash": f"h{i}", **c} for i, c in enumerate(specs)]

    def test_all_16_complete_gives_23200_rows_and_all_complete_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            conditions = self._conditions()
            for c in conditions:
                self._write_complete_condition(os.path.join(tmp, driver.output_dir_name(c)))
            result = driver.validate_formal_generation_complete(conditions, tmp)
            self.assertTrue(result["all_complete"])
            self.assertEqual(result["rows_total"], 23200)
            self.assertEqual(result["rows_complete"], 23200)
            self.assertEqual(result["conditions_complete"], 16)
            self.assertEqual(result["task_files_complete"], 96)

    def test_one_incomplete_condition_fails_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            conditions = self._conditions()
            for c in conditions:
                self._write_complete_condition(os.path.join(tmp, driver.output_dir_name(c)))
            # Break one condition's samsum.jsonl.
            broken_dir = os.path.join(tmp, driver.output_dir_name(conditions[5]))
            os.remove(os.path.join(broken_dir, "samsum.jsonl"))
            result = driver.validate_formal_generation_complete(conditions, tmp)
            self.assertFalse(result["all_complete"])
            self.assertEqual(result["conditions_complete"], 15)


# --- I2C0: CLI wiring / orchestration (mocked GPU) -------------------------------

class TestI2CCliWiring(unittest.TestCase):
    def test_mode_generation_run_reaches_i2c_orchestration(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(driver, "DEFAULT_OUTPUT_ROOT", tmp):
                argv = ["--run"]
                with mock.patch.object(sys, "argv", ["run_i2_layer_family_sensitivity.py"] + argv):
                    with mock.patch.object(driver, "run_i2c_formal_generation", return_value=0) as mocked:
                        rc = driver.main()
            mocked.assert_called_once()
            self.assertEqual(rc, 0)

    def test_canary_mode_never_calls_i2c_orchestration(self):
        argv = ["--mode", "canary", "--run"]
        with mock.patch.object(sys, "argv", ["run_i2_layer_family_sensitivity.py"] + argv):
            with mock.patch.object(driver, "i2b_preflight_checks", side_effect=driver.I2ConfigError("forced")):
                with mock.patch.object(driver, "run_i2c_formal_generation") as mocked_i2c:
                    with self.assertRaises(driver.I2ConfigError):
                        driver.main()
        mocked_i2c.assert_not_called()


class TestI2CFormalOrchestration(unittest.TestCase):
    def _complete_report(self):
        tasks = {
            t: {"exists": True, "valid_rows": n, "invalid_rows": 0, "expected": n, "leftover_partial": False, "ok": True}
            for t, n in driver.I2_TASK_COUNTS.items()
        }
        return True, {"tasks": tasks, "problems": [], "exit_code": 0, "nonfinite_logits_seen": False,
                       "total_rows": sum(driver.I2_TASK_COUNTS.values())}

    def _run(self, tmp, run_condition_side_effect):
        tmp_out = os.path.join(tmp, "out")
        tmp_policies = os.path.join(tmp, "policies")
        args = _formal_args(output_root=tmp_out, policies_dir=tmp_policies)
        fake_torch = _FakeTorch()
        canary_summary = {"overall_pass": True, "collection_head": "deadbeefcafe", "conditions": [{"condition_pass": True}] * 6}
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            with mock.patch.object(driver, "DEFAULT_OUTPUT_ROOT", tmp_out):
                with mock.patch.object(driver, "DEFAULT_POLICIES_DIR", tmp_policies):
                    with mock.patch.object(driver, "i2b_preflight_checks", return_value=("deadbeefcafe", {"checked": True, "warning": None})):
                        with mock.patch.object(driver, "i2c_dataset_identity_preflight", return_value=[{"task": "trec"}]):
                            with mock.patch.object(driver, "find_passing_i2b_canary_for_head", return_value=canary_summary):
                                with mock.patch.object(driver, "run_condition", side_effect=run_condition_side_effect):
                                    with mock.patch.object(driver, "validate_condition_completion", side_effect=lambda *a, **k: self._complete_report()):
                                        rc = driver.run_i2c_formal_generation(args)
        return rc, tmp_out

    def test_exactly_16_conditions_in_deterministic_layer_then_family_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen_order = []

            def fake_run_condition(condition, args, dataset2prompt, dataset2maxlen, task_counts=None, dataset_revision=None):
                seen_order.append((condition["layer_idx"], condition["family"]))
                return 0

            rc, tmp_out = self._run(tmp, fake_run_condition)
            self.assertEqual(rc, 0)
            expected_order = [(l, f) for l in driver.I2_LAYERS for f in driver.I2_FAMILIES]
            self.assertEqual(seen_order, expected_order)
            self.assertEqual(len(seen_order), 16)

    def test_condition_failure_stops_later_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            call_count = {"n": 0}
            exit_codes = [0, 0, 1] + [0] * 13  # fails on the 3rd condition

            def fake_run_condition(*a, **k):
                idx = call_count["n"]
                call_count["n"] += 1
                return exit_codes[idx]

            with self.assertRaises(driver.I2ConfigError):
                self._run(tmp, fake_run_condition)
            self.assertEqual(call_count["n"], 3)

    def test_manifest_written_with_aggregate_all_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, tmp_out = self._run(tmp, lambda *a, **k: 0)
            manifest_path = os.path.join(tmp_out, driver.I2C_FORMAL_MANIFEST_NAME)
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["dataset_revision"], driver.I2_DATASET_REVISION)
            self.assertEqual(len(manifest["conditions"]), 16)
            self.assertTrue(all(c["status"] == "complete" for c in manifest["conditions"]))
            self.assertTrue(manifest["aggregate"]["all_complete"])
            self.assertEqual(manifest["aggregate"]["rows_total"], 23200)
            self.assertEqual(rc, 0)

    def test_lock_file_prevents_concurrent_formal_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(tmp, exist_ok=True)
            lock_path = os.path.join(tmp, driver.I2C_FORMAL_LOCK_NAME)
            fh = driver.acquire_i2_lock(lock_path)
            try:
                with self.assertRaises(driver.I2LockError):
                    driver.acquire_i2_lock(lock_path)
            finally:
                driver.release_i2_lock(fh)

    def test_analysis_module_is_never_imported_or_invoked(self):
        # Comments/docstrings legitimately NAME
        # analysis/analyze_i2_layer_family_sensitivity.py to document that
        # it is deliberately never called (Section 16) -- what must be
        # absent is an actual import or call, not the string appearing in
        # prose. Check via AST rather than a raw substring search.
        import ast

        source = Path(driver.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn("analyze_i2_layer_family_sensitivity", alias.name)
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn("analyze_i2_layer_family_sensitivity", node.module or "")
            elif isinstance(node, ast.Call):
                func = node.func
                chain = []
                while isinstance(func, ast.Attribute):
                    chain.append(func.attr)
                    func = func.value
                if isinstance(func, ast.Name):
                    chain.append(func.id)
                self.assertNotIn("analyze_i2_layer_family_sensitivity", ".".join(reversed(chain)))

    def test_formal_output_isolated_from_canary_subtree(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, tmp_out = self._run(tmp, lambda *a, **k: 0)
            self.assertFalse(os.path.isdir(os.path.join(tmp_out, "canary")))
            top_level_entries = set(os.listdir(tmp_out))
            self.assertNotIn("canary", top_level_entries)


# --- I2A must never touch production kernel files -----------------------------

class TestProductionFilesUntouchedByI2AModules(unittest.TestCase):
    def test_condition_module_does_not_import_production_kernels(self):
        source = Path(driver.__file__).read_text(encoding="utf-8")
        # The GPU-path functions legitimately import models.llama_kivi /
        # quant.* deep inside their bodies (deferred imports) -- but nothing
        # at module scope (import time) should reach them.
        import ast

        tree = ast.parse(source)
        top_level_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        for node in top_level_imports:
            names = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            for name in names:
                if name is None:
                    continue
                self.assertNotIn("llama_kivi", name)
                self.assertNotIn("quant.new_pack", name)
                self.assertNotIn("quant.matmul", name)


if __name__ == "__main__":
    unittest.main()
