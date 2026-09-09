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
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
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

    def test_canary_run_flag_refuses_not_executes(self):
        argv = ["--mode", "canary", "--run"]
        with mock.patch.object(sys, "argv", ["run_i2_layer_family_sensitivity.py"] + argv):
            with self.assertRaises(driver.I2CanaryError):
                driver.main()


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
