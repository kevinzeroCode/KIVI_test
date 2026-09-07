"""CPU-only tests for scripts/i1_rotation_kivi_parity_canary.py (Stage
I1-C) and the H2 loader's family generalization
(scripts/h2_attention_decode_parity_canary.py::load_canary_model).

No GPU: neither --run --mode c0 nor --run --mode c1 is invoked anywhere in
this file. What IS exercised directly: argument parsing, safety refusal
paths (torch-free), the locked C0/C1 identity constants, output-root
isolation, and the C0 structural-gate combination logic (factored out into
compute_c0_structural_gate specifically so it is testable with synthetic,
already-computed tensor_diff_report-shaped dicts instead of real GPU
tensors).
"""
import inspect
import runpy
import sys
import unittest
from unittest import mock

import torch

import scripts.h2_attention_decode_parity_canary as h2
import scripts.i1_rotation_kivi_parity_canary as i1


class H2LoaderFamilyGeneralizationTest(unittest.TestCase):
    def test_default_family_is_kivi(self):
        params = inspect.signature(h2.load_canary_model).parameters
        self.assertIn("family", params)
        self.assertEqual(params["family"].default, "kivi")

    def test_original_required_params_unchanged(self):
        params = list(inspect.signature(h2.load_canary_model).parameters)
        self.assertEqual(params[:5], ["model_name_or_path", "cache_dir", "k_bits", "v_bits", "seed"])

    def test_h2_own_call_site_omits_family(self):
        # H2's only internal call site (inside run_axis_canary) must still
        # never pass family= -- proving the default-preservation claim at
        # the source level, not just the signature level.
        source = inspect.getsource(h2.run_axis_canary)
        call_line = next(line for line in source.splitlines() if "load_canary_model(" in line)
        self.assertNotIn("family", call_line)

    def test_h3_call_site_omits_family(self):
        import scripts.run_layer_attention_feature_pilot as h3

        source = inspect.getsource(h3.default_load_model_fn)
        self.assertNotIn("family=", source)

    def test_target_family_wired_into_override_cell(self):
        source = inspect.getsource(h2.load_canary_model)
        self.assertIn('"overrides": {str(layer_idx): {"k_bits": k_bits, "v_bits": v_bits, "family": family}}', source)

    def test_non_target_layers_always_kivi_regardless_of_family_argument(self):
        # The "default" policy cell must be hardcoded family="kivi" and NOT
        # parameterized by the `family` argument -- non-target layers are
        # never affected by what family the target layer uses.
        source = inspect.getsource(h2.load_canary_model)
        self.assertIn('"default": {"k_bits": 16, "v_bits": 16, "family": "kivi"}', source)

    def test_post_construction_validation_checks_family_too(self):
        source = inspect.getsource(h2.load_canary_model)
        self.assertIn("attn.family", source)
        self.assertIn("l.self_attn.family", source)
        self.assertIn('!= (16, 16, "kivi")', source)


class LockedIdentityTest(unittest.TestCase):
    def test_c0_and_c1_share_task_index_layer_seed_group_residual_horizon(self):
        self.assertEqual(i1.CANARY_TASK, "lcc")
        self.assertEqual(i1.CANARY_DATASET_INDEX, 122)
        self.assertEqual(i1.CANARY_LAYER, 0)
        self.assertEqual(i1.CANARY_SEED, 42)
        self.assertEqual(i1.CANARY_GROUP_SIZE, 32)
        self.assertEqual(i1.CANARY_RESIDUAL_LENGTH, 128)
        self.assertEqual(i1.CANARY_MAX_NEW_TOKENS, 6)  # reuses H2's exact horizon

    def test_c0_bit_widths(self):
        self.assertEqual((i1.C0_K_BITS, i1.C0_V_BITS), (16, 16))

    def test_c1_bit_widths(self):
        self.assertEqual((i1.C1_K_BITS, i1.C1_V_BITS), (2, 16))

    def test_preferred_invariance_steps(self):
        self.assertEqual(i1.REQUESTED_INVARIANCE_STEPS, (1, 2, 4))


class ExplicitRunSafetyTest(unittest.TestCase):
    def test_defaults_are_safe(self):
        args = i1.parse_args([])
        self.assertFalse(args.run)
        self.assertIsNone(args.mode)

    def test_main_refuses_without_run(self):
        with mock.patch.object(sys, "argv", ["i1_rotation_kivi_parity_canary.py"]):
            self.assertEqual(i1.main(), 1)

    def test_main_refuses_with_run_but_no_mode(self):
        with mock.patch.object(sys, "argv", ["i1_rotation_kivi_parity_canary.py", "--run"]):
            self.assertEqual(i1.main(), 1)

    def test_main_refuses_with_mode_but_no_run(self):
        with mock.patch.object(sys, "argv", ["i1_rotation_kivi_parity_canary.py", "--mode", "c0"]):
            self.assertEqual(i1.main(), 1)

    def test_invalid_mode_rejected_by_argparse(self):
        with self.assertRaises(SystemExit):
            i1.parse_args(["--mode", "c2"])

    def test_no_torch_import_when_refusing_without_run(self):
        argv_backup = sys.argv
        modules_backup = dict(sys.modules)
        sys.modules.pop("torch", None)
        try:
            sys.argv = ["i1_rotation_kivi_parity_canary.py"]
            try:
                runpy.run_path("scripts/i1_rotation_kivi_parity_canary.py", run_name="__main__")
            except SystemExit:
                pass
            self.assertNotIn("torch", sys.modules)
        finally:
            sys.argv = argv_backup
            # Restore torch to sys.modules if it was there before (other
            # tests in this process may depend on it already being loaded).
            if "torch" in modules_backup:
                sys.modules["torch"] = modules_backup["torch"]

    def test_refuses_on_dirty_git_before_any_gpu_check(self):
        with mock.patch.object(i1, "get_git_status_short", return_value=" M some_file.py\n"):
            with self.assertRaises(i1.CanaryConfigError):
                i1.preflight_checks()

    def test_refuses_on_conflicting_process(self):
        with mock.patch.object(i1, "get_git_status_short", return_value=""), \
             mock.patch.object(i1, "get_git_commit", return_value="deadbeef"), \
             mock.patch.object(i1, "check_no_conflicting_process", return_value=["fake conflicting process"]):
            with self.assertRaises(i1.CanaryLockError):
                i1.preflight_checks()

    def test_refuses_on_busy_gpu(self):
        with mock.patch.object(i1, "get_git_status_short", return_value=""), \
             mock.patch.object(i1, "get_git_commit", return_value="deadbeef"), \
             mock.patch.object(i1, "check_no_conflicting_process", return_value=[]), \
             mock.patch.object(i1, "gpu_preflight", return_value={"checked": True, "warning": "GPU busy"}):
            with self.assertRaises(i1.CanaryConfigError):
                i1.preflight_checks()

    def test_passes_when_all_clear(self):
        with mock.patch.object(i1, "get_git_status_short", return_value=""), \
             mock.patch.object(i1, "get_git_commit", return_value="deadbeef"), \
             mock.patch.object(i1, "check_no_conflicting_process", return_value=[]), \
             mock.patch.object(i1, "gpu_preflight", return_value={"checked": True, "warning": None}):
            head, preflight = i1.preflight_checks()
            self.assertEqual(head, "deadbeef")


class OutputRootIsolationTest(unittest.TestCase):
    def test_c0_c1_roots_under_dedicated_directory(self):
        self.assertIn("i1_rotation_kivi_canary", i1.C0_OUTPUT_ROOT)
        self.assertIn("i1_rotation_kivi_canary", i1.C1_OUTPUT_ROOT)
        self.assertTrue(i1.C0_OUTPUT_ROOT.endswith("/c0"))
        self.assertTrue(i1.C1_OUTPUT_ROOT.endswith("/c1"))

    def test_never_points_at_frozen_or_reserved_roots(self):
        for root in (i1.C0_OUTPUT_ROOT, i1.C1_OUTPUT_ROOT, i1.DEFAULT_OUTPUT_ROOT):
            self.assertNotIn("layer_attention_feature_pilot", root)
            self.assertNotIn("layer_sensitivity_pilot", root)
            self.assertNotIn("layer_feature_pilot", root)


def _diff(a_shape=(1, 2, 3, 4), allclose=True, torch_equal=False, shape_equal=True):
    return {
        "shape_equal": shape_equal, "dtype_equal": True, "torch_equal": torch_equal,
        "allclose_1e-3": allclose, "max_abs_diff": 0.0 if torch_equal else 1e-4,
        "mean_abs_diff": 0.0 if torch_equal else 1e-5, "relative_l2": 0.0 if torch_equal else 1e-5,
    }


class C0StructuralGateTest(unittest.TestCase):
    """Exercises compute_c0_structural_gate directly with synthetic,
    already-computed tensor_diff_report-shaped dicts -- no GPU, no real
    tensors, no model."""

    def _base_kwargs(self, **overrides):
        kwargs = dict(
            cache_checks={
                "rotation_key_quant_trans_is_none": True,
                "reference_key_quant_trans_is_none": True,
                "rotation_key_full_vs_K_expected_rot": _diff(allclose=True, torch_equal=False),
                "reference_key_full_vs_raw_post_rope": _diff(allclose=True, torch_equal=True),
            },
            decode_invariance_records=[_diff(allclose=True, torch_equal=False) for _ in range(3)],
            production_comparison_records=[
                {"pre_softmax_logits": _diff(shape_equal=True), "attn_output_pre_o_proj": _diff(shape_equal=True)} for _ in range(3)
            ],
            generated_prefix_equal=True,
            hook_neutral_reference=True,
            hook_neutral_rotation=True,
            finite_checks=[True, True, True],
        )
        kwargs.update(overrides)
        return kwargs

    def test_all_clear_passes(self):
        gate = i1.compute_c0_structural_gate(**self._base_kwargs())
        self.assertTrue(gate["overall_pass"])

    def test_torch_equal_not_required_for_rotation_math(self):
        # Every rotation-vs-raw/rotation-vs-reference comparison has
        # torch_equal=False (expected, per Section 10) but allclose_1e-3=True
        # -- must still pass.
        kwargs = self._base_kwargs()
        for r in kwargs["decode_invariance_records"]:
            self.assertFalse(r["torch_equal"])
        gate = i1.compute_c0_structural_gate(**kwargs)
        self.assertTrue(gate["qk_invariance_within_tolerance"])
        self.assertTrue(gate["overall_pass"])

    def test_qk_invariance_outside_tolerance_fails(self):
        kwargs = self._base_kwargs(decode_invariance_records=[_diff(allclose=False) for _ in range(3)])
        gate = i1.compute_c0_structural_gate(**kwargs)
        self.assertFalse(gate["qk_invariance_within_tolerance"])
        self.assertFalse(gate["overall_pass"])

    def test_generated_prefix_equality_required(self):
        kwargs = self._base_kwargs(generated_prefix_equal=False)
        gate = i1.compute_c0_structural_gate(**kwargs)
        self.assertFalse(gate["generated_prefix_equal"])
        self.assertFalse(gate["overall_pass"])

    def test_hook_neutrality_required_both_families(self):
        kwargs = self._base_kwargs(hook_neutral_rotation=False)
        gate = i1.compute_c0_structural_gate(**kwargs)
        self.assertFalse(gate["overall_pass"])

    def test_nonfinite_tensor_fails(self):
        kwargs = self._base_kwargs(finite_checks=[True, False, True])
        gate = i1.compute_c0_structural_gate(**kwargs)
        self.assertFalse(gate["finite_tensors"])
        self.assertFalse(gate["overall_pass"])

    def test_unexpected_quantized_prefix_at_k16_fails(self):
        kwargs = self._base_kwargs()
        kwargs["cache_checks"]["rotation_key_quant_trans_is_none"] = False
        gate = i1.compute_c0_structural_gate(**kwargs)
        self.assertFalse(gate["no_unexpected_cache_representation"])
        self.assertFalse(gate["cache_representation_correct"])
        self.assertFalse(gate["overall_pass"])

    def test_reference_cache_must_be_bit_exact_raw_not_just_close(self):
        # The reference (standard kivi) K16 cache must equal raw post-RoPE K
        # EXACTLY (torch_equal), not merely allclose -- it never touches a
        # Hadamard matmul at all, so there is no floating-point excuse for it.
        kwargs = self._base_kwargs()
        kwargs["cache_checks"]["reference_key_full_vs_raw_post_rope"] = _diff(allclose=True, torch_equal=False)
        gate = i1.compute_c0_structural_gate(**kwargs)
        self.assertFalse(gate["cache_representation_correct"])
        self.assertFalse(gate["overall_pass"])


class C1DesignStructureTest(unittest.TestCase):
    """C1 is implemented but never run in this task -- these tests check
    its SOURCE-level design invariants without executing it."""

    def test_generated_prefix_not_required_for_c1(self):
        source = inspect.getsource(i1.run_c1)
        self.assertIn('"generated_prefix_required_to_match": False', source)

    def test_raw_and_rotated_key_never_mixed(self):
        # rotated_call must replace BOTH q_post_rope and k_post_rope
        # together -- never just one, which would silently mix a raw query
        # against a rotated key or vice versa.
        source = inspect.getsource(i1.run_c1)
        self.assertIn('rotated_call["q_post_rope"] = Q_rot', source)
        self.assertIn('rotated_call["k_post_rope"] = K_rot', source)

    def test_reuses_reconstruct_key_step_and_key_decode_distortion_not_reimplemented(self):
        source = inspect.getsource(i1.run_c1)
        self.assertIn("reconstruct_key_step(", source)
        self.assertIn("key_decode_distortion(", source)
        # Never redefines these names locally in this module.
        self.assertNotIn("def reconstruct_key_step", inspect.getsource(i1))
        self.assertNotIn("def key_decode_distortion", inspect.getsource(i1))

    def test_diagnostic_raw_shadow_never_feeds_the_distortion_measurement(self):
        source = inspect.getsource(i1.run_c1)
        # rot_shadow_raw is only used for the invariance cross-check, never
        # passed into reconstruct_key_step (which receives rot_shadow_rotated).
        distortion_call_line = next(line for line in source.splitlines() if "reconstruct_key_step(rotated_call" in line)
        self.assertIn("rot_shadow_rotated", distortion_call_line)
        self.assertNotIn("rot_shadow_raw", distortion_call_line)


class ValueRepresentationOnlyClaimTest(unittest.TestCase):
    def test_value_check_is_a_plain_tensor_diff_not_a_gate_criterion(self):
        # "value" alone would incidentally match "gate.values()" (a dict
        # method call unrelated to the Value KV-cache axis) -- check the
        # specific tokens that would appear if Value were ever folded into
        # the gate instead of staying a separate, non-gating diagnostic.
        gate_source = inspect.getsource(i1.compute_c0_structural_gate)
        for token in ("value_check", "value_full", "value_untouched", "Value"):
            self.assertNotIn(token, gate_source)

    def test_c0_docstring_and_module_docstring_make_no_efficacy_claim(self):
        for text in (i1.run_c0.__doc__ or "", inspect.getsource(i1).split('"""')[1]):
            lowered = text.lower()
            self.assertNotIn("beneficial", lowered)
            self.assertNotIn("better", lowered)
            self.assertNotIn("improves accuracy", lowered)


class C0BehaviorUnchangedByC0DiagTest(unittest.TestCase):
    """Stage I1-C0D Section 3: c0diag must be a strictly additive
    read-only diagnostic -- these tests prove run_c0/run_c1/
    compute_c0_structural_gate were not touched to add it."""

    def test_c0_and_c1_and_gate_functions_still_exist_unchanged_in_signature(self):
        self.assertEqual(list(inspect.signature(i1.run_c0).parameters), ["model_name_or_path", "cache_dir"])
        self.assertEqual(list(inspect.signature(i1.run_c1).parameters), ["model_name_or_path", "cache_dir"])
        self.assertEqual(
            list(inspect.signature(i1.compute_c0_structural_gate).parameters),
            ["cache_checks", "decode_invariance_records", "production_comparison_records",
             "generated_prefix_equal", "hook_neutral_reference", "hook_neutral_rotation", "finite_checks"],
        )

    def test_run_c0_source_never_mentions_c0diag(self):
        source = inspect.getsource(i1.run_c0)
        self.assertNotIn("c0diag", source)

    def test_run_c1_source_never_mentions_c0diag(self):
        source = inspect.getsource(i1.run_c1)
        self.assertNotIn("c0diag", source)

    def test_compute_c0_structural_gate_source_never_mentions_c0diag(self):
        source = inspect.getsource(i1.compute_c0_structural_gate)
        self.assertNotIn("c0diag", source)

    def test_c0diag_never_calls_run_c0_or_run_c1(self):
        source = inspect.getsource(i1.run_c0diag)
        self.assertNotIn("run_c0(", source)
        self.assertNotIn("run_c1(", source)
        self.assertNotIn("compute_c0_structural_gate(", source)

    def test_mode_dispatch_still_resolves_c0_and_c1_to_the_same_functions(self):
        # The refactor from a ternary to a dict lookup must be behaviorally
        # identical -- same function objects, not reimplementations.
        source = inspect.getsource(i1.main)
        self.assertIn('{"c0": run_c0, "c1": run_c1, "c0diag": run_c0diag}', source)

    def test_models_llama_kivi_and_hadamard_and_kernels_not_imported_by_this_module_at_module_level(self):
        # c0diag must not require touching production files -- confirmed by
        # this canary script itself never importing them at module level
        # (only ever inside function bodies, deferred).
        with open(i1.__file__, "r", encoding="utf-8") as f:
            head = "".join(f.readlines()[:60])
        self.assertNotIn("import models.llama_kivi", head)
        self.assertNotIn("from utils.hadamard", head)
        self.assertNotIn("from quant.", head)


class C0DiagDesignTest(unittest.TestCase):
    """Source-level checks for the new --mode c0diag path -- no GPU run."""

    def test_mode_choices_include_c0diag_alongside_c0_c1(self):
        args = i1.parse_args(["--run", "--mode", "c0diag"])
        self.assertEqual(args.mode, "c0diag")

    def test_c0diag_output_root_distinct_from_c0_and_c1(self):
        self.assertNotEqual(i1.C0DIAG_OUTPUT_ROOT, i1.C0_OUTPUT_ROOT)
        self.assertNotEqual(i1.C0DIAG_OUTPUT_ROOT, i1.C1_OUTPUT_ROOT)
        self.assertTrue(i1.C0DIAG_OUTPUT_ROOT.endswith("/c0diag"))
        self.assertIn("i1_rotation_kivi_canary", i1.C0DIAG_OUTPUT_ROOT)

    def test_c0diag_uses_c0_bit_widths_not_c1(self):
        source = inspect.getsource(i1.run_c0diag)
        self.assertIn("C0_K_BITS, C0_V_BITS", source)
        self.assertNotIn("C1_K_BITS", source)
        self.assertNotIn("C1_V_BITS", source)

    def test_c0diag_only_loads_rotation_kivi_never_kivi_reference(self):
        # Section 5/8: only the rotation trajectory is (re)run; the
        # reference route is never re-loaded here.
        source = inspect.getsource(i1.run_c0diag)
        self.assertEqual(source.count("_fresh_model_load("), 1)
        self.assertIn('"rotation_kivi"', source)

    def test_c0diag_defines_no_new_pass_fail_gate(self):
        source = inspect.getsource(i1.run_c0diag)
        self.assertNotIn("overall_pass", source)
        self.assertNotIn('"pass"', source)
        self.assertNotIn("structural_gate", source)

    def test_c0diag_references_frozen_original_c0_run_not_a_recomputation(self):
        source = inspect.getsource(i1.run_c0diag)
        self.assertIn("20260907T054459769185Z", source)
        self.assertIn("NOT recomputed", source)

    def test_fp32_oracle_uses_float32_not_float64_or_float16(self):
        source = inspect.getsource(i1.run_c0diag)
        self.assertIn("Qf = Q.float()", source)
        self.assertIn("Kf = K.float()", source)

    def test_fp16_oracle_casts_from_fp32_rotated_not_directly_from_raw(self):
        # Section 7's QH16/KH16 must derive from the FP32-rotated
        # intermediate (QH32/KH32), not recompute the rotation directly in
        # fp16 from Q/K -- otherwise it would not isolate representation
        # error from rotation-math error.
        source = inspect.getsource(i1.run_c0diag)
        self.assertIn("QH16 = QH32.to(torch.float16)", source)
        self.assertIn("KH16 = KH32.to(torch.float16)", source)

    def test_raw_shadow_never_fed_back_into_generation(self):
        source = inspect.getsource(i1.run_c0diag)
        # generation happens via _generate_and_capture BEFORE the shadow is
        # ever constructed -- the shadow object is never passed to model,
        # tokenizer, or any generate call.
        shadow_construction_idx = source.index("raw_shadow = ShadowKVCache()")
        generate_call_idx = source.index("_generate_and_capture(")
        self.assertLess(generate_call_idx, shadow_construction_idx)
        self.assertNotIn("model.generate", source)


if __name__ == "__main__":
    unittest.main()
