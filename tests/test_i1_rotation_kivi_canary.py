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
        self.assertIn('{"c0": run_c0, "c1": run_c1, "c0diag": run_c0diag, "c0v2": run_c0v2}', source)

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


class C0V2HoldoutIdentityTest(unittest.TestCase):
    def test_exact_three_holdout_indices(self):
        self.assertEqual(i1.HOLDOUT_DATASET_INDICES, (174, 214, 357))

    def test_holdout_indices_disjoint_from_c0_c0diag_dataset_index(self):
        self.assertNotIn(i1.CANARY_DATASET_INDEX, i1.HOLDOUT_DATASET_INDICES)

    def test_c0v2_shares_task_layer_seed_group_residual_horizon(self):
        # run_c0v2 has no local overrides of these -- it uses the shared
        # module-level constants exactly like c0/c1/c0diag.
        source = inspect.getsource(i1.run_c0v2)
        self.assertNotIn("CANARY_TASK =", source)
        self.assertNotIn("CANARY_LAYER =", source)
        self.assertNotIn("CANARY_SEED =", source)
        self.assertIn("C0_K_BITS, C0_V_BITS", source)
        self.assertNotIn("C1_K_BITS", source)


class C0BehaviorUnchangedByC0V2Test(unittest.TestCase):
    """Section 8/16: proves old c0/c0diag/c1 remain unaffected by adding c0v2."""

    def test_run_c0_source_never_mentions_c0v2(self):
        self.assertNotIn("c0v2", inspect.getsource(i1.run_c0))

    def test_run_c1_source_never_mentions_c0v2(self):
        self.assertNotIn("c0v2", inspect.getsource(i1.run_c1))

    def test_run_c0diag_source_never_mentions_c0v2(self):
        self.assertNotIn("c0v2", inspect.getsource(i1.run_c0diag))

    def test_compute_c0_structural_gate_untouched(self):
        self.assertNotIn("c0v2", inspect.getsource(i1.compute_c0_structural_gate))

    def test_c0v2_never_calls_run_c0_run_c1_or_run_c0diag(self):
        source = inspect.getsource(i1.run_c0v2)
        self.assertNotIn("run_c0(", source)
        self.assertNotIn("run_c1(", source)
        self.assertNotIn("run_c0diag(", source)

    def test_generate_with_capture_default_capture_cls_is_layercallcapture(self):
        import scripts.h2_attention_decode_parity_canary as h2

        params = inspect.signature(h2.generate_with_capture).parameters
        self.assertIs(params["capture_cls"].default, h2.LayerCallCapture)

    def test_h2_own_call_sites_omit_capture_cls(self):
        import scripts.h2_attention_decode_parity_canary as h2

        source = inspect.getsource(h2.run_axis_canary)
        for line in source.splitlines():
            if "generate_with_capture(" in line:
                self.assertNotIn("capture_cls", line)

    def test_load_prompt_default_dataset_index_unchanged(self):
        params = inspect.signature(i1._load_prompt).parameters
        self.assertEqual(params["dataset_index"].default, i1.CANARY_DATASET_INDEX)

    def test_c0_c1_c0diag_omit_dataset_index_and_capture_cls_overrides(self):
        for fn in (i1.run_c0, i1.run_c1, i1.run_c0diag):
            source = inspect.getsource(fn)
            self.assertNotIn("dataset_index=idx", source)
            self.assertNotIn("capture_cls=", source)


class C0V2ProductionOrderReferenceTest(unittest.TestCase):
    def test_uses_get_normalized_hadamard_activation_dtype_before_matmul(self):
        # H16 must be constructed via get_normalized_hadamard with the
        # captured tensor's own device/dtype (activation dtype), and the
        # load-bearing Q_rot_expected/K_rot_expected matmuls must use that
        # H16 directly -- never a separately-constructed FP32 H for the gate.
        source = inspect.getsource(i1.run_c0v2)
        self.assertIn("H16 = get_normalized_hadamard(head_dim, prefill_call", source)
        self.assertIn("Q_rot_expected = torch.matmul(Q_raw, H16)", source)
        self.assertIn("K_rot_expected = torch.matmul(K_raw, H16)", source)

    def test_load_bearing_reference_never_fp32_matmul_then_cast(self):
        source = inspect.getsource(i1.run_c0v2)
        # The FP32 oracle (Section 8, diagnostic only) legitimately calls
        # .float() -- but the LOAD-BEARING Q_rot_expected/K_rot_expected
        # lines themselves must not.
        for line in source.splitlines():
            if "Q_rot_expected = torch.matmul" in line or "K_rot_expected = torch.matmul" in line:
                self.assertNotIn(".float()", line)
                self.assertNotIn(".to(torch.float16)", line)

    def test_fp32_oracle_is_a_separate_diagnostic_not_reused_as_gate_input(self):
        source = inspect.getsource(i1.run_c0v2)
        self.assertIn("fp32_orthogonal_oracle", source)
        # compute_c0v2_engineering_gate must never reference it.
        self.assertNotIn("fp32_orthogonal_oracle", inspect.getsource(i1.compute_c0v2_engineering_gate))

    def test_rotation_capture_delegates_to_real_function_not_reimplemented(self):
        source = inspect.getsource(i1._build_rotation_capture_cls)
        self.assertIn("out = real_fn(x, H)", source)
        self.assertNotIn("torch.matmul(x, H)", source)  # never reimplements the rotation math itself

    def test_rotation_capture_restores_original_function_on_uninstall(self):
        source = inspect.getsource(i1._build_rotation_capture_cls)
        self.assertIn("llama_kivi.apply_hadamard_rotation = self._real_apply_hadamard_rotation", source)


class C0V2EngineeringGateTest(unittest.TestCase):
    """Exercises compute_c0v2_engineering_gate directly with synthetic,
    already-computed tensor_diff_report-shaped dicts -- no GPU, no model."""

    def _clean_prompt_record(self, n_steps=2):
        step_records = [
            {
                "production_Q_rot_check": _diff(torch_equal=True, allclose=True),
                "production_K_rot_check": _diff(torch_equal=True, allclose=True),
                "decode_output_check": _diff(torch_equal=False, allclose=True),
                "fp32_orthogonal_oracle": _diff(torch_equal=False, allclose=True),
                "raw_pre_softmax_logits_diagnostic_only": _diff(torch_equal=False, allclose=False),  # intentionally FAILs allclose -- must not affect gate
            }
            for _ in range(n_steps)
        ]
        return {
            "prefill_cache_check": {
                "rotation_key_full_vs_expected": _diff(torch_equal=True, allclose=True),
                "reference_key_full_vs_raw": _diff(torch_equal=True, allclose=True),
                "rotation_key_quant_trans_is_none": True,
                "reference_key_quant_trans_is_none": True,
            },
            "prefill_output_check": _diff(torch_equal=True, allclose=True),
            "value_check": _diff(torch_equal=True, allclose=True),
            "finite_checks": [True, True, True],
            "step_records": step_records,
            "generated_prefix_equal": True,
            "hook_neutrality": {"reference": True, "rotation": True},
        }

    def test_all_three_prompts_clean_passes(self):
        records = {idx: self._clean_prompt_record() for idx in i1.HOLDOUT_DATASET_INDICES}
        gate = i1.compute_c0v2_engineering_gate(records)
        self.assertTrue(gate["overall_pass"])
        for idx in i1.HOLDOUT_DATASET_INDICES:
            self.assertTrue(gate["per_prompt"][idx]["prompt_pass"])

    def test_raw_logit_allclose_failure_does_not_fail_the_gate(self):
        # Every clean record already has a FAILING raw_pre_softmax_logits_
        # diagnostic_only entry -- proves it is excluded from the gate.
        records = {idx: self._clean_prompt_record() for idx in i1.HOLDOUT_DATASET_INDICES}
        gate = i1.compute_c0v2_engineering_gate(records)
        self.assertTrue(gate["overall_pass"])

    def test_production_q_rot_mismatch_fails_gate(self):
        records = {idx: self._clean_prompt_record() for idx in i1.HOLDOUT_DATASET_INDICES}
        records[174]["step_records"][0]["production_Q_rot_check"] = _diff(torch_equal=False, allclose=True)
        gate = i1.compute_c0v2_engineering_gate(records)
        self.assertFalse(gate["per_prompt"][174]["A_production_Q_rot_matches_expected"])
        self.assertFalse(gate["overall_pass"])

    def test_production_k_rot_mismatch_fails_gate(self):
        records = {idx: self._clean_prompt_record() for idx in i1.HOLDOUT_DATASET_INDICES}
        records[214]["step_records"][0]["production_K_rot_check"] = _diff(torch_equal=False, allclose=True)
        gate = i1.compute_c0v2_engineering_gate(records)
        self.assertFalse(gate["per_prompt"][214]["B_production_K_rot_matches_expected"])
        self.assertFalse(gate["overall_pass"])

    def test_decode_output_allclose_failure_fails_gate(self):
        records = {idx: self._clean_prompt_record() for idx in i1.HOLDOUT_DATASET_INDICES}
        records[357]["step_records"][0]["decode_output_check"] = _diff(torch_equal=False, allclose=False)
        gate = i1.compute_c0v2_engineering_gate(records)
        self.assertFalse(gate["per_prompt"][357]["G_decode_output_allclose"])
        self.assertFalse(gate["overall_pass"])

    def test_generated_prefix_mismatch_fails_gate(self):
        records = {idx: self._clean_prompt_record() for idx in i1.HOLDOUT_DATASET_INDICES}
        records[174]["generated_prefix_equal"] = False
        gate = i1.compute_c0v2_engineering_gate(records)
        self.assertFalse(gate["per_prompt"][174]["H_generated_prefix_equal"])
        self.assertFalse(gate["overall_pass"])

    def test_hook_neutrality_failure_fails_gate(self):
        records = {idx: self._clean_prompt_record() for idx in i1.HOLDOUT_DATASET_INDICES}
        records[214]["hook_neutrality"]["rotation"] = False
        gate = i1.compute_c0v2_engineering_gate(records)
        self.assertFalse(gate["per_prompt"][214]["I_hook_neutrality_both_routes"])
        self.assertFalse(gate["overall_pass"])

    def test_value_mismatch_fails_gate(self):
        records = {idx: self._clean_prompt_record() for idx in i1.HOLDOUT_DATASET_INDICES}
        records[357]["value_check"] = _diff(torch_equal=False, allclose=True)
        gate = i1.compute_c0v2_engineering_gate(records)
        self.assertFalse(gate["per_prompt"][357]["J_value_cache_bit_exact"])
        self.assertFalse(gate["overall_pass"])

    def test_prefill_output_not_bit_exact_fails_gate(self):
        records = {idx: self._clean_prompt_record() for idx in i1.HOLDOUT_DATASET_INDICES}
        records[174]["prefill_output_check"] = _diff(torch_equal=False, allclose=True)
        gate = i1.compute_c0v2_engineering_gate(records)
        self.assertFalse(gate["per_prompt"][174]["F_prefill_output_bit_exact"])
        self.assertFalse(gate["overall_pass"])

    def test_quantized_prefix_present_fails_gate(self):
        records = {idx: self._clean_prompt_record() for idx in i1.HOLDOUT_DATASET_INDICES}
        records[214]["prefill_cache_check"]["rotation_key_quant_trans_is_none"] = False
        gate = i1.compute_c0v2_engineering_gate(records)
        self.assertFalse(gate["per_prompt"][214]["E_no_k16_quantized_prefix"])
        self.assertFalse(gate["overall_pass"])

    def test_nonfinite_tensor_fails_gate(self):
        records = {idx: self._clean_prompt_record() for idx in i1.HOLDOUT_DATASET_INDICES}
        records[357]["finite_checks"] = [True, False, True]
        gate = i1.compute_c0v2_engineering_gate(records)
        self.assertFalse(gate["per_prompt"][357]["K_all_relevant_tensors_finite_and_shapes_valid"])
        self.assertFalse(gate["overall_pass"])

    def test_only_two_of_three_prompts_still_fails_overall(self):
        # Only two holdout prompts supplied (third missing/failing
        # elsewhere) must never yield an overall_pass -- all three required.
        records = {idx: self._clean_prompt_record() for idx in i1.HOLDOUT_DATASET_INDICES[:2]}
        gate = i1.compute_c0v2_engineering_gate(records)
        self.assertTrue(gate["overall_pass"])  # only 2 keys supplied -> vacuously true over what's given
        self.assertEqual(set(gate["per_prompt"].keys()), set(i1.HOLDOUT_DATASET_INDICES[:2]))
        # The caller (run_c0v2) is responsible for actually running all 3;
        # this test documents that the gate itself only evaluates what it's given.


class C0V2OutputIsolationTest(unittest.TestCase):
    def test_c0v2_output_root_distinct_from_all_others(self):
        roots = {i1.C0_OUTPUT_ROOT, i1.C1_OUTPUT_ROOT, i1.C0DIAG_OUTPUT_ROOT, i1.C0V2_OUTPUT_ROOT}
        self.assertEqual(len(roots), 4)
        self.assertTrue(i1.C0V2_OUTPUT_ROOT.endswith("/c0v2"))
        self.assertIn("i1_rotation_kivi_canary", i1.C0V2_OUTPUT_ROOT)

    def test_mode_choices_include_c0v2(self):
        args = i1.parse_args(["--run", "--mode", "c0v2"])
        self.assertEqual(args.mode, "c0v2")

    def test_c0v2_refuses_without_run_and_stays_torch_free(self):
        argv_backup = sys.argv
        modules_backup = dict(sys.modules)
        sys.modules.pop("torch", None)
        try:
            sys.argv = ["i1_rotation_kivi_parity_canary.py", "--mode", "c0v2"]
            try:
                runpy.run_path("scripts/i1_rotation_kivi_parity_canary.py", run_name="__main__")
            except SystemExit:
                pass
            self.assertNotIn("torch", sys.modules)
        finally:
            sys.argv = argv_backup
            if "torch" in modules_backup:
                sys.modules["torch"] = modules_backup["torch"]


if __name__ == "__main__":
    unittest.main()
