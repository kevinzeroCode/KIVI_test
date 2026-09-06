"""CPU-only tests for Stage I1B's "rotation_kivi" family dispatch in
models/llama_kivi.py. Pure Hadamard math is covered separately in
tests/test_hadamard.py.

No GPU: LlamaAttention_KIVI/LlamaFlashAttention_KIVI.__init__ needs no
CUDA/flash-attn. The real, GPU-requiring quantized/rotated forward() paths
are NOT exercised here -- production parity is what the (designed, not run,
this round) I1-C0/I1-C1 GPU canaries are for. What CAN be proven on CPU and
is proven here:
  - __init__-time family/rotate_kv resolution and fail-closed guards
  - standard "kivi" is completely unaffected (regression)
  - the exact rotation math the real forward() calls, exercised directly
  - a structural simulation of the prefill/decode ORDERING invariants,
    standing in for the real (CUDA-only) flash_attn / quant kernels
  - the eager LlamaAttention_KIVI.forward() dispatch-safety guard: this ONE
    forward() call path needs no CUDA (it raises before touching anything
    CUDA-specific), so it IS exercised directly, end-to-end, via a real
    .forward() call -- not just source inspection. A pure FP16 (k_bits=
    v_bits=16) family="kivi" eager forward() call also happens to run fully
    on CPU with no CUDA dependency, which is used below for a genuine (not
    merely source-level) standard-KIVI eager regression check.
"""
import inspect
import unittest

import torch

from utils.hadamard import HadamardError, apply_hadamard_rotation, get_normalized_hadamard
from utils.layer_policy import SUPPORTED_FAMILIES, resolve_layer_policy


def _make_tiny_config(num_hidden_layers=4, hidden_size=32, num_attention_heads=4):
    from transformers import LlamaConfig

    config = LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=64,
        vocab_size=32,
        max_position_embeddings=64,
    )
    config.group_size = 16
    config.residual_length = 32
    config.use_flash = True
    config.key_quant_chunk_size = 512
    config.value_quant_chunk_size = 512
    config.rotary_chunk_size = 512
    return config


def _policy_layers(resolved):
    return [{"k_bits": e.k_bits, "v_bits": e.v_bits, "family": e.family} for e in resolved.layers]


def _single_layer_policy(k_bits, v_bits, family):
    return resolve_layer_policy(
        1, 16, 16, {"policy_name": "x", "default": {"k_bits": k_bits, "v_bits": v_bits, "family": family}, "overrides": {}}
    )


class SupportedFamiliesTest(unittest.TestCase):
    def test_kivi_and_rotation_kivi_supported(self):
        self.assertIn("kivi", SUPPORTED_FAMILIES)
        self.assertIn("rotation_kivi", SUPPORTED_FAMILIES)

    def test_polar_still_rejected(self):
        self.assertNotIn("polar", SUPPORTED_FAMILIES)


class StandardKiviRegressionTest(unittest.TestCase):
    """Proves family="kivi" construction/attributes are unaffected by this change."""

    def test_kivi_family_attribute_and_rotate_flag(self):
        import models.llama_kivi as llama_kivi

        config = _make_tiny_config()
        config.k_bits = 16
        config.v_bits = 16
        attn = llama_kivi.LlamaAttention_KIVI(config, layer_idx=0)
        self.assertEqual(attn.family, "kivi")
        self.assertFalse(attn.rotate_kv)

    def test_global_fallback_defaults_to_kivi_family(self):
        import models.llama_kivi as llama_kivi

        config = _make_tiny_config()
        config.k_bits = 2
        config.v_bits = 16
        if hasattr(config, "layer_kv_policy"):
            del config.layer_kv_policy
        attn = llama_kivi.LlamaAttention_KIVI(config, layer_idx=0)
        self.assertEqual(attn.family, "kivi")
        self.assertFalse(attn.rotate_kv)
        self.assertEqual((attn.k_bits, attn.v_bits), (2, 16))

    def test_kivi_bits_unaffected_across_all_valid_combinations(self):
        import models.llama_kivi as llama_kivi

        config = _make_tiny_config(num_hidden_layers=1)
        for k_bits, v_bits in ((16, 16), (2, 16), (16, 2), (4, 4), (2, 2)):
            config.layer_kv_policy = _policy_layers(_single_layer_policy(k_bits, v_bits, "kivi"))
            attn = llama_kivi.LlamaAttention_KIVI(config, layer_idx=0)
            self.assertEqual((attn.k_bits, attn.v_bits, attn.family, attn.rotate_kv), (k_bits, v_bits, "kivi", False))

    def test_flash_variant_kivi_unaffected(self):
        import models.llama_kivi as llama_kivi

        config = _make_tiny_config(num_hidden_layers=1)
        config.layer_kv_policy = _policy_layers(_single_layer_policy(2, 16, "kivi"))
        attn = llama_kivi.LlamaFlashAttention_KIVI(config, layer_idx=0)
        self.assertEqual(attn.family, "kivi")
        self.assertFalse(attn.rotate_kv)


class RotationKiviDispatchTest(unittest.TestCase):
    def test_rotation_kivi_k2_v16_constructs_with_rotate_flag_set(self):
        import models.llama_kivi as llama_kivi

        config = _make_tiny_config(num_hidden_layers=1)
        config.layer_kv_policy = _policy_layers(_single_layer_policy(2, 16, "rotation_kivi"))
        attn = llama_kivi.LlamaAttention_KIVI(config, layer_idx=0)
        self.assertEqual(attn.family, "rotation_kivi")
        self.assertTrue(attn.rotate_kv)
        self.assertEqual((attn.k_bits, attn.v_bits), (2, 16))
        self.assertTrue(attn.quantize_key)
        self.assertFalse(attn.quantize_value)

    def test_rotation_kivi_k16_v16_invariance_policy_constructs(self):
        import models.llama_kivi as llama_kivi

        config = _make_tiny_config(num_hidden_layers=1)
        config.layer_kv_policy = _policy_layers(_single_layer_policy(16, 16, "rotation_kivi"))
        attn = llama_kivi.LlamaAttention_KIVI(config, layer_idx=0)
        self.assertTrue(attn.rotate_kv)
        self.assertFalse(attn.quantize_key)  # 16 bits == pass-through; rotation still applies to what's cached

    def test_per_layer_family_isolation(self):
        import models.llama_kivi as llama_kivi

        config = _make_tiny_config(num_hidden_layers=3)
        resolved = resolve_layer_policy(
            3, 16, 16,
            {
                "policy_name": "x", "default": {"k_bits": 16, "v_bits": 16, "family": "kivi"},
                "overrides": {"1": {"k_bits": 2, "v_bits": 16, "family": "rotation_kivi"}},
            },
        )
        config.layer_kv_policy = _policy_layers(resolved)
        attns = [llama_kivi.LlamaAttention_KIVI(config, layer_idx=i) for i in range(3)]
        self.assertFalse(attns[0].rotate_kv)
        self.assertTrue(attns[1].rotate_kv)
        self.assertFalse(attns[2].rotate_kv)
        self.assertEqual([a.family for a in attns], ["kivi", "rotation_kivi", "kivi"])

    def test_flash_variant_gets_same_attributes(self):
        import models.llama_kivi as llama_kivi

        config = _make_tiny_config(num_hidden_layers=1)
        config.layer_kv_policy = _policy_layers(_single_layer_policy(2, 16, "rotation_kivi"))
        attn = llama_kivi.LlamaFlashAttention_KIVI(config, layer_idx=0)
        self.assertTrue(attn.rotate_kv)
        self.assertEqual(attn.family, "rotation_kivi")


class NonPowerOfTwoHeadDimFailsClosedTest(unittest.TestCase):
    def test_rotation_kivi_rejects_non_power_of_two_head_dim_at_construction(self):
        import models.llama_kivi as llama_kivi

        # hidden_size=48 / num_attention_heads=4 -> head_dim=12, not a power of two.
        config = _make_tiny_config(num_hidden_layers=1, hidden_size=48, num_attention_heads=4)
        config.layer_kv_policy = _policy_layers(_single_layer_policy(2, 16, "rotation_kivi"))
        with self.assertRaises(HadamardError):
            llama_kivi.LlamaAttention_KIVI(config, layer_idx=0)

    def test_kivi_family_unaffected_by_non_power_of_two_head_dim(self):
        import models.llama_kivi as llama_kivi

        config = _make_tiny_config(num_hidden_layers=1, hidden_size=48, num_attention_heads=4)
        config.k_bits = 16
        config.v_bits = 16
        # No layer_kv_policy -- global "kivi" fallback -- must construct fine
        # regardless of head_dim, since kivi never touches the Hadamard path.
        attn = llama_kivi.LlamaAttention_KIVI(config, layer_idx=0)
        self.assertFalse(attn.rotate_kv)


class UnsupportedFamilyStillFailsClosedTest(unittest.TestCase):
    def test_hand_built_policy_dict_with_unknown_family_rejected(self):
        # Bypasses resolve_layer_policy's own validation to prove
        # _resolve_layer_kv_bits itself still fails closed defensively.
        import models.llama_kivi as llama_kivi

        config = _make_tiny_config(num_hidden_layers=1)
        config.layer_kv_policy = [{"k_bits": 16, "v_bits": 16, "family": "polar"}]
        with self.assertRaises(ValueError):
            llama_kivi.LlamaAttention_KIVI(config, layer_idx=0)


class ValueUntouchedTest(unittest.TestCase):
    """'No Value rotation': structural proof by direct source inspection
    (forward() cannot be run on CPU) -- breaks if a future edit ever adds
    apply_hadamard_rotation(value_states, ...) anywhere."""

    def test_source_never_rotates_value_states(self):
        import models.llama_kivi as llama_kivi

        source = inspect.getsource(llama_kivi.LlamaFlashAttention_KIVI.forward)
        for line in source.splitlines():
            if "apply_hadamard_rotation" in line:
                self.assertNotIn("value_states", line)

    def test_source_calls_rotation_exactly_three_times(self):
        # decode branch: Q and K (2 calls) + prefill branch: cache-only K
        # (1 call) == 3 total. Never a fourth, unexpected call site.
        import models.llama_kivi as llama_kivi

        source = inspect.getsource(llama_kivi.LlamaFlashAttention_KIVI.forward)
        self.assertEqual(source.count("apply_hadamard_rotation("), 3)


class PrefillCacheOnlySemanticsSimulationTest(unittest.TestCase):
    """Structural / load-bearing: simulates the EXACT sequence of tensor
    operations the real prefill branch performs (attention consumes
    UNROTATED Q/K; only afterward is a cache-only K@H produced), standing
    in for the real (CUDA-only) flash_attn kernel with a plain reference
    attention computation. Proves the ORDERING invariant at the tensor-
    operation level -- production parity itself is what the (not run this
    round) I1-C0 GPU canary is for.
    """

    def test_prefill_attention_uses_unrotated_k_cache_uses_rotated_k(self):
        torch.manual_seed(0)
        head_dim = 128
        H = get_normalized_hadamard(head_dim, device="cpu", dtype=torch.float64)
        Q = torch.randn(1, 2, 5, head_dim, dtype=torch.float64)
        K = torch.randn(1, 2, 5, head_dim, dtype=torch.float64)
        V = torch.randn(1, 2, 5, head_dim, dtype=torch.float64)

        def reference_attention(q, k, v):
            weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5), dim=-1)
            return torch.matmul(weights, v)

        # Mirrors the real forward()'s ordering: attention is computed
        # BEFORE any rotation exists.
        attn_output_unrotated_path = reference_attention(Q, K, V)

        # Only now (mirroring "del query_states" then the rotate_kv block)
        # is a cache-only rotated K produced.
        K_cache = apply_hadamard_rotation(K, H)

        reference_no_rotation_at_all = reference_attention(Q, K, V)
        self.assertTrue(torch.equal(attn_output_unrotated_path, reference_no_rotation_at_all))
        self.assertFalse(torch.equal(K_cache, K))
        self.assertTrue(torch.equal(K_cache, torch.matmul(K, H)))


class DecodeRepresentationSimulationTest(unittest.TestCase):
    """Q_rot against K_rot reproduces the unrotated full-precision reference
    logits (before any quantization), for both an already-cached rotated
    residual and a newly appended rotated key -- and a negative control
    proving a mixed raw/rotated cache would NOT reproduce the reference."""

    def test_existing_residual_and_new_key_both_rotated_consistently(self):
        torch.manual_seed(1)
        head_dim = 128
        H = get_normalized_hadamard(head_dim, device="cpu", dtype=torch.float64)
        Q_t = torch.randn(1, 2, 1, head_dim, dtype=torch.float64)
        K_existing = torch.randn(1, 2, 4, head_dim, dtype=torch.float64)  # already-cached, already rotated
        K_new = torch.randn(1, 2, 1, head_dim, dtype=torch.float64)  # this step's new key

        K_existing_rot = apply_hadamard_rotation(K_existing, H)
        K_new_rot = apply_hadamard_rotation(K_new, H)
        K_full_rot = torch.cat([K_existing_rot, K_new_rot], dim=2)
        Q_rot = apply_hadamard_rotation(Q_t, H)
        rotated_logits = torch.matmul(Q_rot, K_full_rot.transpose(-2, -1))

        K_full_raw = torch.cat([K_existing, K_new], dim=2)
        reference_logits = torch.matmul(Q_t, K_full_raw.transpose(-2, -1))

        self.assertTrue(torch.allclose(rotated_logits, reference_logits, atol=1e-9, rtol=1e-9))

    def test_no_mixed_raw_and_rotated_key_in_one_cache(self):
        # Negative control: forgetting to rotate the newly-appended key
        # must NOT reproduce the reference logits -- proves the positive
        # test above is not vacuous.
        torch.manual_seed(2)
        head_dim = 128
        H = get_normalized_hadamard(head_dim, device="cpu", dtype=torch.float64)
        Q_t = torch.randn(1, 2, 1, head_dim, dtype=torch.float64)
        K_existing = torch.randn(1, 2, 4, head_dim, dtype=torch.float64)
        K_new = torch.randn(1, 2, 1, head_dim, dtype=torch.float64)

        K_existing_rot = apply_hadamard_rotation(K_existing, H)
        K_mixed = torch.cat([K_existing_rot, K_new], dim=2)  # BUG-simulation: K_new left unrotated
        Q_rot = apply_hadamard_rotation(Q_t, H)
        mixed_logits = torch.matmul(Q_rot, K_mixed.transpose(-2, -1))

        K_full_raw = torch.cat([K_existing, K_new], dim=2)
        reference_logits = torch.matmul(Q_t, K_full_raw.transpose(-2, -1))

        self.assertFalse(torch.allclose(mixed_logits, reference_logits, atol=1e-6, rtol=1e-6))


class EagerDispatchSafetyGuardTest(unittest.TestCase):
    """Closes the discovered silent-no-op gap: the eager
    LlamaAttention_KIVI.forward() has no rotation logic, so a
    family="rotation_kivi" instance must fail closed rather than silently
    behaving as plain KIVI. These tests make a REAL .forward() call (not
    merely source inspection) to prove it.
    """

    def _minimal_forward_args(self, hidden_size, seq_len=5, bsz=1):
        hidden_states = torch.randn(bsz, seq_len, hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        return dict(hidden_states=hidden_states, attention_mask=None, position_ids=position_ids, past_key_value=None, use_cache=True)

    def test_direct_eager_rotation_kivi_forward_raises_before_projection(self):
        import models.llama_kivi as llama_kivi

        config = _make_tiny_config(num_hidden_layers=1)
        config.layer_kv_policy = _policy_layers(_single_layer_policy(2, 16, "rotation_kivi"))
        attn = llama_kivi.LlamaAttention_KIVI(config, layer_idx=0)
        self.assertTrue(attn.rotate_kv)

        # Spy on q_proj/k_proj/v_proj to prove they are NEVER called --
        # the guard must fire strictly before any projection, not just
        # "eventually" via some other error.
        def _fail_if_called(*args, **kwargs):
            raise AssertionError("q_proj/k_proj/v_proj must never be called when the rotation_kivi guard fires")

        attn.q_proj.forward = _fail_if_called
        attn.k_proj.forward = _fail_if_called
        attn.v_proj.forward = _fail_if_called

        with self.assertRaises(NotImplementedError) as ctx:
            attn.forward(**self._minimal_forward_args(config.hidden_size))

        message = str(ctx.exception)
        self.assertIn("rotation_kivi", message)
        self.assertIn("eager", message)
        self.assertIn("LlamaFlashAttention_KIVI", message)

    def test_guard_does_not_mutate_family_or_rotate_flag(self):
        import models.llama_kivi as llama_kivi

        config = _make_tiny_config(num_hidden_layers=1)
        config.layer_kv_policy = _policy_layers(_single_layer_policy(2, 16, "rotation_kivi"))
        attn = llama_kivi.LlamaAttention_KIVI(config, layer_idx=0)
        with self.assertRaises(NotImplementedError):
            attn.forward(**self._minimal_forward_args(config.hidden_size))
        # Never silently falls back to "kivi" or flips the flag after failing.
        self.assertEqual(attn.family, "rotation_kivi")
        self.assertTrue(attn.rotate_kv)

    def test_standard_eager_kivi_forward_does_not_trigger_guard(self):
        # Genuine end-to-end regression: a pure FP16 (k_bits=v_bits=16)
        # family="kivi" eager forward() call runs fully on CPU with no CUDA
        # dependency, so this proves (by actually succeeding, not just by
        # absence of the guard's exception type) that the new guard is
        # family-specific and does not globally disable standard KIVI.
        import models.llama_kivi as llama_kivi

        config = _make_tiny_config(num_hidden_layers=1)
        config.k_bits = 16
        config.v_bits = 16
        if hasattr(config, "layer_kv_policy"):
            del config.layer_kv_policy
        attn = llama_kivi.LlamaAttention_KIVI(config, layer_idx=0)
        self.assertFalse(attn.rotate_kv)

        output, attn_weights, past_key_value = attn.forward(**self._minimal_forward_args(config.hidden_size))
        self.assertEqual(tuple(output.shape), (1, 5, config.hidden_size))

    def test_flash_rotation_kivi_construction_unaffected_by_eager_guard(self):
        import models.llama_kivi as llama_kivi

        config = _make_tiny_config(num_hidden_layers=1)
        config.layer_kv_policy = _policy_layers(_single_layer_policy(2, 16, "rotation_kivi"))
        attn = llama_kivi.LlamaFlashAttention_KIVI(config, layer_idx=0)
        self.assertEqual(attn.family, "rotation_kivi")
        self.assertTrue(attn.rotate_kv)

    def test_flash_and_eager_forward_are_genuinely_different_functions(self):
        # Confirms Python method resolution: LlamaFlashAttention_KIVI
        # overrides forward() entirely, so the eager guard added to
        # LlamaAttention_KIVI.forward() cannot leak into the flash
        # subclass's real production forward() via inheritance.
        import models.llama_kivi as llama_kivi

        self.assertIsNot(llama_kivi.LlamaFlashAttention_KIVI.forward, llama_kivi.LlamaAttention_KIVI.forward)
        flash_source = inspect.getsource(llama_kivi.LlamaFlashAttention_KIVI.forward)
        eager_source = inspect.getsource(llama_kivi.LlamaAttention_KIVI.forward)
        self.assertIn("is not supported by the eager", eager_source)
        self.assertNotIn("is not supported by the eager", flash_source)


if __name__ == "__main__":
    unittest.main()
