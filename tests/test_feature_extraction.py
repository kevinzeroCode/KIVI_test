"""Stage G1 CPU-only tests for utils/feature_extraction.py.

No GPU, no Triton, no model weights loaded. The RoPE-parity test imports
models.llama_kivi (confirmed CPU-importable despite its Triton sub-imports,
since Triton kernels are only invoked, not required at import time) purely
to compare its manual in-place RoPE application against the standard
transformers apply_rotary_pos_emb on synthetic tensors -- no checkpoint is
loaded.
"""
import unittest

import torch

from utils.feature_extraction import (
    DISTRIBUTION_FIELDS,
    RECONSTRUCTION_FIELDS,
    build_feature_record,
    cpu_reference_quantize_dequantize,
    distribution_stats,
    max_abs_error,
    mse,
    reconstruction_stats,
    relative_l2,
)


class RelativeL2Test(unittest.TestCase):
    def test_known_value(self):
        x = torch.tensor([3.0, 4.0])
        x_hat = torch.tensor([0.0, 0.0])
        # ||x-x_hat|| = 5, ||x|| = 5 -> ratio 1.0
        self.assertAlmostEqual(relative_l2(x, x_hat), 1.0, places=10)

    def test_perfect_reconstruction_is_zero(self):
        x = torch.randn(10, 4)
        self.assertAlmostEqual(relative_l2(x, x.clone()), 0.0, places=10)

    def test_zero_norm_x_is_stable(self):
        x = torch.zeros(5)
        x_hat = torch.zeros(5)
        self.assertEqual(relative_l2(x, x_hat), 0.0)

    def test_zero_norm_x_nonzero_x_hat_is_finite(self):
        x = torch.zeros(5)
        x_hat = torch.full((5,), 2.0)
        val = relative_l2(x, x_hat)
        self.assertTrue(torch.isfinite(torch.tensor(val)))
        self.assertGreater(val, 0.0)


class MseTest(unittest.TestCase):
    def test_known_value(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        x_hat = torch.tensor([2.0, 2.0, 2.0])
        # errors: 1, 0, 1 -> squared: 1, 0, 1 -> mean = 2/3
        self.assertAlmostEqual(mse(x, x_hat), 2.0 / 3.0, places=10)

    def test_zero_for_identical(self):
        x = torch.randn(6)
        self.assertEqual(mse(x, x.clone()), 0.0)


class MaxAbsErrorTest(unittest.TestCase):
    def test_known_value(self):
        x = torch.tensor([1.0, -5.0, 3.0])
        x_hat = torch.tensor([1.0, 0.0, 3.0])
        self.assertAlmostEqual(max_abs_error(x, x_hat), 5.0, places=10)


class ReconstructionStatsTest(unittest.TestCase):
    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            reconstruction_stats(torch.randn(3), torch.randn(4))

    def test_returns_exactly_three_fields(self):
        stats = reconstruction_stats(torch.randn(5), torch.randn(5))
        self.assertEqual(set(stats.keys()), set(RECONSTRUCTION_FIELDS))

    def test_deterministic(self):
        x = torch.randn(7, 3)
        x_hat = torch.randn(7, 3)
        s1 = reconstruction_stats(x, x_hat)
        s2 = reconstruction_stats(x, x_hat)
        self.assertEqual(s1, s2)


class DistributionStatsTest(unittest.TestCase):
    def test_mean_std_variance_known(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        stats = distribution_stats(x)
        self.assertAlmostEqual(stats["mean"], 2.5, places=10)
        # population std (divide by N): sqrt(mean((x-mean)^2)) = sqrt(1.25)
        self.assertAlmostEqual(stats["std"], 1.25 ** 0.5, places=10)
        self.assertAlmostEqual(stats["variance"], 1.25, places=10)
        self.assertAlmostEqual(stats["max_abs"], 4.0, places=10)

    def test_percentiles_known(self):
        x = torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        stats = distribution_stats(x)
        self.assertAlmostEqual(stats["p50_abs"], 50.0, places=6)
        self.assertAlmostEqual(stats["p99_abs"], 99.0, places=6)

    def test_outlier_fraction_constant_tensor_is_zero(self):
        x = torch.full((20,), 3.0)
        stats = distribution_stats(x)
        self.assertEqual(stats["std"], 0.0)
        self.assertEqual(stats["outlier_fraction"], 0.0)

    def test_outlier_fraction_detects_single_outlier(self):
        x = torch.cat([torch.zeros(99), torch.tensor([1000.0])])
        stats = distribution_stats(x, outlier_k=3.0)
        self.assertGreater(stats["outlier_fraction"], 0.0)
        self.assertLessEqual(stats["outlier_fraction"], 0.01 + 1e-9)

    def test_returns_exactly_expected_fields(self):
        stats = distribution_stats(torch.randn(10))
        self.assertEqual(set(stats.keys()), set(DISTRIBUTION_FIELDS))

    def test_deterministic(self):
        x = torch.randn(50)
        self.assertEqual(distribution_stats(x), distribution_stats(x.clone()))

    def test_does_not_silently_collapse_head_dimension(self):
        # Two different heads with different scales must yield different
        # per-head stats when the caller passes one head's slice at a time
        # (this module never reduces across a dimension the caller didn't
        # flatten themselves).
        x = torch.zeros(2, 8)
        x[0] = 1.0
        x[1] = 100.0
        stats_head0 = distribution_stats(x[0])
        stats_head1 = distribution_stats(x[1])
        self.assertNotEqual(stats_head0["mean"], stats_head1["mean"])
        self.assertNotEqual(stats_head0["max_abs"], stats_head1["max_abs"])


class DistributionStatsCudaDeviceTest(unittest.TestCase):
    """Regression test for a real bug found during the Stage G2 GPU canary:
    distribution_stats built its percentile tensor with
    torch.tensor(percentiles, dtype=torch.float64) (implicit CPU device),
    which crashed torch.quantile with 'q tensor must be on the same device
    as the input tensor' whenever x lived on CUDA. Not meaningfully
    reproducible on a CPU-only tensor (both sides would trivially be
    'cpu'), so this is intentionally CUDA-gated rather than faked -- it
    auto-skips on CPU-only hosts and does not make the normal suite
    require CUDA."""

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA to reproduce the device-mismatch regression")
    def test_distribution_stats_on_cuda_tensor_does_not_raise(self):
        x = torch.randn(200, device="cuda:0")
        stats = distribution_stats(x)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in stats.values()))


class FeatureRecordSchemaTest(unittest.TestCase):
    def _stats(self):
        x = torch.randn(4, 8)
        x_hat = x + 0.01 * torch.randn(4, 8)
        return reconstruction_stats(x, x_hat), distribution_stats(x)

    def test_valid_axis_key_and_value(self):
        recon, dist = self._stats()
        rec_key = build_feature_record("trec", 0, 5, "key", 8, recon, dist)
        rec_val = build_feature_record("trec", 0, 5, "value", 8, recon, dist)
        self.assertEqual(rec_key["tensor_axis"], "key")
        self.assertEqual(rec_val["tensor_axis"], "value")

    def test_invalid_axis_raises(self):
        recon, dist = self._stats()
        with self.assertRaises(ValueError):
            build_feature_record("trec", 0, 5, "bogus", 8, recon, dist)

    def test_per_layer_per_task_not_collapsed(self):
        recon, dist = self._stats()
        rec_a = build_feature_record("trec", 0, 3, "key", 8, recon, dist)
        rec_b = build_feature_record("lcc", 0, 7, "key", 8, recon, dist)
        self.assertNotEqual(rec_a["task"], rec_b["task"])
        self.assertNotEqual(rec_a["layer_idx"], rec_b["layer_idx"])

    def test_incomplete_stats_raises(self):
        recon, dist = self._stats()
        incomplete_recon = dict(recon)
        del incomplete_recon["mse"]
        with self.assertRaises(ValueError):
            build_feature_record("trec", 0, 5, "key", 8, incomplete_recon, dist)

    def test_aggregation_field_is_explicit(self):
        recon, dist = self._stats()
        rec = build_feature_record("trec", 0, 5, "key", 8, recon, dist, aggregation="no_aggregation_single_group")
        self.assertEqual(rec["aggregation"], "no_aggregation_single_group")


class CpuReferenceQuantizeDequantizeTest(unittest.TestCase):
    def test_shape_round_trip_last_dim(self):
        x = torch.randn(2, 3, 16)
        x_hat = cpu_reference_quantize_dequantize(x, group_size=4, bits=4, dim=-1)
        self.assertEqual(x.shape, x_hat.shape)

    def test_shape_round_trip_other_dim(self):
        x = torch.randn(2, 16, 3)
        x_hat = cpu_reference_quantize_dequantize(x, group_size=8, bits=4, dim=-2)
        self.assertEqual(x.shape, x_hat.shape)

    def test_non_divisible_group_size_raises(self):
        x = torch.randn(2, 10)
        with self.assertRaises(ValueError):
            cpu_reference_quantize_dequantize(x, group_size=4, bits=4, dim=-1)

    def test_deterministic(self):
        x = torch.randn(2, 4, 16)
        x_hat_1 = cpu_reference_quantize_dequantize(x, group_size=4, bits=4)
        x_hat_2 = cpu_reference_quantize_dequantize(x, group_size=4, bits=4)
        self.assertTrue(torch.equal(x_hat_1, x_hat_2))

    def test_constant_group_reconstructs_exactly(self):
        x = torch.full((1, 1, 8), 5.0)
        x_hat = cpu_reference_quantize_dequantize(x, group_size=4, bits=4)
        self.assertTrue(torch.allclose(x, x_hat, atol=1e-6))

    def test_reconstruction_error_monotonic_in_bits(self):
        # Sanity-only: more bits should not make reconstruction worse. This
        # is a property of this CPU reference's own math -- it is NOT a
        # claim about the Triton production kernel's behavior, which
        # cannot be exercised in this CPU-only environment.
        torch.manual_seed(0)
        x = torch.randn(1, 4, 32)
        err_2bit = relative_l2(x, cpu_reference_quantize_dequantize(x, group_size=8, bits=2))
        err_8bit = relative_l2(x, cpu_reference_quantize_dequantize(x, group_size=8, bits=8))
        self.assertLessEqual(err_8bit, err_2bit)


class RopeReproductionParityTest(unittest.TestCase):
    """Confirms that manually reapplying RoPE outside the model, using the
    standard transformers apply_rotary_pos_emb, exactly reproduces
    models.llama_kivi's in-place production RoPE application -- for BOTH
    of its internal branches (the torch.is_grad_enabled() library-fallback
    branch and the no-grad chunked in-place branch), including LongChat's
    linear rope_scaling. This directly resolves (for the RoPE piece only)
    whether external Key-capture-then-manual-RoPE is numerically sound.
    """

    @classmethod
    def setUpClass(cls):
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

        import models.llama_kivi as llama_kivi

        cls.apply_rotary_pos_emb = staticmethod(apply_rotary_pos_emb)
        cls.apply_rotary_pos_emb_inplace = staticmethod(llama_kivi._apply_rotary_pos_emb_inplace)

        cls.config = LlamaConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_hidden_layers=1,
            intermediate_size=64,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            rope_scaling={"type": "linear", "factor": 8.0},
        )
        cls.rotary_emb = LlamaRotaryEmbedding(config=cls.config)

    def _make_inputs(self, batch=1, num_heads=4, seq_len=6, head_dim=8):
        torch.manual_seed(42)
        q = torch.randn(batch, num_heads, seq_len, head_dim)
        k = torch.randn(batch, num_heads, seq_len, head_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        dummy = torch.randn(batch, seq_len, head_dim)
        cos, sin = self.rotary_emb(dummy, position_ids)
        return q, k, cos, sin, position_ids

    def test_grad_enabled_branch_matches_reference_exactly(self):
        q, k, cos, sin, position_ids = self._make_inputs()
        ref_q, ref_k = self.apply_rotary_pos_emb(q.clone(), k.clone(), cos, sin, position_ids)

        self.assertTrue(torch.is_grad_enabled())
        out_q, out_k = self.apply_rotary_pos_emb_inplace(q.clone(), k.clone(), cos, sin, position_ids, chunk_size=None)

        self.assertTrue(torch.allclose(out_q, ref_q, atol=1e-6))
        self.assertTrue(torch.allclose(out_k, ref_k, atol=1e-6))

    def test_no_grad_chunked_inplace_branch_matches_reference(self):
        q, k, cos, sin, position_ids = self._make_inputs()
        ref_q, ref_k = self.apply_rotary_pos_emb(q.clone(), k.clone(), cos, sin, position_ids)

        with torch.no_grad():
            out_q, out_k = self.apply_rotary_pos_emb_inplace(
                q.clone(), k.clone(), cos, sin, position_ids, chunk_size=None
            )

        self.assertTrue(torch.allclose(out_q, ref_q, atol=1e-6))
        self.assertTrue(torch.allclose(out_k, ref_k, atol=1e-6))

    def test_no_grad_small_chunk_size_matches_reference(self):
        q, k, cos, sin, position_ids = self._make_inputs(seq_len=10)
        ref_q, ref_k = self.apply_rotary_pos_emb(q.clone(), k.clone(), cos, sin, position_ids)

        with torch.no_grad():
            out_q, out_k = self.apply_rotary_pos_emb_inplace(
                q.clone(), k.clone(), cos, sin, position_ids, chunk_size=3
            )

        self.assertTrue(torch.allclose(out_q, ref_q, atol=1e-6))
        self.assertTrue(torch.allclose(out_k, ref_k, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
