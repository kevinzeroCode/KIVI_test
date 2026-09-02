"""CPU-only tests for utils/attention_decode_features.py (Stage H0-PRE
measurement infrastructure). No GPU/model/production state -- every test
here uses synthetic tensors or synthetic tuples.
"""
import unittest

import torch

from utils.attention_decode_features import (
    DEFAULT_REQUESTED_STEPS,
    AttentionDecodeFeatureError,
    KIVICacheState,
    ShadowCacheError,
    ShadowKVCache,
    aggregate_sample_decode_distortion,
    build_decode_feature_record,
    key_decode_distortion,
    parse_kivi_cache_tuple,
    value_decode_distortion,
)


class KeyDecodeDistortionTest(unittest.TestCase):
    def test_zero_for_identical_tensors(self):
        x = torch.randn(1, 4, 3, 8)
        self.assertEqual(key_decode_distortion(x, x.clone()), 0.0)

    def test_finite_and_nonnegative_for_random_tensors(self):
        torch.manual_seed(0)
        a = torch.randn(1, 4, 5, 8)
        b = torch.randn(1, 4, 5, 8)
        val = key_decode_distortion(a, b)
        self.assertTrue(torch.isfinite(torch.tensor(val)))
        self.assertGreaterEqual(val, 0.0)

    def test_known_value(self):
        L_fp16 = torch.tensor([[3.0, 4.0]])  # norm 5
        L_kivi = torch.tensor([[0.0, 0.0]])
        self.assertAlmostEqual(key_decode_distortion(L_kivi, L_fp16), 1.0, places=10)

    def test_deterministic(self):
        torch.manual_seed(1)
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 3, 4)
        self.assertEqual(key_decode_distortion(a, b), key_decode_distortion(a, b))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(AttentionDecodeFeatureError):
            key_decode_distortion(torch.randn(3), torch.randn(4))

    def test_zero_norm_reference_is_stable(self):
        L_fp16 = torch.zeros(4)
        L_kivi = torch.full((4,), 2.0)
        val = key_decode_distortion(L_kivi, L_fp16)
        self.assertTrue(torch.isfinite(torch.tensor(val)))
        self.assertGreater(val, 0.0)


class ValueDecodeDistortionTest(unittest.TestCase):
    def test_zero_for_identical_tensors(self):
        x = torch.randn(1, 4, 3, 8)
        self.assertEqual(value_decode_distortion(x, x.clone()), 0.0)

    def test_finite_and_nonnegative(self):
        torch.manual_seed(2)
        a = torch.randn(1, 4, 5, 8)
        b = torch.randn(1, 4, 5, 8)
        val = value_decode_distortion(a, b)
        self.assertTrue(torch.isfinite(torch.tensor(val)))
        self.assertGreaterEqual(val, 0.0)

    def test_deterministic(self):
        torch.manual_seed(3)
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 3, 4)
        self.assertEqual(value_decode_distortion(a, b), value_decode_distortion(a, b))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(AttentionDecodeFeatureError):
            value_decode_distortion(torch.randn(3), torch.randn(4))


class AggregateSampleDecodeDistortionTest(unittest.TestCase):
    def test_all_requested_steps_available_equal_mean(self):
        steps = {1: 0.1, 2: 0.2, 4: 0.3, 8: 0.4, 16: 0.5}
        result = aggregate_sample_decode_distortion(steps)
        self.assertEqual(result["valid_steps"], (1, 2, 4, 8, 16))
        self.assertAlmostEqual(result["sample_attention_distortion"], 0.3, places=10)
        self.assertFalse(result["no_decode_exposure"])
        self.assertEqual(result["valid_step_count"], 5)

    def test_partial_steps_available_early_eos_example(self):
        # docs Section 5 example: available steps = {1,2} -> aggregate only those.
        steps = {1: 0.2, 2: 0.6}
        result = aggregate_sample_decode_distortion(steps)
        self.assertEqual(result["valid_steps"], (1, 2))
        self.assertAlmostEqual(result["sample_attention_distortion"], 0.4, places=10)
        self.assertEqual(result["valid_step_count"], 2)
        self.assertFalse(result["no_decode_exposure"])

    def test_zero_decode_exposure(self):
        result = aggregate_sample_decode_distortion({})
        self.assertTrue(result["no_decode_exposure"])
        self.assertEqual(result["sample_attention_distortion"], 0.0)
        self.assertEqual(result["valid_step_count"], 0)
        self.assertEqual(result["valid_steps"], ())

    def test_unavailable_requested_step_never_imputed(self):
        # step 4 is requested but not present -- must be silently excluded,
        # never zero-imputed into the mean.
        steps = {1: 1.0, 2: 1.0, 8: 1.0, 16: 1.0}  # step 4 missing
        result = aggregate_sample_decode_distortion(steps)
        self.assertEqual(result["valid_steps"], (1, 2, 8, 16))
        self.assertAlmostEqual(result["sample_attention_distortion"], 1.0, places=10)  # not pulled toward 0

    def test_non_requested_step_present_is_ignored(self):
        # step 3 was somehow measured but is not in the requested set --
        # must never enter the mean.
        steps = {1: 0.0, 3: 100.0}
        result = aggregate_sample_decode_distortion(steps)
        self.assertEqual(result["valid_steps"], (1,))
        self.assertAlmostEqual(result["sample_attention_distortion"], 0.0, places=10)

    def test_requested_step_ordering_is_always_ascending(self):
        result = aggregate_sample_decode_distortion({16: 0.1, 1: 0.1, 4: 0.1}, requested_steps=(16, 1, 4, 8, 2))
        self.assertEqual(result["requested_steps"], (1, 2, 4, 8, 16))
        self.assertEqual(result["valid_steps"], (1, 4, 16))

    def test_default_requested_steps_match_preregistration(self):
        result = aggregate_sample_decode_distortion({})
        self.assertEqual(result["requested_steps"], DEFAULT_REQUESTED_STEPS)
        self.assertEqual(DEFAULT_REQUESTED_STEPS, (1, 2, 4, 8, 16))


class ShadowKVCacheTest(unittest.TestCase):
    def _prefill(self, t=5, nh=2, d=4):
        return torch.arange(t * nh * d, dtype=torch.float32).reshape(1, nh, t, d)

    def test_seed_prefill_sets_token_count(self):
        cache = ShadowKVCache()
        k = self._prefill()
        v = self._prefill() * 2
        cache.seed_prefill(k, v)
        self.assertEqual(cache.token_count, 5)
        self.assertTrue(torch.equal(cache.k(), k))
        self.assertTrue(torch.equal(cache.v(), v))

    def test_append_decode_step_order_preserved(self):
        cache = ShadowKVCache()
        k0 = self._prefill(t=3)
        cache.seed_prefill(k0, k0.clone())
        k1 = torch.full((1, 2, 1, 4), 100.0)
        k2 = torch.full((1, 2, 1, 4), 200.0)
        cache.append_decode_step(k1, k1.clone())
        cache.append_decode_step(k2, k2.clone())
        self.assertEqual(cache.token_count, 5)
        full_k = cache.k()
        self.assertTrue(torch.equal(full_k[:, :, 3:4, :], k1))
        self.assertTrue(torch.equal(full_k[:, :, 4:5, :], k2))  # exact call order preserved

    def test_double_seed_raises(self):
        cache = ShadowKVCache()
        k = self._prefill()
        cache.seed_prefill(k, k.clone())
        with self.assertRaises(ShadowCacheError):
            cache.seed_prefill(k, k.clone())

    def test_append_before_seed_raises(self):
        cache = ShadowKVCache()
        with self.assertRaises(ShadowCacheError):
            cache.append_decode_step(torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4))

    def test_multi_token_decode_append_raises(self):
        cache = ShadowKVCache()
        k = self._prefill()
        cache.seed_prefill(k, k.clone())
        with self.assertRaises(ShadowCacheError):
            cache.append_decode_step(torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4))

    def test_seeding_never_aliases_caller_tensor(self):
        # Mutating the caller's original tensor after seeding must not
        # change what the shadow cache reports -- proves defensive cloning.
        cache = ShadowKVCache()
        k = self._prefill()
        v = self._prefill()
        cache.seed_prefill(k, v)
        k.fill_(-999.0)
        self.assertFalse(torch.equal(cache.k(), k))

    def test_k_v_shape_mismatch_raises(self):
        cache = ShadowKVCache()
        with self.assertRaises(ShadowCacheError):
            cache.seed_prefill(torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 8))


class ParseKiviCacheTupleTest(unittest.TestCase):
    def test_synthetic_tuple_parses_by_field(self):
        t = ("kqt", "kfull", "kscale", "kmn", "vquant", "vfull", "vscale", "vmn", 42)
        parsed = parse_kivi_cache_tuple(t)
        self.assertIsInstance(parsed, KIVICacheState)
        self.assertEqual(parsed.key_quant_trans, "kqt")
        self.assertEqual(parsed.key_full, "kfull")
        self.assertEqual(parsed.value_mn, "vmn")
        self.assertEqual(parsed.kv_seq_len, 42)

    def test_none_fields_preserved_not_fabricated(self):
        t = (None, "kfull", None, None, None, "vfull", None, None, 5)
        parsed = parse_kivi_cache_tuple(t)
        self.assertIsNone(parsed.key_quant_trans)
        self.assertIsNone(parsed.value_quant)
        self.assertEqual(parsed.key_full, "kfull")

    def test_wrong_length_raises(self):
        with self.assertRaises(AttentionDecodeFeatureError):
            parse_kivi_cache_tuple((1, 2, 3))


class BuildDecodeFeatureRecordTest(unittest.TestCase):
    def _agg(self):
        return aggregate_sample_decode_distortion({1: 0.1, 2: 0.2})

    def test_deterministic_field_order(self):
        rec = build_decode_feature_record("lcc", 122, 0, "key", "K2/V16", self._agg(), generated_token_count=3)
        self.assertEqual(
            list(rec.keys()),
            ["task", "dataset_index", "layer_idx", "tensor_axis", "policy", "generated_token_count",
             "requested_step_count", "requested_steps", "valid_steps", "valid_step_count",
             "no_decode_exposure", "sample_attention_distortion"],
        )

    def test_invalid_axis_raises(self):
        with self.assertRaises(AttentionDecodeFeatureError):
            build_decode_feature_record("lcc", 122, 0, "bogus", "K2/V16", self._agg(), 3)

    def test_policy_identity_metadata_preserved(self):
        rec = build_decode_feature_record("lcc", 122, 0, "value", "K16/V2", self._agg(), 3)
        self.assertEqual(rec["policy"], "K16/V2")
        self.assertEqual(rec["tensor_axis"], "value")
        self.assertEqual(rec["layer_idx"], 0)
        self.assertEqual(rec["dataset_index"], 122)

    def test_incomplete_aggregation_result_raises(self):
        incomplete = dict(self._agg())
        del incomplete["valid_step_count"]
        with self.assertRaises(AttentionDecodeFeatureError):
            build_decode_feature_record("lcc", 122, 0, "key", "K2/V16", incomplete, 3)


if __name__ == "__main__":
    unittest.main()
