"""CPU-only unit tests for the pure/testable logic in
scripts/layer_policy_smoke.py: layer-record extraction, validation, boot-id
retrieval, and CLI parsing. Does not import torch/transformers/datasets and
does not construct a real model -- run_smoke() itself (the GPU generation
loop) is exercised only by an actual Stage-C run, not here.

Run with:
    ./.venv/bin/python -m unittest tests.test_layer_policy_smoke -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import layer_policy_smoke as smoke  # noqa: E402


class FakeAttn:
    def __init__(self, layer_idx, k_bits, v_bits):
        self.layer_idx = layer_idx
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.quantize_key = k_bits < 16
        self.quantize_value = v_bits < 16


class FakeLayer:
    def __init__(self, layer_idx, k_bits, v_bits):
        self.self_attn = FakeAttn(layer_idx, k_bits, v_bits)


def make_layer17_policy_layers(num_layers=32):
    return [FakeLayer(i, 2 if i == 17 else 16, 16) for i in range(num_layers)]


def make_all_global_layers(num_layers=32, k_bits=16, v_bits=16):
    return [FakeLayer(i, k_bits, v_bits) for i in range(num_layers)]


class TestBuildResolvedLayerRecords(unittest.TestCase):
    def test_extracts_and_sorts_by_layer_idx(self):
        # Deliberately out of order to confirm sorting happens.
        layers = [FakeLayer(2, 16, 16), FakeLayer(0, 16, 16), FakeLayer(1, 2, 16)]
        records = smoke.build_resolved_layer_records(layers)
        self.assertEqual([r["layer_idx"] for r in records], [0, 1, 2])
        self.assertEqual(records[1], {"layer_idx": 1, "k_bits": 2, "v_bits": 16, "quantize_key": True, "quantize_value": False})


class TestValidateLayerCount(unittest.TestCase):
    def test_passes_on_32(self):
        records = smoke.build_resolved_layer_records(make_all_global_layers(32))
        smoke.validate_layer_count(records)  # must not raise

    def test_fails_on_wrong_count(self):
        records = smoke.build_resolved_layer_records(make_all_global_layers(31))
        with self.assertRaises(smoke.SmokeError):
            smoke.validate_layer_count(records)


class TestValidateLayer17Policy(unittest.TestCase):
    def test_passes_on_correct_layer17_policy(self):
        records = smoke.build_resolved_layer_records(make_layer17_policy_layers())
        smoke.validate_layer17_k2_v16_policy(records)  # must not raise

    def test_fails_if_layer17_is_not_k2_v16(self):
        layers = make_layer17_policy_layers()
        layers[17].self_attn.k_bits = 16  # corrupt layer 17 back to global
        layers[17].self_attn.quantize_key = False
        records = smoke.build_resolved_layer_records(layers)
        with self.assertRaises(smoke.SmokeError):
            smoke.validate_layer17_k2_v16_policy(records)

    def test_fails_if_another_layer_deviates(self):
        layers = make_layer17_policy_layers()
        layers[5].self_attn.k_bits = 2  # unexpected extra override
        layers[5].self_attn.quantize_key = True
        records = smoke.build_resolved_layer_records(layers)
        with self.assertRaises(smoke.SmokeError):
            smoke.validate_layer17_k2_v16_policy(records)

    def test_fails_on_wrong_layer_count(self):
        records = smoke.build_resolved_layer_records(make_layer17_policy_layers(num_layers=30))
        with self.assertRaises(smoke.SmokeError):
            smoke.validate_layer17_k2_v16_policy(records)


class FakeResolvedPolicy:
    """Minimal stand-in for utils.layer_policy.ResolvedLayerPolicy -- only
    .layers (a list of objects with k_bits/v_bits) is read."""

    class _Entry:
        def __init__(self, k_bits, v_bits):
            self.k_bits = k_bits
            self.v_bits = v_bits

    def __init__(self, layer_bits):
        self.layers = [self._Entry(k, v) for k, v in layer_bits]


class TestValidateRecordsMatchResolvedPolicy(unittest.TestCase):
    """Regression test for the bug where the harness's validation was
    hardcoded to the Key-probe pattern (k_bits=2 at layer 17) and would
    incorrectly reject a Value-probe policy (v_bits=2 at layer 17)."""

    def test_passes_for_key_axis_policy(self):
        bits = [(16, 16)] * 32
        bits[17] = (2, 16)
        records = smoke.build_resolved_layer_records([FakeLayer(i, k, v) for i, (k, v) in enumerate(bits)])
        smoke.validate_records_match_resolved_policy(records, FakeResolvedPolicy(bits))  # must not raise

    def test_passes_for_value_axis_policy(self):
        bits = [(16, 16)] * 32
        bits[17] = (16, 2)
        records = smoke.build_resolved_layer_records([FakeLayer(i, k, v) for i, (k, v) in enumerate(bits)])
        smoke.validate_records_match_resolved_policy(records, FakeResolvedPolicy(bits))  # must not raise

    def test_fails_when_model_does_not_match_resolved_policy(self):
        bits = [(16, 16)] * 32
        bits[17] = (16, 2)
        records = smoke.build_resolved_layer_records([FakeLayer(i, k, v) for i, (k, v) in enumerate(bits)])
        wrong_expected = list(bits)
        wrong_expected[17] = (2, 16)  # model built for one policy, checked against a different one
        with self.assertRaises(smoke.SmokeError):
            smoke.validate_records_match_resolved_policy(records, FakeResolvedPolicy(wrong_expected))


class TestValidateAllGlobal(unittest.TestCase):
    def test_passes_on_all_k16_v16(self):
        records = smoke.build_resolved_layer_records(make_all_global_layers(32, 16, 16))
        smoke.validate_all_global_k16_v16(records)  # must not raise

    def test_fails_if_any_layer_is_quantized(self):
        layers = make_all_global_layers(32, 16, 16)
        layers[9].self_attn.k_bits = 4
        layers[9].self_attn.quantize_key = True
        records = smoke.build_resolved_layer_records(layers)
        with self.assertRaises(smoke.SmokeError):
            smoke.validate_all_global_k16_v16(records)


class TestBootId(unittest.TestCase):
    def test_returns_string_or_none_without_crashing(self):
        result = smoke.get_boot_id()
        self.assertTrue(result is None or isinstance(result, str))


class TestArgParsing(unittest.TestCase):
    def test_defaults_match_stage_c_spec(self):
        args = smoke.parse_args([])
        self.assertEqual(args.model_name_or_path, "lmsys/longchat-7b-v1.5-32k")
        self.assertEqual(args.layer_policy, "analysis/policies/key_probe_layer17_k2.json")
        self.assertEqual(args.dataset, "hotpotqa")
        self.assertEqual(args.num_samples, 5)
        self.assertEqual(args.group_size, 32)
        self.assertEqual(args.residual_length, 128)
        self.assertEqual(args.seed, 42)

    def test_none_literal_selects_control_mode(self):
        args = smoke.parse_args(["--layer_policy", "none"])
        self.assertEqual(args.layer_policy, "none")


if __name__ == "__main__":
    unittest.main()
