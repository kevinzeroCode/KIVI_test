"""CPU-only unit tests for Stage A of the layer-wise KV sensitivity project:
utils/layer_policy.py, its wiring into pred_long_bench.py (directory naming,
run_config.json resume identity), and models/llama_kivi.py's layer_idx /
per-layer bit resolution.

No GPU, no model download, no dataset download, no flash-attn import
required: LlamaAttention_KIVI.__init__ builds nn.Linear/nn.Embedding layers
and reads config -- only its forward() needs CUDA/flash-attn (lazily
imported there, not here).

Run with:
    ./.venv/bin/python -m unittest tests.test_layer_policy -v
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.layer_policy import (  # noqa: E402
    LayerPolicyError,
    canonical_policy_dict,
    parse_policy_json,
    policy_hash,
    resolve_layer_policy,
)

import pred_long_bench as plb  # noqa: E402


def make_policy_obj(policy_name="test_policy", default=(16, 16), overrides=None):
    d = {"k_bits": default[0], "v_bits": default[1], "family": "kivi"}
    ov = {}
    for layer_idx, (k, v) in (overrides or {}).items():
        ov[str(layer_idx)] = {"k_bits": k, "v_bits": v, "family": "kivi"}
    return {"policy_name": policy_name, "default": d, "overrides": ov}


class FakeModelArgs:
    def __init__(self, k_bits=2, v_bits=4, model_name_or_path="lmsys/longchat-7b-v1.5-32k",
                 group_size=32, residual_length=128):
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.model_name_or_path = model_name_or_path
        self.group_size = group_size
        self.residual_length = residual_length


# --- A. no-policy global fallback ------------------------------------------

class TestGlobalFallback(unittest.TestCase):
    def test_no_policy_resolves_to_global_on_every_layer(self):
        resolved = resolve_layer_policy(32, global_k_bits=2, global_v_bits=4, policy_obj=None)
        self.assertEqual(len(resolved.layers), 32)
        self.assertEqual(resolved.source, "global_fallback")
        for entry in resolved.layers:
            self.assertEqual((entry.k_bits, entry.v_bits, entry.family), (2, 4, "kivi"))


# --- B. one override ---------------------------------------------------------

class TestSingleOverride(unittest.TestCase):
    def test_layer_17_override_only_layer_17_differs(self):
        policy_obj = make_policy_obj(default=(16, 16), overrides={17: (2, 16)})
        resolved = resolve_layer_policy(32, 16, 16, policy_obj)
        self.assertEqual(len(resolved.layers), 32)
        for i, entry in enumerate(resolved.layers):
            if i == 17:
                self.assertEqual((entry.k_bits, entry.v_bits), (2, 16))
            else:
                self.assertEqual((entry.k_bits, entry.v_bits), (16, 16))


# --- C. multiple overrides ---------------------------------------------------

class TestMultipleOverrides(unittest.TestCase):
    def test_several_layers_overridden_independently(self):
        policy_obj = make_policy_obj(default=(16, 16), overrides={0: (2, 16), 15: (16, 2), 31: (4, 4)})
        resolved = resolve_layer_policy(32, 16, 16, policy_obj)
        expected = {0: (2, 16), 15: (16, 2), 31: (4, 4)}
        for i, entry in enumerate(resolved.layers):
            want = expected.get(i, (16, 16))
            self.assertEqual((entry.k_bits, entry.v_bits), want, f"layer {i}")


# --- D. invalid bit width rejected -------------------------------------------

class TestInvalidBitWidth(unittest.TestCase):
    def test_invalid_default_bits_rejected(self):
        policy_obj = make_policy_obj(default=(8, 16))
        with self.assertRaises(LayerPolicyError):
            resolve_layer_policy(32, 16, 16, policy_obj)

    def test_invalid_override_bits_rejected(self):
        policy_obj = make_policy_obj(default=(16, 16), overrides={3: (3, 16)})
        with self.assertRaises(LayerPolicyError):
            resolve_layer_policy(32, 16, 16, policy_obj)

    def test_invalid_global_fallback_bits_rejected(self):
        with self.assertRaises(LayerPolicyError):
            resolve_layer_policy(32, global_k_bits=8, global_v_bits=16, policy_obj=None)


# --- E. negative layer index rejected ----------------------------------------

class TestNegativeLayerIndex(unittest.TestCase):
    def test_negative_index_rejected(self):
        policy_obj = make_policy_obj(default=(16, 16), overrides={-1: (2, 16)})
        with self.assertRaises(LayerPolicyError):
            resolve_layer_policy(32, 16, 16, policy_obj)


# --- F. layer >= num_hidden_layers rejected ----------------------------------

class TestOutOfRangeLayerIndex(unittest.TestCase):
    def test_index_equal_to_num_layers_rejected(self):
        policy_obj = make_policy_obj(default=(16, 16), overrides={32: (2, 16)})
        with self.assertRaises(LayerPolicyError):
            resolve_layer_policy(32, 16, 16, policy_obj)

    def test_index_far_out_of_range_rejected(self):
        policy_obj = make_policy_obj(default=(16, 16), overrides={999: (2, 16)})
        with self.assertRaises(LayerPolicyError):
            resolve_layer_policy(32, 16, 16, policy_obj)


# --- G. malformed JSON rejected ----------------------------------------------

class TestMalformedJSON(unittest.TestCase):
    def test_broken_json_syntax_rejected(self):
        with self.assertRaises(LayerPolicyError):
            parse_policy_json("{not valid json")

    def test_json_array_root_rejected(self):
        with self.assertRaises(LayerPolicyError):
            parse_policy_json("[1, 2, 3]")

    def test_missing_top_level_field_rejected(self):
        policy_obj = {"policy_name": "x", "default": {"k_bits": 16, "v_bits": 16, "family": "kivi"}}
        with self.assertRaises(LayerPolicyError):
            resolve_layer_policy(32, 16, 16, policy_obj)

    def test_missing_cell_field_rejected(self):
        policy_obj = {"policy_name": "x", "default": {"k_bits": 16, "v_bits": 16}, "overrides": {}}
        with self.assertRaises(LayerPolicyError):
            resolve_layer_policy(32, 16, 16, policy_obj)

    def test_non_integer_override_key_rejected(self):
        policy_obj = {
            "policy_name": "x",
            "default": {"k_bits": 16, "v_bits": 16, "family": "kivi"},
            "overrides": {"seventeen": {"k_bits": 2, "v_bits": 16, "family": "kivi"}},
        }
        with self.assertRaises(LayerPolicyError):
            resolve_layer_policy(32, 16, 16, policy_obj)


# --- H. unsupported family rejected ------------------------------------------

class TestUnsupportedFamily(unittest.TestCase):
    def test_unsupported_default_family_rejected(self):
        # "polar" remains a schema placeholder only (Stage I1B added
        # "rotation_kivi" as executable; "polar" is still explicitly not).
        policy_obj = {
            "policy_name": "x",
            "default": {"k_bits": 16, "v_bits": 16, "family": "polar"},
            "overrides": {},
        }
        with self.assertRaises(LayerPolicyError):
            resolve_layer_policy(32, 16, 16, policy_obj)

    def test_unsupported_override_family_rejected(self):
        policy_obj = {
            "policy_name": "x",
            "default": {"k_bits": 16, "v_bits": 16, "family": "kivi"},
            "overrides": {"5": {"k_bits": 2, "v_bits": 16, "family": "polar"}},
        }
        with self.assertRaises(LayerPolicyError):
            resolve_layer_policy(32, 16, 16, policy_obj)


# --- I. deterministic canonical policy/hash ----------------------------------

class TestDeterministicHash(unittest.TestCase):
    def test_same_policy_same_hash(self):
        policy_obj = make_policy_obj(default=(16, 16), overrides={17: (2, 16)})
        r1 = resolve_layer_policy(32, 16, 16, policy_obj)
        r2 = resolve_layer_policy(32, 16, 16, json.loads(json.dumps(policy_obj)))
        self.assertEqual(policy_hash(r1), policy_hash(r2))
        self.assertEqual(canonical_policy_dict(r1)["layers"], canonical_policy_dict(r2)["layers"])

    def test_hash_is_deterministic_across_calls(self):
        policy_obj = make_policy_obj(default=(16, 16), overrides={3: (4, 4), 9: (2, 2)})
        resolved = resolve_layer_policy(32, 16, 16, policy_obj)
        hashes = {policy_hash(resolved) for _ in range(5)}
        self.assertEqual(len(hashes), 1)


# --- J. policy A != policy B for resume identity -----------------------------

class TestPolicyIdentityDiffers(unittest.TestCase):
    def test_different_override_layer_differs_hash(self):
        resolved_a = resolve_layer_policy(32, 16, 16, make_policy_obj(default=(16, 16), overrides={17: (2, 16)}))
        resolved_b = resolve_layer_policy(32, 16, 16, make_policy_obj(default=(16, 16), overrides={18: (2, 16)}))
        self.assertNotEqual(policy_hash(resolved_a), policy_hash(resolved_b))

    def test_same_policy_name_different_content_differs_hash(self):
        # Same human-chosen policy_name, genuinely different resolved bits --
        # must not be treated as the same policy for resume purposes.
        resolved_a = resolve_layer_policy(32, 16, 16, make_policy_obj(policy_name="probe", overrides={17: (2, 16)}))
        resolved_b = resolve_layer_policy(32, 16, 16, make_policy_obj(policy_name="probe", overrides={17: (4, 16)}))
        self.assertNotEqual(policy_hash(resolved_a), policy_hash(resolved_b))

    def test_global_run_config_never_matches_layer_policy_run_config(self):
        model_args = FakeModelArgs(k_bits=2, v_bits=16)
        global_run_config = plb.build_run_config(model_args, 31500, "LlamaForCausalLM_KIVI", True, False, seed=42)

        resolved = resolve_layer_policy(32, 2, 16, make_policy_obj(default=(2, 16), overrides={17: (2, 16)}))
        layer_run_config = plb.build_run_config(
            model_args, 31500, "LlamaForCausalLM_KIVI", None, None, seed=42, resolved_layer_policy=resolved
        )

        self.assertIsNone(global_run_config["layer_policy_hash"])
        self.assertIsNotNone(layer_run_config["layer_policy_hash"])
        mismatches = {
            key: (global_run_config.get(key), layer_run_config.get(key))
            for key in plb.RUN_CONFIG_CORE_KEYS
            if global_run_config.get(key) != layer_run_config.get(key)
        }
        self.assertIn("layer_policy_hash", mismatches)

    def test_prepare_run_directory_refuses_resume_across_different_policies(self):
        model_args = FakeModelArgs(k_bits=16, v_bits=16)
        resolved_a = resolve_layer_policy(32, 16, 16, make_policy_obj(policy_name="probe", overrides={17: (2, 16)}))
        resolved_b = resolve_layer_policy(32, 16, 16, make_policy_obj(policy_name="probe", overrides={17: (4, 16)}))

        with tempfile.TemporaryDirectory() as tmp:
            pred_dir = os.path.join(tmp, "some_layerpolicy_dir")
            run_config_a = plb.build_run_config(
                model_args, 31500, "LlamaForCausalLM_KIVI", None, None, seed=42, resolved_layer_policy=resolved_a
            )
            status = plb.prepare_run_directory(pred_dir, run_config_a)
            self.assertEqual(status, "created")

            run_config_b = plb.build_run_config(
                model_args, 31500, "LlamaForCausalLM_KIVI", None, None, seed=42, resolved_layer_policy=resolved_b
            )
            with self.assertRaises(RuntimeError):
                plb.prepare_run_directory(pred_dir, run_config_b)


# --- K. global-run directory naming remains byte-for-byte compatible --------

class TestGlobalDirectoryNamingUnchanged(unittest.TestCase):
    def test_symmetric_and_mixed_global_names_unchanged(self):
        self.assertEqual(
            plb.build_pred_dir_name("longchat-7b-v1.5-32k", 31500, 16, 16, 32, 128),
            "longchat-7b-v1.5-32k_31500_16bits_group32_residual128",
        )
        self.assertEqual(
            plb.build_pred_dir_name("longchat-7b-v1.5-32k", 31500, 2, 4, 32, 128),
            "longchat-7b-v1.5-32k_31500_k2_v4_group32_residual128",
        )


# --- L. layer-policy directory names cannot collide with global directories -

class TestLayerPolicyDirectoryNoCollision(unittest.TestCase):
    def test_layer_policy_name_never_equals_any_global_name(self):
        model_name, max_length, group_size, residual_length = "longchat-7b-v1.5-32k", 31500, 32, 128
        global_names = {
            plb.build_pred_dir_name(model_name, max_length, k, v, group_size, residual_length)
            for k in (2, 4, 16) for v in (2, 4, 16)
        }
        resolved = resolve_layer_policy(32, 16, 16, make_policy_obj(policy_name="16bits", overrides={17: (2, 16)}))
        layer_name = plb.build_layer_policy_pred_dir_name(
            model_name, max_length, resolved.policy_name, policy_hash(resolved), group_size, residual_length
        )
        self.assertNotIn(layer_name, global_names)
        self.assertTrue(layer_name.startswith(f"{model_name}_{max_length}_layerpolicy_"))

    def test_two_different_policies_never_share_a_directory_name(self):
        model_name, max_length, group_size, residual_length = "longchat-7b-v1.5-32k", 31500, 32, 128
        resolved_a = resolve_layer_policy(32, 16, 16, make_policy_obj(policy_name="probe", overrides={17: (2, 16)}))
        resolved_b = resolve_layer_policy(32, 16, 16, make_policy_obj(policy_name="probe", overrides={18: (2, 16)}))
        name_a = plb.build_layer_policy_pred_dir_name(
            model_name, max_length, resolved_a.policy_name, policy_hash(resolved_a), group_size, residual_length
        )
        name_b = plb.build_layer_policy_pred_dir_name(
            model_name, max_length, resolved_b.policy_name, policy_hash(resolved_b), group_size, residual_length
        )
        self.assertNotEqual(name_a, name_b)


# --- M/N. attention instance attributes + quantize_key/quantize_value -------

def _make_tiny_llama_config(num_hidden_layers=4):
    from transformers import LlamaConfig
    config = LlamaConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
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


class TestAttentionInstanceAttributes(unittest.TestCase):
    """Section M: different layers receive different self.k_bits/v_bits
    when a layer_kv_policy is set on the config, constructed CPU-only
    (LlamaAttention_KIVI.__init__ needs no CUDA/flash-attn)."""

    def test_layers_receive_distinct_bits_from_policy(self):
        import models.llama_kivi as llama_kivi

        config = _make_tiny_llama_config(num_hidden_layers=4)
        config.k_bits = 16  # global fallback values -- must be ignored where policy covers a layer
        config.v_bits = 16
        resolved = resolve_layer_policy(4, 16, 16, make_policy_obj(default=(16, 16), overrides={2: (2, 4)}))
        config.layer_kv_policy = [
            {"k_bits": e.k_bits, "v_bits": e.v_bits, "family": e.family} for e in resolved.layers
        ]

        attns = [llama_kivi.LlamaAttention_KIVI(config, layer_idx=i) for i in range(4)]
        for i, attn in enumerate(attns):
            self.assertEqual(attn.layer_idx, i)
            if i == 2:
                self.assertEqual((attn.k_bits, attn.v_bits), (2, 4))
            else:
                self.assertEqual((attn.k_bits, attn.v_bits), (16, 16))

    def test_no_policy_all_layers_get_global_bits(self):
        import models.llama_kivi as llama_kivi

        config = _make_tiny_llama_config(num_hidden_layers=3)
        config.k_bits = 2
        config.v_bits = 4
        # config.layer_kv_policy deliberately left unset -- must reproduce
        # today's global-only behavior exactly.
        attns = [llama_kivi.LlamaAttention_KIVI(config, layer_idx=i) for i in range(3)]
        for attn in attns:
            self.assertEqual((attn.k_bits, attn.v_bits), (2, 4))


class TestQuantizeFlags(unittest.TestCase):
    """Section N: quantize_key/quantize_value derived correctly per layer."""

    def test_quantize_flags_for_each_combination(self):
        import models.llama_kivi as llama_kivi

        cases = [
            ((16, 16), (False, False)),
            ((2, 16), (True, False)),
            ((16, 2), (False, True)),
            ((4, 4), (True, True)),
        ]
        config = _make_tiny_llama_config(num_hidden_layers=1)
        for (k_bits, v_bits), (expect_qk, expect_qv) in cases:
            with self.subTest(k_bits=k_bits, v_bits=v_bits):
                config.k_bits = k_bits
                config.v_bits = v_bits
                if hasattr(config, "layer_kv_policy"):
                    del config.layer_kv_policy
                attn = llama_kivi.LlamaAttention_KIVI(config, layer_idx=0)
                self.assertEqual(attn.quantize_key, expect_qk)
                self.assertEqual(attn.quantize_value, expect_qv)

    def test_quantize_flags_derived_in_canonical_policy_dict(self):
        resolved = resolve_layer_policy(4, 16, 16, make_policy_obj(default=(16, 16), overrides={1: (2, 16), 2: (16, 2), 3: (4, 4)}))
        layers = canonical_policy_dict(resolved)["layers"]
        expected = {0: (False, False), 1: (True, False), 2: (False, True), 3: (True, True)}
        for layer in layers:
            self.assertEqual((layer["quantize_key"], layer["quantize_value"]), expected[layer["layer_idx"]])


if __name__ == "__main__":
    unittest.main()
