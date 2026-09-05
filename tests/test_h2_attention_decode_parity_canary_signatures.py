"""Stage H3B: signature-contract regression guard for the additive,
default-preserving parameterization added to
scripts/h2_attention_decode_parity_canary.py (load_canary_model,
generate_with_capture, build_prompt) so scripts/run_layer_attention_feature_pilot.py
can reuse them for arbitrary layers/tasks. Every new parameter's default
must equal the original hardcoded canary constant, so every pre-existing
H2 call site (which never passes these new parameters) is byte-identical
to before this round's change. This module imports torch transitively
(via utils.attention_decode_features) -- it is a CPU-only test, not part
of any --dry-run path.
"""
import inspect
import unittest

from scripts.h2_attention_decode_parity_canary import (
    CANARY_GROUP_SIZE,
    CANARY_LAYER,
    CANARY_RESIDUAL_LENGTH,
    CANARY_TASK,
    build_prompt,
    generate_with_capture,
    load_canary_model,
)


class LoadCanaryModelSignatureTest(unittest.TestCase):
    def test_new_defaults_match_original_hardcoded_constants(self):
        params = inspect.signature(load_canary_model).parameters
        self.assertEqual(params["layer_idx"].default, CANARY_LAYER)
        self.assertEqual(params["group_size"].default, CANARY_GROUP_SIZE)
        self.assertEqual(params["residual_length"].default, CANARY_RESIDUAL_LENGTH)

    def test_original_required_params_unchanged(self):
        params = list(inspect.signature(load_canary_model).parameters)
        self.assertEqual(params[:5], ["model_name_or_path", "cache_dir", "k_bits", "v_bits", "seed"])


class GenerateWithCaptureSignatureTest(unittest.TestCase):
    def test_new_defaults_match_original_hardcoded_constants(self):
        params = inspect.signature(generate_with_capture).parameters
        self.assertEqual(params["layer_idx"].default, CANARY_LAYER)
        self.assertIsNone(params["extra_generate_kwargs"].default)

    def test_original_required_params_unchanged(self):
        params = list(inspect.signature(generate_with_capture).parameters)
        self.assertEqual(params[:4], ["model", "tokenizer", "prompt", "max_new_tokens"])


class BuildPromptSignatureTest(unittest.TestCase):
    def test_new_default_matches_original_hardcoded_constant(self):
        params = inspect.signature(build_prompt).parameters
        self.assertEqual(params["task"].default, CANARY_TASK)

    def test_original_required_params_unchanged(self):
        params = list(inspect.signature(build_prompt).parameters)
        self.assertEqual(params[:3], ["tokenizer", "model_short_name", "json_obj"])


if __name__ == "__main__":
    unittest.main()
