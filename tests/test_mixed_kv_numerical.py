"""Phase 10 numerical tests for the 16-bit pass-through side.

Independent reference: transformers' own eager `LlamaAttention` (no KIVI at
all), with q/k/v/o projection weights copied from the KIVI attention module
under test, run once over the full (prefill + one decode token) sequence
with no causal mask (safe here because we only ever read the *last* query
position's attention row, which never needs masking -- it has no future
tokens to hide from). This gives a genuinely independent "plain FP16
torch.matmul" computation to compare against, not a value reused from
inside the module under test.

  - K16/V2: attn_weights (the softmax(QK^T/sqrt(d)) scores) at the decode
    step must be close to the reference's, because attn_weights depend only
    on Q and K, and K is full precision on both sides here.
  - K2/V16: attn_weights are allowed to differ (K is quantized), but given
    the module's own attn_weights, mapping them to the final output via V
    must match a plain matmul against the reference's full-precision V --
    i.e. the value side must not have been silently quantized.
  - K2/V2 and K4/V4 sanity checks confirm attn_weights differ measurably
    from the full-precision reference (quantization error is present, as
    expected), so the K16/V2 and K2/V16 "closeness" assertions above are not
    vacuously true.

Requires a CUDA GPU (KIVI's quant/pack path is Triton/CUDA-only); skipped if
none is available.

Run with:
    ./.venv/bin/python -m unittest tests.test_mixed_kv_numerical -v
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaAttention
    import models.llama_kivi as llama_kivi

HIDDEN_SIZE = 128
NUM_HEADS = 4
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
GROUP_SIZE = 16
RESIDUAL_LENGTH = 32
PREFILL_LEN = 40
DEVICE = "cuda"
DTYPE = torch.float16

# rtol/atol for comparisons against the independent full-precision reference.
# Full-precision-vs-full-precision (K16 attn_weights, or the V16 output
# reconstruction) should be extremely close: the only source of difference
# is fp16 op-ordering (single big matmul in the reference vs a cached
# two-step concat-then-matmul in the KIVI module), not any quantization.
CLOSE_RTOL = 1e-2
CLOSE_ATOL = 1e-2


def make_config(k_bits, v_bits):
    config = LlamaConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        num_hidden_layers=1,
        intermediate_size=256,
        vocab_size=32,
        max_position_embeddings=512,
    )
    config.k_bits = k_bits
    config.v_bits = v_bits
    config.group_size = GROUP_SIZE
    config.residual_length = RESIDUAL_LENGTH
    config.use_flash = True
    config.key_quant_chunk_size = 512
    config.value_quant_chunk_size = 512
    config.rotary_chunk_size = 512
    return config


def copy_projection_weights(src, dst):
    dst.q_proj.weight.data.copy_(src.q_proj.weight.data)
    dst.k_proj.weight.data.copy_(src.k_proj.weight.data)
    dst.v_proj.weight.data.copy_(src.v_proj.weight.data)
    dst.o_proj.weight.data.copy_(src.o_proj.weight.data)


class MixedNumericalFixture:
    """Builds a KIVI attention module and an independent plain-attention
    reference module with identical weights, and runs both over the same
    synthetic sequence."""

    def __init__(self, k_bits, v_bits, seed=0):
        torch.manual_seed(seed)
        config = make_config(k_bits, v_bits)
        self.kivi_attn = llama_kivi.LlamaFlashAttention_KIVI(config).to(DEVICE, dtype=DTYPE).eval()
        self.ref_attn = LlamaAttention(config, layer_idx=0).to(DEVICE, dtype=DTYPE).eval()
        copy_projection_weights(self.kivi_attn, self.ref_attn)

        self.full_hidden = torch.randn(1, PREFILL_LEN + 1, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
        self.prefill_hidden = self.full_hidden[:, :PREFILL_LEN, :]
        self.decode_hidden = self.full_hidden[:, PREFILL_LEN:, :]
        self.prefill_position_ids = torch.arange(PREFILL_LEN, device=DEVICE).unsqueeze(0)
        self.decode_position_ids = torch.tensor([[PREFILL_LEN]], device=DEVICE)
        self.full_position_ids = torch.arange(PREFILL_LEN + 1, device=DEVICE).unsqueeze(0)

    def run_kivi_decode(self):
        """Runs prefill + one decode step; captures the module's own
        post-softmax attn_weights for the decode step via a narrowly scoped
        softmax patch, and returns (attn_output, attn_weights)."""
        captured = {}
        real_softmax = torch.nn.functional.softmax

        def spy_softmax(input, dim=None, dtype=None):
            out = real_softmax(input, dim=dim, dtype=dtype)
            # The module casts softmax's fp32 output back to the working
            # dtype (query_states.dtype) immediately after this call before
            # using it further; capture that same post-cast value so the
            # comparison reflects what the module actually computes with.
            captured["weights"] = out.to(DTYPE)
            return out

        with torch.no_grad():
            _, _, past_key_value = self.kivi_attn(
                hidden_states=self.prefill_hidden,
                attention_mask=None,
                position_ids=self.prefill_position_ids,
                past_key_value=None,
                use_cache=True,
            )
            with mock.patch("torch.nn.functional.softmax", spy_softmax):
                attn_output, _, _ = self.kivi_attn(
                    hidden_states=self.decode_hidden,
                    attention_mask=None,
                    position_ids=self.decode_position_ids,
                    past_key_value=past_key_value,
                    use_cache=True,
                )
        return attn_output, captured["weights"]

    def run_reference_last_token_attn_weights(self):
        """Independent single-shot forward over the full sequence; returns
        the last query position's attention distribution over all kv
        positions. Safe without a causal mask: the last position has no
        future tokens to hide from regardless of masking."""
        with torch.no_grad():
            _, attn_weights, _ = self.ref_attn(
                hidden_states=self.full_hidden,
                attention_mask=None,
                position_ids=self.full_position_ids,
                past_key_value=None,
                output_attentions=True,
                use_cache=False,
            )
        return attn_weights[:, :, -1:, :]

    def reference_value_states(self):
        """Independent full-precision V, projected (no rope) from the exact
        same hidden states and copied v_proj weights."""
        with torch.no_grad():
            value_states = self.ref_attn.v_proj(self.full_hidden)
            value_states = value_states.view(1, PREFILL_LEN + 1, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        return value_states


@unittest.skipUnless(CUDA_AVAILABLE, "requires a CUDA GPU")
class TestK16V2KeyScorePath(unittest.TestCase):
    """K16/V2: Key score path must match the independent FP16 reference closely."""

    def test_attn_weights_close_to_reference(self):
        fixture = MixedNumericalFixture(k_bits=16, v_bits=2)
        _, kivi_weights = fixture.run_kivi_decode()
        ref_weights = fixture.run_reference_last_token_attn_weights()
        self.assertEqual(kivi_weights.shape, ref_weights.shape)
        torch.testing.assert_close(kivi_weights, ref_weights, rtol=CLOSE_RTOL, atol=CLOSE_ATOL)


@unittest.skipUnless(CUDA_AVAILABLE, "requires a CUDA GPU")
class TestK2V16ValueOutputPath(unittest.TestCase):
    """K2/V16: given the module's own attn_weights (with K quantization
    error baked in, which is allowed), the value-side computation must be a
    plain matmul against full-precision V, not a silently quantized one."""

    def test_output_matches_manual_matmul_with_reference_value(self):
        fixture = MixedNumericalFixture(k_bits=2, v_bits=16)
        attn_output, kivi_weights = fixture.run_kivi_decode()
        value_states_full = fixture.reference_value_states()  # (1, heads, seq, head_dim)

        manual_context = torch.matmul(kivi_weights, value_states_full)  # (1, heads, 1, head_dim)
        manual_context = manual_context.transpose(1, 2).reshape(1, 1, HIDDEN_SIZE)
        manual_output = fixture.kivi_attn.o_proj(manual_context)

        self.assertEqual(attn_output.shape, manual_output.shape)
        torch.testing.assert_close(attn_output, manual_output, rtol=CLOSE_RTOL, atol=CLOSE_ATOL)


@unittest.skipUnless(CUDA_AVAILABLE, "requires a CUDA GPU")
class TestSymmetricQuantizationStillHasError(unittest.TestCase):
    """Regression / sanity guard: K2/V2 and K4/V4 attn_weights should differ
    measurably from the full-precision reference (quantization error is
    present), so the "closeness" assertions above are not vacuous -- i.e.
    CLOSE_RTOL/CLOSE_ATOL are tight enough to actually detect quantization
    error when it's present on the Key side."""

    def _max_abs_diff(self, k_bits, v_bits):
        fixture = MixedNumericalFixture(k_bits=k_bits, v_bits=v_bits)
        _, kivi_weights = fixture.run_kivi_decode()
        ref_weights = fixture.run_reference_last_token_attn_weights()
        return (kivi_weights.float() - ref_weights.float()).abs().max().item()

    def test_k2_v2_attn_weights_differ_from_full_precision_reference(self):
        diff = self._max_abs_diff(2, 2)
        self.assertGreater(diff, CLOSE_ATOL, "K2/V2 attn_weights should show visible quantization error vs FP16")

    def test_k4_v4_attn_weights_closer_but_still_not_identical(self):
        # 4-bit should be closer to FP16 than 2-bit, but not bit-identical.
        diff_k4 = self._max_abs_diff(4, 4)
        diff_k2 = self._max_abs_diff(2, 2)
        self.assertGreater(diff_k4, 0.0)
        self.assertLess(diff_k4, diff_k2, "K4/V4 quantization error should be smaller than K2/V2's")


@unittest.skipUnless(CUDA_AVAILABLE, "requires a CUDA GPU")
class TestK16V16FullPassthroughMatchesReferenceEndToEnd(unittest.TestCase):
    """Regression guard: with both sides full precision, the KIVI module's
    decode output should match the independent reference end-to-end
    (attn_weights AND final output), since nothing is quantized anywhere."""

    def test_attn_weights_and_output_close(self):
        fixture = MixedNumericalFixture(k_bits=16, v_bits=16)
        attn_output, kivi_weights = fixture.run_kivi_decode()
        ref_weights = fixture.run_reference_last_token_attn_weights()
        torch.testing.assert_close(kivi_weights, ref_weights, rtol=CLOSE_RTOL, atol=CLOSE_ATOL)

        value_states_full = fixture.reference_value_states()
        manual_context = torch.matmul(ref_weights, value_states_full)
        manual_context = manual_context.transpose(1, 2).reshape(1, 1, HIDDEN_SIZE)
        manual_output = fixture.kivi_attn.o_proj(manual_context)
        torch.testing.assert_close(attn_output, manual_output, rtol=CLOSE_RTOL, atol=CLOSE_ATOL)


if __name__ == "__main__":
    unittest.main()
