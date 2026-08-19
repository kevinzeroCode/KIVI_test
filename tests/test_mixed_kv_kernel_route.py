"""Section E: kernel-route tests for mixed K/V quantization.

Verifies, by counting real calls (not by comparing generated text), that:
  - K2/V16: Key quantization and Key quantized matmul ARE called;
            Value quantization and Value quantized matmul are NOT called.
  - K16/V2: the mirror image.

This directly exercises LlamaFlashAttention_KIVI (the class actually used in
production, since pred_long_bench.py always sets config.use_flash = True) on
a tiny synthetic config -- no real model weights, no dataset, no baseline
directories touched. Requires a CUDA GPU (the KIVI quant/pack kernels are
Triton/CUDA-only); this test is skipped if none is available.

Run with:
    ./.venv/bin/python -m unittest tests.test_mixed_kv_kernel_route -v
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
    import models.llama_kivi as llama_kivi


def make_config(k_bits, v_bits, group_size=16, residual_length=32):
    config = LlamaConfig(
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=1,
        intermediate_size=256,
        vocab_size=32,
        max_position_embeddings=512,
    )
    config.k_bits = k_bits
    config.v_bits = v_bits
    config.group_size = group_size
    config.residual_length = residual_length
    config.use_flash = True
    config.key_quant_chunk_size = 512
    config.value_quant_chunk_size = 512
    config.rotary_chunk_size = 512
    return config


def run_prefill_then_decode(attn, bsz=1, prefill_len=40, hidden_size=128, device="cuda", dtype=torch.float16):
    """Run one prefill forward (use_cache=True) then one decode forward,
    which is enough to exercise both the quantize-on-prefill path and the
    quantized-matmul-on-decode path for whichever side is quantized."""
    torch.manual_seed(0)
    hidden_states = torch.randn(bsz, prefill_len, hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(prefill_len, device=device).unsqueeze(0)
    with torch.no_grad():
        _, _, past_key_value = attn(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=None,
            use_cache=True,
        )
        decode_hidden = torch.randn(bsz, 1, hidden_size, device=device, dtype=dtype)
        decode_position_ids = torch.tensor([[prefill_len]], device=device)
        attn(
            hidden_states=decode_hidden,
            attention_mask=None,
            position_ids=decode_position_ids,
            past_key_value=past_key_value,
            use_cache=True,
        )


@unittest.skipUnless(CUDA_AVAILABLE, "KIVI quant/pack kernels require a CUDA GPU")
class TestKernelRoute(unittest.TestCase):
    def _spy_counts(self):
        real_quant = llama_kivi.triton_quantize_and_pack_along_last_dim
        real_bmm = llama_kivi.cuda_bmm_fA_qB_outer
        counts = {"quant_calls": 0, "bmm_calls": 0}

        def spy_quant(*args, **kwargs):
            counts["quant_calls"] += 1
            return real_quant(*args, **kwargs)

        def spy_bmm(*args, **kwargs):
            counts["bmm_calls"] += 1
            return real_bmm(*args, **kwargs)

        return counts, spy_quant, spy_bmm

    def test_k2_v16_key_quantized_value_passthrough(self):
        config = make_config(k_bits=2, v_bits=16)
        attn = llama_kivi.LlamaFlashAttention_KIVI(config, layer_idx=0).to("cuda", dtype=torch.float16).eval()
        self.assertTrue(attn.quantize_key)
        self.assertFalse(attn.quantize_value)

        counts, spy_quant, spy_bmm = self._spy_counts()
        # models.llama_kivi calls these as bare names bound at import time via
        # `from quant.new_pack import triton_quantize_and_pack_along_last_dim`
        # and `from quant.matmul import cuda_bmm_fA_qB_outer`, so patching the
        # names inside llama_kivi's own module namespace is what actually
        # intercepts the calls made from LlamaFlashAttention_KIVI.forward.
        with mock.patch.object(llama_kivi, "triton_quantize_and_pack_along_last_dim", spy_quant), \
             mock.patch.object(llama_kivi, "cuda_bmm_fA_qB_outer", spy_bmm):
            run_prefill_then_decode(attn)

        self.assertGreater(counts["quant_calls"], 0, "Key quantization should have been called")
        self.assertGreater(counts["bmm_calls"], 0, "Key quantized matmul should have been called")

    def test_k16_v2_key_passthrough_value_quantized(self):
        config = make_config(k_bits=16, v_bits=2)
        attn = llama_kivi.LlamaFlashAttention_KIVI(config, layer_idx=0).to("cuda", dtype=torch.float16).eval()
        self.assertFalse(attn.quantize_key)
        self.assertTrue(attn.quantize_value)

        counts, spy_quant, spy_bmm = self._spy_counts()
        with mock.patch.object(llama_kivi, "triton_quantize_and_pack_along_last_dim", spy_quant), \
             mock.patch.object(llama_kivi, "cuda_bmm_fA_qB_outer", spy_bmm):
            run_prefill_then_decode(attn)

        self.assertGreater(counts["quant_calls"], 0, "Value quantization should have been called")
        self.assertGreater(counts["bmm_calls"], 0, "Value quantized matmul should have been called")

    def test_k2_v16_never_quantizes_or_matmuls_value_side(self):
        # Isolate the value side specifically: patch cuda_bmm_fA_qB_outer to
        # record the `bits` argument of every call. For K2/V16 every call
        # must be bits=2 (the key side); bits=16 must never appear, since
        # that would mean the value side was incorrectly routed through the
        # quantized kernel (which only supports 2/4-bit).
        config = make_config(k_bits=2, v_bits=16)
        attn = llama_kivi.LlamaFlashAttention_KIVI(config, layer_idx=0).to("cuda", dtype=torch.float16).eval()
        real_bmm = llama_kivi.cuda_bmm_fA_qB_outer
        seen_bits = []

        def spy_bmm(group_size, fA, qB, scales, zeros, bits):
            seen_bits.append(bits)
            return real_bmm(group_size, fA, qB, scales, zeros, bits)

        real_quant = llama_kivi.triton_quantize_and_pack_along_last_dim
        seen_quant_bits = []

        def spy_quant(data, group_size, bit):
            seen_quant_bits.append(bit)
            return real_quant(data, group_size, bit)

        with mock.patch.object(llama_kivi, "cuda_bmm_fA_qB_outer", spy_bmm), \
             mock.patch.object(llama_kivi, "triton_quantize_and_pack_along_last_dim", spy_quant):
            run_prefill_then_decode(attn)

        self.assertTrue(all(b == 2 for b in seen_bits), f"unexpected bits in cuda_bmm_fA_qB_outer calls: {seen_bits}")
        self.assertTrue(all(b == 2 for b in seen_quant_bits), f"unexpected bits in quantize calls: {seen_quant_bits}")
        self.assertGreater(len(seen_bits), 0)
        self.assertGreater(len(seen_quant_bits), 0)

    def test_k16_v2_never_quantizes_or_matmuls_key_side(self):
        config = make_config(k_bits=16, v_bits=2)
        attn = llama_kivi.LlamaFlashAttention_KIVI(config, layer_idx=0).to("cuda", dtype=torch.float16).eval()
        real_bmm = llama_kivi.cuda_bmm_fA_qB_outer
        seen_bits = []

        def spy_bmm(group_size, fA, qB, scales, zeros, bits):
            seen_bits.append(bits)
            return real_bmm(group_size, fA, qB, scales, zeros, bits)

        real_quant = llama_kivi.triton_quantize_and_pack_along_last_dim
        seen_quant_bits = []

        def spy_quant(data, group_size, bit):
            seen_quant_bits.append(bit)
            return real_quant(data, group_size, bit)

        with mock.patch.object(llama_kivi, "cuda_bmm_fA_qB_outer", spy_bmm), \
             mock.patch.object(llama_kivi, "triton_quantize_and_pack_along_last_dim", spy_quant):
            run_prefill_then_decode(attn)

        self.assertTrue(all(b == 2 for b in seen_bits), f"unexpected bits in cuda_bmm_fA_qB_outer calls: {seen_bits}")
        self.assertTrue(all(b == 2 for b in seen_quant_bits), f"unexpected bits in quantize calls: {seen_quant_bits}")
        self.assertGreater(len(seen_bits), 0)
        self.assertGreater(len(seen_quant_bits), 0)

    def test_k2_v2_both_sides_still_quantized(self):
        # Regression guard: symmetric KIVI-2 must still call both kernels for
        # both sides after the pass-through branches were added.
        config = make_config(k_bits=2, v_bits=2)
        attn = llama_kivi.LlamaFlashAttention_KIVI(config, layer_idx=0).to("cuda", dtype=torch.float16).eval()
        counts, spy_quant, spy_bmm = self._spy_counts()
        with mock.patch.object(llama_kivi, "triton_quantize_and_pack_along_last_dim", spy_quant), \
             mock.patch.object(llama_kivi, "cuda_bmm_fA_qB_outer", spy_bmm):
            run_prefill_then_decode(attn)
        self.assertGreater(counts["quant_calls"], 0)
        self.assertGreater(counts["bmm_calls"], 0)

    def test_k4_v4_both_sides_still_quantized(self):
        config = make_config(k_bits=4, v_bits=4)
        attn = llama_kivi.LlamaFlashAttention_KIVI(config, layer_idx=0).to("cuda", dtype=torch.float16).eval()
        counts, spy_quant, spy_bmm = self._spy_counts()
        with mock.patch.object(llama_kivi, "triton_quantize_and_pack_along_last_dim", spy_quant), \
             mock.patch.object(llama_kivi, "cuda_bmm_fA_qB_outer", spy_bmm):
            run_prefill_then_decode(attn)
        self.assertGreater(counts["quant_calls"], 0)
        self.assertGreater(counts["bmm_calls"], 0)

    def test_k16_v16_never_quantizes_or_matmuls(self):
        config = make_config(k_bits=16, v_bits=16)
        attn = llama_kivi.LlamaFlashAttention_KIVI(config, layer_idx=0).to("cuda", dtype=torch.float16).eval()
        self.assertFalse(attn.quantize_key)
        self.assertFalse(attn.quantize_value)
        counts, spy_quant, spy_bmm = self._spy_counts()
        with mock.patch.object(llama_kivi, "triton_quantize_and_pack_along_last_dim", spy_quant), \
             mock.patch.object(llama_kivi, "cuda_bmm_fA_qB_outer", spy_bmm):
            run_prefill_then_decode(attn)
        self.assertEqual(counts["quant_calls"], 0)
        self.assertEqual(counts["bmm_calls"], 0)


if __name__ == "__main__":
    unittest.main()
