"""CPU-only tests for utils/hadamard.py (Stage I1B). No GPU, no model --
pure math verifying the deterministic normalized Sylvester-Hadamard
construction and the row-vector right-multiplication rotation convention
locked in Stage I1A/I1B.
"""
import unittest

import torch

from utils.hadamard import (
    HadamardError,
    apply_hadamard_rotation,
    check_orthogonality,
    get_normalized_hadamard,
    is_power_of_two,
    normalized_hadamard_matrix,
    sylvester_hadamard_matrix,
)


class PowerOfTwoTest(unittest.TestCase):
    def test_valid_powers(self):
        for n in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            self.assertTrue(is_power_of_two(n), n)

    def test_invalid(self):
        for n in (0, -1, -8, 3, 5, 100, 127, 129, 1.5, True, False):
            self.assertFalse(is_power_of_two(n), n)


class SylvesterConstructionTest(unittest.TestCase):
    def test_deterministic_same_matrix_every_call(self):
        a = sylvester_hadamard_matrix(128)
        b = sylvester_hadamard_matrix(128)
        self.assertTrue(torch.equal(a, b))

    def test_rejects_non_power_of_two(self):
        with self.assertRaises(HadamardError):
            sylvester_hadamard_matrix(100)

    def test_entries_are_plus_minus_one(self):
        H = sylvester_hadamard_matrix(16)
        self.assertTrue(torch.all((H == 1.0) | (H == -1.0)))

    def test_h1_base_case(self):
        H = sylvester_hadamard_matrix(1)
        self.assertEqual(tuple(H.shape), (1, 1))
        self.assertEqual(H.item(), 1.0)


class NormalizedHadamardTest(unittest.TestCase):
    def test_orthogonal_head_dim_128(self):
        H = normalized_hadamard_matrix(128)
        self.assertTrue(check_orthogonality(H))

    def test_orthogonal_several_sizes(self):
        for n in (2, 4, 8, 16, 32, 64, 128):
            H = normalized_hadamard_matrix(n)
            self.assertTrue(check_orthogonality(H), n)

    def test_normalization_scale(self):
        raw = sylvester_hadamard_matrix(64)
        normed = normalized_hadamard_matrix(64)
        self.assertTrue(torch.allclose(normed, raw / 8.0))  # sqrt(64) == 8


class GetNormalizedHadamardCacheTest(unittest.TestCase):
    def test_same_object_returned_for_same_key(self):
        a = get_normalized_hadamard(128, device="cpu", dtype=torch.float32)
        b = get_normalized_hadamard(128, device="cpu", dtype=torch.float32)
        self.assertIs(a, b)

    def test_different_dtype_returns_different_tensor(self):
        a = get_normalized_hadamard(128, device="cpu", dtype=torch.float32)
        b = get_normalized_hadamard(128, device="cpu", dtype=torch.float16)
        self.assertIsNot(a, b)
        self.assertEqual(a.dtype, torch.float32)
        self.assertEqual(b.dtype, torch.float16)

    def test_cached_matrix_is_orthogonal(self):
        H = get_normalized_hadamard(128, device="cpu", dtype=torch.float64)
        self.assertTrue(check_orthogonality(H))

    def test_rejects_non_power_of_two_head_dim(self):
        with self.assertRaises(HadamardError):
            get_normalized_hadamard(100, device="cpu", dtype=torch.float32)


class ApplyRotationConventionTest(unittest.TestCase):
    def test_row_vector_right_multiply(self):
        torch.manual_seed(0)
        H = get_normalized_hadamard(8, device="cpu", dtype=torch.float64)
        x = torch.randn(3, 5, 8, dtype=torch.float64)
        rotated = apply_hadamard_rotation(x, H)
        expected = torch.matmul(x, H)
        self.assertTrue(torch.equal(rotated, expected))

    def test_casts_h_to_input_dtype_device(self):
        H64 = get_normalized_hadamard(8, device="cpu", dtype=torch.float64)
        x32 = torch.randn(2, 8, dtype=torch.float32)
        rotated = apply_hadamard_rotation(x32, H64)
        self.assertEqual(rotated.dtype, torch.float32)


class QKInvarianceTest(unittest.TestCase):
    """QK^T vs (QH)(KH)^T, FP32/FP64/FP16, with an explicitly justified
    tolerance for FP16 (an extra real-valued matmul introduces ordinary
    floating-point rounding on top of FP16's own reduced mantissa -- not a
    correctness defect in the rotation itself)."""

    def _check(self, dtype, atol, rtol):
        torch.manual_seed(42)
        head_dim = 128
        H = get_normalized_hadamard(head_dim, device="cpu", dtype=dtype)
        Q = torch.randn(1, 4, 6, head_dim, dtype=dtype)
        K = torch.randn(1, 4, 6, head_dim, dtype=dtype)
        ref = torch.matmul(Q.double(), K.double().transpose(-2, -1))
        Qr = apply_hadamard_rotation(Q, H)
        Kr = apply_hadamard_rotation(K, H)
        rotated = torch.matmul(Qr.double(), Kr.double().transpose(-2, -1))
        max_abs = (ref - rotated).abs().max().item()
        rel_l2 = ((ref - rotated).norm() / ref.norm().clamp_min(1e-12)).item()
        self.assertLess(max_abs, atol, f"max_abs={max_abs}")
        self.assertLess(rel_l2, rtol, f"rel_l2={rel_l2}")

    def test_fp32_tight_tolerance(self):
        self._check(torch.float32, atol=1e-3, rtol=1e-4)

    def test_fp64_near_exact(self):
        self._check(torch.float64, atol=1e-9, rtol=1e-9)

    def test_fp16_looser_justified_tolerance(self):
        # FP16 carries ~3 decimal digits of precision; stacking a second
        # (head_dim x head_dim) matmul on top of FP16 Q/K accumulates more
        # rounding than FP32 does -- a looser, but still small and
        # explicitly stated, tolerance; not evidence the rotation is wrong.
        self._check(torch.float16, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    unittest.main()
