"""Stage I1B: deterministic normalized Sylvester-Hadamard rotation utility.

Used by the "rotation_kivi" family (models/llama_kivi.py) for the
QuaRot-inspired post-RoPE per-head Q/K rotation locked in Stage I1A/I1B:
"QuaRot-inspired post-RoPE per-head Hadamard Q/K rotation applied to KIVI
Key-cache quantization" -- NOT full QuaRot (no global hidden-state rotation,
no weight/activation quantization, no W_v/W_out Value rotation).

Deterministic, no calibration, no random signs, no learned parameters: the
matrix depends only on `n` (head_dim) -- same H for every head, every layer,
every run. Cached per (n, device, dtype) so it is materialized at most once
per distinct combination actually used, never rebuilt per decode token.
"""
import torch

_HADAMARD_CACHE = {}


class HadamardError(ValueError):
    """A Hadamard construction/application precondition was violated --
    fails closed rather than silently padding, truncating, or falling back
    to a different transform."""


def is_power_of_two(n):
    return isinstance(n, int) and not isinstance(n, bool) and n > 0 and (n & (n - 1)) == 0


def sylvester_hadamard_matrix(n, dtype=torch.float64):
    """The classic +-1 Sylvester/Walsh-Hadamard matrix of size n (n must be
    a power of two). Built via repeated doubling:
        H_1 = [1]
        H_{2k} = [[H_k, H_k], [H_k, -H_k]]
    Fully deterministic -- no randomness, no calibration. Returned
    UNnormalized (entries are +-1); callers wanting an orthonormal matrix
    should use normalized_hadamard_matrix() instead.
    """
    if not is_power_of_two(n):
        raise HadamardError(f"n={n!r} must be a positive power of two for the Sylvester Hadamard construction")
    h = torch.ones((1, 1), dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h


def normalized_hadamard_matrix(n, dtype=torch.float64):
    """H such that H @ H.T == I exactly (up to floating-point rounding):
    the Sylvester matrix scaled by 1/sqrt(n). Built in float64 for
    numerical accuracy of the scaling/orthogonality itself, regardless of
    the dtype the caller will eventually cast it to."""
    h = sylvester_hadamard_matrix(n, dtype=dtype)
    return h / (float(n) ** 0.5)


def check_orthogonality(H, atol=1e-6):
    """True iff H @ H.T is close to the identity. Computed in float64
    regardless of H's own dtype, so an FP16-cast H is still checked against
    a numerically trustworthy reference rather than compounding FP16
    rounding into the check itself."""
    n = H.shape[0]
    Hd = H.to(dtype=torch.float64)
    identity = torch.eye(n, dtype=torch.float64, device=H.device)
    return torch.allclose(Hd @ Hd.T, identity, atol=atol)


def get_normalized_hadamard(n, device=None, dtype=torch.float32):
    """Cached accessor -- returns the same tensor object for a repeated
    (n, device, dtype) request instead of rebuilding it (e.g. on every
    decode token). The cache key uses the device's string form so CPU and
    a specific CUDA device index are never conflated."""
    device = torch.device(device) if device is not None else torch.device("cpu")
    key = (n, str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is None:
        cached = normalized_hadamard_matrix(n, dtype=torch.float64).to(device=device, dtype=dtype)
        _HADAMARD_CACHE[key] = cached
    return cached


def apply_hadamard_rotation(x, H):
    """Row-vector, right-multiplication convention: x' = x @ H, applied to
    the LAST axis of `x` (head_dim). For orthogonal H, this preserves inner
    products along that axis: (x@H) @ (y@H)^T == x @ y^T (see
    tests/test_hadamard.py for the exact numerical verification). Device/
    dtype-safe: casts H to x's device/dtype if they differ (e.g. an FP16
    activation against a cached FP32 H), never the other way around, so the
    activation's own working precision is preserved."""
    if H.device != x.device or H.dtype != x.dtype:
        H = H.to(device=x.device, dtype=x.dtype)
    return torch.matmul(x, H)
