"""Stage G1: CPU-testable feature-extraction primitives for the layer x task
x K/V-quantization-response feature audit (Stage G0 design).

Scope (deliberately limited): reconstruction-error and distribution-shape
statistics for Key/Value tensors. Attention-geometry features are NOT
implemented here (Stage G0 explicitly deferred them) -- see
ATTENTION_GEOMETRY_EXTENSION_POINT below for where they would plug in later.

Two distinct quantizers are involved, and they must never be conflated:
  - `cpu_reference_quantize_dequantize`: a pure PyTorch (no Triton), CPU-safe
    reimplementation of KIVI's per-group min-max affine quantization math,
    used ONLY to unit-test the reconstruction-error formulas on synthetic
    data. It does NOT claim numerical parity with the production kernel.
  - The actual production quantizer, `quant.new_pack.triton_quantize_and_pack_along_last_dim`,
    is Triton-JIT and CUDA-only -- it cannot run on this CPU-only test
    environment at all. Whether it is numerically compatible with this
    module's reconstruction-error functions is a separate, NOT-YET-VERIFIED
    question -- see docs/stage_g1_quant_dequant_audit.md for the exact
    finding (production dequantization never calls a separate
    unpack_and_dequant_* function at all; it's fused inside
    quant.matmul.cuda_bmm_fA_qB_outer). This module's functions operate on
    plain (already-dequantized-or-reference-dequantized) tensors, so they
    are reusable regardless of which quantizer eventually supplies x_hat --
    but no GPU-side numerical parity claim is made here.
"""
from collections import OrderedDict

import torch

from utils.feature_schema import (
    DISTRIBUTION_FIELDS,
    FEATURE_RECORD_FIELDS,
    IDENTITY_FIELDS,
    RECONSTRUCTION_FIELDS,
)

DEFAULT_EPS = 1e-12
DEFAULT_OUTLIER_K = 3.0
DEFAULT_PERCENTILES = (0.50, 0.95, 0.99)


# ---------------------------------------------------------------------------
# A. Reconstruction statistics
# ---------------------------------------------------------------------------

def relative_l2(x, x_hat, eps=DEFAULT_EPS):
    """||x - x_hat||_2 / max(||x||_2, eps). Scale-invariant reconstruction
    error; the primary reconstruction metric. eps guards the zero-tensor
    edge case (an all-zero x would otherwise divide by zero)."""
    diff_norm = torch.linalg.vector_norm((x - x_hat).to(torch.float64))
    x_norm = torch.linalg.vector_norm(x.to(torch.float64))
    return float(diff_norm / torch.clamp(x_norm, min=eps))


def mse(x, x_hat):
    """mean((x - x_hat)^2), unnormalized. Not scale-invariant -- two tensors
    with the same relative error but different magnitudes will have
    different MSE. Reported alongside relative_l2 (not as a replacement)
    because it preserves absolute-error information relative_l2 discards."""
    return float(torch.mean(((x - x_hat).to(torch.float64)) ** 2))


def max_abs_error(x, x_hat):
    """max(|x - x_hat|): the single worst-case per-element error. Catches
    outlier-driven reconstruction failure that a mean-based metric (MSE)
    can dilute across many well-reconstructed elements."""
    return float(torch.max(torch.abs((x - x_hat).to(torch.float64))))


# normalized_mse = mean((x-x_hat)^2) / mean(x^2) is deliberately NOT
# implemented as a separate metric: algebraically, normalized_mse ==
# relative_l2 ** 2 (both are the same ratio of sum-of-squares; the factor of
# N in mean() cancels in the ratio). Reporting both would be a redundant
# metric under a different name, which the design brief explicitly asked to
# avoid. Callers who want it can compute relative_l2(...) ** 2.


def reconstruction_stats(x, x_hat, eps=DEFAULT_EPS):
    """x, x_hat: any two tensors of identical shape (original vs.
    reconstructed). Returns the 3 non-redundant reconstruction metrics."""
    if x.shape != x_hat.shape:
        raise ValueError(f"shape mismatch: x={tuple(x.shape)} x_hat={tuple(x_hat.shape)}")
    return OrderedDict(
        [
            ("relative_l2", relative_l2(x, x_hat, eps)),
            ("mse", mse(x, x_hat)),
            ("max_abs_error", max_abs_error(x, x_hat)),
        ]
    )


# ---------------------------------------------------------------------------
# B. Distribution statistics
# ---------------------------------------------------------------------------

def distribution_stats(x, outlier_k=DEFAULT_OUTLIER_K, percentiles=DEFAULT_PERCENTILES):
    """x: any tensor (flattened internally -- caller decides what axis/slice
    to pass in; this function does not itself average across incompatible
    dimensions). All stats computed in float64 for numerical stability
    regardless of x's input dtype (production K/V tensors are fp16).

    Definitions:
      mean, std, variance: population statistics (std uses unbiased=False,
        i.e. divide by N not N-1 -- matches numpy's default and is the
        appropriate convention for "this is the whole population of values
        in this tensor", not a sample estimate of a larger population).
      max_abs: max(|x|).
      p50_abs / p95_abs / p99_abs: percentiles of |x| (magnitude
        distribution, not signed distribution -- captures tail/outlier
        behavior directly relevant to quantization range).
      outlier_fraction: fraction of elements with |x| > outlier_k * std.
        0.0 (not NaN/error) when std == 0 (a constant tensor has no
        outliers by this definition).
    """
    flat = x.reshape(-1).to(torch.float64)
    mean = float(torch.mean(flat))
    std = float(torch.std(flat, unbiased=False))
    variance = std * std
    abs_flat = torch.abs(flat)
    max_abs = float(torch.max(abs_flat))

    q = torch.quantile(abs_flat, torch.tensor(percentiles, dtype=torch.float64, device=abs_flat.device))
    percentile_stats = OrderedDict(
        (f"p{int(round(p * 100))}_abs", float(v)) for p, v in zip(percentiles, q)
    )

    if std > 0:
        outlier_fraction = float(torch.mean((abs_flat > outlier_k * std).to(torch.float64)))
    else:
        outlier_fraction = 0.0

    result = OrderedDict([("mean", mean), ("std", std), ("variance", variance), ("max_abs", max_abs)])
    result.update(percentile_stats)
    result["outlier_fraction"] = outlier_fraction
    return result


# ---------------------------------------------------------------------------
# CPU reference quantizer -- math-only, NOT a claim of Triton kernel parity.
# ---------------------------------------------------------------------------

def cpu_reference_quantize_dequantize(x, group_size, bits, dim=-1):
    """Pure PyTorch (CPU-safe, no Triton) per-group affine min-max
    quantize-then-dequantize round trip, mirroring the mathematical scheme
    in quant/new_pack.py's quant_and_pack_kcache/quant_and_pack_vcache
    (grouped min-max affine quantization). Deliberately skips integer
    bit-packing (irrelevant to reconstruction-error math) and works
    directly with the dequantized float result.

    x: any tensor whose size along `dim` is divisible by group_size.
    Returns a tensor of the same shape as x.

    Zero-range groups (all-equal values within a group) are handled by
    substituting scale=1 for that group only (avoids divide-by-zero); the
    quantized code becomes 0 for that group, and dequantization reproduces
    the group's constant value exactly (0 * 1 + mn == mn).

    NOT a claim of parity with quant.new_pack.triton_quantize_and_pack_along_last_dim
    (Triton-JIT, CUDA-only, cannot run here). Use only to test reconstruction-error
    math on synthetic CPU tensors.
    """
    if x.shape[dim] % group_size != 0:
        raise ValueError(f"dim {dim} size {x.shape[dim]} is not divisible by group_size {group_size}")
    orig_dtype = x.dtype
    x64 = x.to(torch.float64)
    x_moved = x64.movedim(dim, -1)
    grouped_shape = x_moved.shape[:-1] + (x_moved.shape[-1] // group_size, group_size)
    grouped = x_moved.reshape(grouped_shape)

    mn = grouped.amin(dim=-1, keepdim=True)
    mx = grouped.amax(dim=-1, keepdim=True)
    max_int = 2 ** bits - 1
    scale = (mx - mn) / max_int
    scale_safe = torch.where(scale == 0, torch.ones_like(scale), scale)

    q = torch.clamp(torch.round((grouped - mn) / scale_safe), 0, max_int)
    dequant = q * scale_safe + mn

    dequant = dequant.reshape(x_moved.shape).movedim(-1, dim)
    return dequant.to(orig_dtype)


# ---------------------------------------------------------------------------
# Attention-geometry extension point (NOT implemented in Stage G1 -- Stage G0
# explicitly deferred this). A future addition would live here as a
# function like `attention_geometry_stats(attn_probs)` operating on a
# captured [B, nh, q_len, kv_seq_len] probability tensor, obtained via the
# smoke-only nn.functional.softmax monkeypatch design documented in the
# Stage G0 audit -- never by disabling FlashAttention globally.
# ---------------------------------------------------------------------------
ATTENTION_GEOMETRY_EXTENSION_POINT = None


# ---------------------------------------------------------------------------
# Feature record assembly (Part 6). Schema constants (IDENTITY_FIELDS,
# RECONSTRUCTION_FIELDS, DISTRIBUTION_FIELDS, FEATURE_RECORD_FIELDS) live in
# utils/feature_schema.py (torch-free) and are re-exported here for
# backward-compatible access from this module.
# ---------------------------------------------------------------------------

def build_feature_record(task, sample_idx, layer_idx, tensor_axis, num_tokens, recon_stats, dist_stats, aggregation="mean_over_tokens_and_heads"):
    """Assembles one deterministic feature record row.

    tensor_axis: "key" or "value" -- preserves K vs V identity explicitly
    (never merged into one row).

    aggregation: MUST be recorded explicitly whenever recon_stats/dist_stats
    were computed after reducing over tokens/heads/channels -- this
    function refuses to guess or default-hide that away; the caller must
    state how the reduction was done (e.g. "mean_over_tokens_and_heads",
    "per_head_then_mean", "no_aggregation_single_group"). This is metadata
    only (not re-derived here) -- computing the reduction correctly is the
    caller's responsibility.
    """
    if tensor_axis not in ("key", "value"):
        raise ValueError(f"tensor_axis must be 'key' or 'value', got {tensor_axis!r}")
    missing_recon = [f for f in RECONSTRUCTION_FIELDS if f not in recon_stats]
    missing_dist = [f for f in DISTRIBUTION_FIELDS if f not in dist_stats]
    if missing_recon or missing_dist:
        raise ValueError(f"incomplete stats: missing recon={missing_recon} missing dist={missing_dist}")

    record = OrderedDict(
        [
            ("task", task),
            ("sample_idx", sample_idx),
            ("layer_idx", layer_idx),
            ("tensor_axis", tensor_axis),
            ("num_tokens", num_tokens),
            ("aggregation", aggregation),
        ]
    )
    for f in RECONSTRUCTION_FIELDS:
        record[f] = recon_stats[f]
    for f in DISTRIBUTION_FIELDS:
        record[f] = dist_stats[f]
    return record
