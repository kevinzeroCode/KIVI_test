"""Deterministic layer x task K/V feature-record schema (Stage G1 Part 6,
hardened in Stage G3A-hardening).

Deliberately torch-free: this module exists separately from
utils/feature_extraction.py so that --dry-run configuration tooling (e.g.
scripts/collect_layer_features.py) can report the schema without importing
torch, matching the same deferred-heavy-import discipline used by
scripts/run_layer_sensitivity_pilot.py's --dry-run path.

Token-count semantics (replaces the ambiguous single "num_tokens" field
used through Stage G3A -- an 18,062-token real lcc sample produced Key
num_tokens=18048 and Value num_tokens=17934, which was not self-explanatory
without tracing models/llama_kivi.py's prefill branch):

  input_tokens: the full prompt length submitted to the model for this
    sample (identical across every layer/axis of that sample).

  distribution_tokens: the token population the FAMILY-AGNOSTIC
    distribution features (mean/std/variance/max_abs/p50_abs/p95_abs/
    p99_abs/outlier_fraction) were computed over -- the real FP16
    representation itself (post-RoPE for Key, pre-exclusion for Value),
    BEFORE any KIVI-specific residual/quantization-subset policy is
    applied. Normally equals input_tokens (both Key and Value are derived
    directly from k_proj/v_proj output with no earlier truncation), unless
    a specific production reason changes it for some future case.

  quantized_tokens: the token population the KIVI-SPECIFIC reconstruction
    features (relative_l2/mse/max_abs_error) were computed over -- exactly
    the tensor production's real Triton quantizer actually received (the
    G2-confirmed production-parity population). Intentionally
    family-specific: Rotation-KIVI/Polar would define their own
    quantized_tokens without changing distribution_tokens.

  residual_tokens: distribution_tokens - quantized_tokens -- the tokens
    production's KIVI route kept at full precision instead of quantizing.
    For Key this is a REMAINDER (distribution_tokens % residual_length,
    kept so the quantized portion is an exact multiple of residual_length
    -- itself a multiple of group_size -- for correct grouped
    quantization); for Value this is a FIXED-SIZE sliding window (always
    exactly residual_length tokens, since V's quantizer groups by channel
    axis, not by token count, so no token-axis alignment constraint
    exists). This asymmetry is why Key and Value residual_tokens differ in
    kind, not just magnitude. For KIVI, every token is accounted for by
    exactly one of quantized_tokens/residual_tokens (no silent drop), so
    quantized_tokens + residual_tokens == distribution_tokens is enforced
    as an invariant in build_feature_record -- this is a KIVI-specific
    partition property, not assumed to hold for every future family.
"""

IDENTITY_FIELDS = ("task", "sample_idx", "layer_idx", "tensor_axis")
TOKEN_COUNT_FIELDS = ("input_tokens", "distribution_tokens", "quantized_tokens", "residual_tokens")
RECONSTRUCTION_FIELDS = ("relative_l2", "mse", "max_abs_error")
DISTRIBUTION_FIELDS = ("mean", "std", "variance", "max_abs", "p50_abs", "p95_abs", "p99_abs", "outlier_fraction")
FEATURE_RECORD_FIELDS = IDENTITY_FIELDS + TOKEN_COUNT_FIELDS + RECONSTRUCTION_FIELDS + DISTRIBUTION_FIELDS
