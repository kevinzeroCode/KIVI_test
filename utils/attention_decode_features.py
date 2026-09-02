"""Stage H0-PRE decode-aware attention distortion measurement infrastructure.

Implements ONLY the pure metric math, step-aggregation logic, and the
measurement-only shadow-cache bookkeeping locked in
docs/stage_h0_pre_registration.md. Does NOT implement the real scientific
collector, does NOT collect any feature, does NOT compute a correlation.

Every mathematical function here operates on explicit tensors supplied by
the caller -- none of them look up production state (a hook, a
past_key_value tuple, a model) themselves. Production-state capture is the
canary/collector's responsibility (see
scripts/h2_attention_decode_parity_canary.py); this module is pure
measurement math plus deterministic bookkeeping, fully testable on CPU
with synthetic tensors.
"""
from collections import OrderedDict, namedtuple

import torch

DEFAULT_EPS = 1e-12
DEFAULT_REQUESTED_STEPS = (1, 2, 4, 8, 16)


class ShadowCacheError(RuntimeError):
    """Any misuse of ShadowKVCache (wrong call order, wrong shape) --
    fails closed rather than silently accepting malformed input."""


class AttentionDecodeFeatureError(ValueError):
    """Malformed input to a pure metric/record function in this module."""


# ---------------------------------------------------------------------------
# Part 3/4: pure Key/Value decode-distortion metrics
# ---------------------------------------------------------------------------

def _relative_frobenius_error(perturbed, reference, eps=DEFAULT_EPS):
    """Shared core of both decode-distortion metrics:
    ||perturbed - reference||_F / max(||reference||_F, eps), computed in
    float64 regardless of input dtype (production tensors are fp16) for
    numerical stability. Deterministic, non-negative by construction
    (a norm ratio), exactly 0.0 when perturbed == reference. Does not
    inspect axis/policy/production state -- callers
    (key_decode_distortion / value_decode_distortion) are responsible for
    supplying the semantically-correct tensors.
    """
    if perturbed.shape != reference.shape:
        raise AttentionDecodeFeatureError(f"shape mismatch: {tuple(perturbed.shape)} vs {tuple(reference.shape)}")
    diff_norm = torch.linalg.vector_norm((perturbed - reference).to(torch.float64))
    ref_norm = torch.linalg.vector_norm(reference.to(torch.float64))
    ref_norm_safe = torch.clamp(ref_norm, min=eps)
    return float(diff_norm / ref_norm_safe)


def key_decode_distortion(L_kivi, L_fp16, eps=DEFAULT_EPS):
    """KeyDecodeDistortion_t = ||L_kivi - L_fp16||_F / max(||L_fp16||_F, eps)
    (docs/stage_h0_pre_registration.md Section 5).

    L_kivi: the reconstructed production Key-only attention logits
        (quantized-prefix + FP16-residual columns, concatenated in
        production's own column order, already divided by sqrt(head_dim)).
    L_fp16: the same-shape logits computed against the FP16 shadow Key
        history instead.

    Both must be supplied by the caller as explicit tensors -- this
    function performs no production-state lookup itself.
    """
    return _relative_frobenius_error(L_kivi, L_fp16, eps)


def value_decode_distortion(O_value_kivi, O_fp16, eps=DEFAULT_EPS):
    """ValueDecodeDistortion_t = ||O_value_kivi - O_fp16||_F / max(||O_fp16||_F, eps)
    (docs/stage_h0_pre_registration.md Section 6).

    O_value_kivi: attention output computed with the SAME FP16-Key-derived
        attention weights but the actual quantized+residual Value cache.
    O_fp16: attention output computed with the same attention weights
        against the FP16 shadow Value history instead.
    """
    return _relative_frobenius_error(O_value_kivi, O_fp16, eps)


# ---------------------------------------------------------------------------
# Part 5: sample step aggregation (requested-step / early-EOS / no-exposure)
# ---------------------------------------------------------------------------

def aggregate_sample_decode_distortion(step_distortions, requested_steps=DEFAULT_REQUESTED_STEPS):
    """docs/stage_h0_pre_registration.md Section 8, exact rule.

    step_distortions: dict[int cached_decode_step] -> float distortion,
    for whichever cached decode steps were ACTUALLY executed and measured
    (may be a subset OR superset of requested_steps -- this function
    always performs the intersection itself; it never assumes the caller
    pre-filtered, and it never uses a step outside requested_steps even if
    one happens to be present in step_distortions).

    valid_steps = requested_steps ∩ step_distortions.keys()

    If valid_steps is non-empty: equal-step arithmetic mean over exactly
    those steps' distortion values -- an unavailable requested step (e.g.
    requested includes 4 but only steps {1,2} occurred) is simply excluded
    from the mean, never zero-imputed.

    If valid_steps is empty (the sample never reached even cached decode
    step 1): this is the pre-registered zero-exposure case, NOT a missing
    value -- no_decode_exposure=True, sample_attention_distortion=0.0,
    valid_step_count=0. The caller must still record this observation, not
    drop or replace the sample.
    """
    requested = tuple(sorted(set(requested_steps)))
    valid_steps = tuple(s for s in requested if s in step_distortions)

    if valid_steps:
        values = [step_distortions[s] for s in valid_steps]
        sample_attention_distortion = sum(values) / len(values)
        no_decode_exposure = False
    else:
        sample_attention_distortion = 0.0
        no_decode_exposure = True

    return OrderedDict(
        [
            ("requested_step_count", len(requested)),
            ("requested_steps", requested),
            ("valid_steps", valid_steps),
            ("valid_step_count", len(valid_steps)),
            ("no_decode_exposure", no_decode_exposure),
            ("sample_attention_distortion", sample_attention_distortion),
        ]
    )


# ---------------------------------------------------------------------------
# Part 6: measurement-only shadow K/V cache (one probed layer)
# ---------------------------------------------------------------------------

class ShadowKVCache:
    """Measurement-only, observational FP16 K/V history for ONE probed
    layer. Never touches or mutates production's real `past_key_value` --
    the caller feeds it tensors captured via read-only hooks; this class
    always defensively `.detach().clone()`s on ingest, so it can never
    later reflect (or be corrupted by) an in-place mutation performed by
    production code on the tensor object it was handed, and production
    can never be affected by anything this class does. Nothing here is
    ever written to disk.
    """

    def __init__(self):
        self._k_chunks = []
        self._v_chunks = []
        self._token_count = 0

    @property
    def token_count(self):
        return self._token_count

    def seed_prefill(self, k_full, v_full):
        """k_full/v_full: (B, nh, T_prefill, head_dim) full-precision
        post-RoPE Key and raw Value from the prefill forward. Must be
        called exactly once, before any append_decode_step call.
        """
        if self._token_count != 0:
            raise ShadowCacheError("seed_prefill called after the shadow cache already has content")
        if k_full.shape[-2] == 0:
            raise ShadowCacheError("seed_prefill called with zero prefill tokens")
        if k_full.shape != v_full.shape:
            raise ShadowCacheError(f"K/V prefill shape mismatch: {tuple(k_full.shape)} vs {tuple(v_full.shape)}")
        self._k_chunks = [k_full.detach().clone()]
        self._v_chunks = [v_full.detach().clone()]
        self._token_count = k_full.shape[-2]

    def append_decode_step(self, k_t, v_t):
        """k_t/v_t: (B, nh, 1, head_dim) the current cached-decode step's
        post-RoPE Key and raw Value. Must be called in the exact order
        decode steps actually occur -- this class trusts call order for
        token-position alignment, it does not independently verify it
        (the canary's parity checks verify alignment against production).
        """
        if self._token_count == 0:
            raise ShadowCacheError("append_decode_step called before seed_prefill")
        if k_t.shape[-2] != 1 or v_t.shape[-2] != 1:
            raise ShadowCacheError(
                f"expected a single-token decode append, got K shape {tuple(k_t.shape)}, V shape {tuple(v_t.shape)}"
            )
        if k_t.shape != v_t.shape:
            raise ShadowCacheError(f"K/V decode-step shape mismatch: {tuple(k_t.shape)} vs {tuple(v_t.shape)}")
        self._k_chunks.append(k_t.detach().clone())
        self._v_chunks.append(v_t.detach().clone())
        self._token_count += 1

    def k(self):
        """Full FP16 shadow Key history, position-ordered exactly as
        ingested (prefill first, then each decode step in append order)."""
        if not self._k_chunks:
            raise ShadowCacheError("k() called on an empty shadow cache (seed_prefill was never called)")
        return self._k_chunks[0] if len(self._k_chunks) == 1 else torch.cat(self._k_chunks, dim=2)

    def v(self):
        """Full FP16 shadow Value history, same ordering guarantee as k()."""
        if not self._v_chunks:
            raise ShadowCacheError("v() called on an empty shadow cache (seed_prefill was never called)")
        return self._v_chunks[0] if len(self._v_chunks) == 1 else torch.cat(self._v_chunks, dim=2)


# ---------------------------------------------------------------------------
# Part 7: production past_key_value tuple parsing (torch-agnostic structure check)
# ---------------------------------------------------------------------------

KIVICacheState = namedtuple(
    "KIVICacheState",
    [
        "key_quant_trans", "key_full", "key_scale_trans", "key_mn_trans",
        "value_quant", "value_full", "value_scale", "value_mn", "kv_seq_len",
    ],
)


def parse_kivi_cache_tuple(past_key_value):
    """Restructures the exact 9-element `past_key_value` tuple produced by
    models/llama_kivi.py's LlamaFlashAttention_KIVI (key_quant_trans,
    key_full, key_scale_trans, key_mn_trans, value_quant, value_full,
    value_scale, value_mn, kv_seq_len) into a named, field-accessible
    form. Any *_quant/*_full/*_scale/*_mn field may legitimately be None
    (production's own pass-through/empty-buffer convention, e.g. before
    the first quantization rollover) -- this function only restructures,
    it never guesses or fabricates a missing field. Deliberately
    torch-agnostic: works identically on real tensors or on synthetic
    placeholder objects, so it is fully testable without CUDA/torch
    tensors at all.
    """
    if len(past_key_value) != 9:
        raise AttentionDecodeFeatureError(
            f"expected a 9-element KIVI past_key_value tuple, got {len(past_key_value)} elements"
        )
    return KIVICacheState(*past_key_value)


# ---------------------------------------------------------------------------
# Part 19 (forward-looking, unused by the canary): deterministic record shape
# for a future real collector. Not written to disk by anything in this round.
# ---------------------------------------------------------------------------

DECODE_FEATURE_IDENTITY_FIELDS = ("task", "dataset_index", "layer_idx", "tensor_axis", "policy")
DECODE_FEATURE_STEP_FIELDS = (
    "requested_step_count", "requested_steps", "valid_steps",
    "valid_step_count", "no_decode_exposure", "sample_attention_distortion",
)


def build_decode_feature_record(task, dataset_index, layer_idx, tensor_axis, policy, aggregation_result, generated_token_count):
    """Deterministic field order for one future sample-level decode-feature
    record. Not used to write any scientific data this round -- exists so
    the eventual real collector and this round's canary share one tested
    record shape.
    """
    if tensor_axis not in ("key", "value"):
        raise AttentionDecodeFeatureError(f"tensor_axis must be 'key' or 'value', got {tensor_axis!r}")
    missing = [f for f in DECODE_FEATURE_STEP_FIELDS if f not in aggregation_result]
    if missing:
        raise AttentionDecodeFeatureError(f"aggregation_result missing field(s): {missing}")

    record = OrderedDict(
        [
            ("task", task),
            ("dataset_index", dataset_index),
            ("layer_idx", layer_idx),
            ("tensor_axis", tensor_axis),
            ("policy", policy),
            ("generated_token_count", generated_token_count),
        ]
    )
    for field in DECODE_FEATURE_STEP_FIELDS:
        record[field] = aggregation_result[field]
    return record
