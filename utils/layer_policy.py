"""Dependency-light per-layer K/V bit-width policy parsing/resolution.

Deliberately has no torch/transformers import (mirrors utils/jsonl_integrity.py)
so it can be parsed/validated/tested without pulling in the model stack.

Scope: bit-width allocation for the existing KIVI kernels ("kivi", Stage A),
plus the Stage I1B "rotation_kivi" family (QuaRot-inspired post-RoPE
per-head Hadamard Q/K rotation applied to Key-cache quantization only --
Value stays standard KIVI; see models/llama_kivi.py). "family" is kept
generic in the schema for further extensibility (e.g. Polar) but only
"kivi"/"rotation_kivi" are executable; a bit width of 16 under either family
means the existing FP16/pass-through route (see quantize_key/quantize_value
in models/llama_kivi.py) -- it is not a distinct quantizer.

Global fallback semantics (this is the backward-compatibility contract):
    resolve_layer_policy(num_hidden_layers, k, v, policy_obj=None)
is exactly equivalent to `[(k, v, "kivi")] * num_hidden_layers` -- i.e.
today's global --k_bits/--v_bits behavior, unchanged.
"""
import hashlib
import json
import re
from collections import namedtuple

# Bit widths the KIVI kernels support, plus 16 == "don't quantize this side".
# Mirrors pred_long_bench.py's ALLOWED_BITS.
ALLOWED_BITS = (2, 4, 16)

# "kivi" (Stage A) and "rotation_kivi" (Stage I1B: QuaRot-inspired post-RoPE
# per-head Hadamard Q/K rotation applied to KIVI Key-cache quantization --
# NOT full QuaRot; Value remains standard KIVI) are executable. The field
# was kept in the schema ahead of time so adding "rotation_kivi" required no
# schema migration -- "polar" remains a placeholder only, not executable.
SUPPORTED_FAMILIES = ("kivi", "rotation_kivi")

# Used both to validate policy_name on load and as the directory-naming
# component, so it must already be filesystem-safe -- no separate
# sanitization step is needed at directory-build time.
_POLICY_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_LAYER_KEY_RE = re.compile(r"^-?\d+$")

LayerPolicyEntry = namedtuple("LayerPolicyEntry", ["k_bits", "v_bits", "family"])
ResolvedLayerPolicy = namedtuple("ResolvedLayerPolicy", ["policy_name", "layers", "source"])
# layers: tuple[LayerPolicyEntry], length == num_hidden_layers, index == layer_idx
# source: "global_fallback" | "policy_file"


class LayerPolicyError(RuntimeError):
    """Any malformed/invalid/out-of-range policy input. Always fails closed:
    never guesses, clamps, or silently drops an invalid entry."""


def _validate_cell(cell, where):
    if not isinstance(cell, dict):
        raise LayerPolicyError(f"{where}: expected an object with k_bits/v_bits/family, got {type(cell).__name__}")

    required = {"k_bits", "v_bits", "family"}
    missing = required - set(cell)
    if missing:
        raise LayerPolicyError(f"{where}: missing required field(s) {sorted(missing)}")
    extra = set(cell) - required
    if extra:
        raise LayerPolicyError(f"{where}: unexpected field(s) {sorted(extra)}")

    k_bits, v_bits, family = cell["k_bits"], cell["v_bits"], cell["family"]
    if not isinstance(k_bits, int) or isinstance(k_bits, bool) or k_bits not in ALLOWED_BITS:
        raise LayerPolicyError(f"{where}: k_bits={k_bits!r} must be one of {ALLOWED_BITS}")
    if not isinstance(v_bits, int) or isinstance(v_bits, bool) or v_bits not in ALLOWED_BITS:
        raise LayerPolicyError(f"{where}: v_bits={v_bits!r} must be one of {ALLOWED_BITS}")
    if not isinstance(family, str) or family not in SUPPORTED_FAMILIES:
        raise LayerPolicyError(
            f"{where}: family={family!r} is not executable in Stage A; supported families: {list(SUPPORTED_FAMILIES)}"
        )
    return LayerPolicyEntry(k_bits=k_bits, v_bits=v_bits, family=family)


def parse_policy_json(text):
    """Strict JSON parse; fails closed with a clear message on malformed input."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        raise LayerPolicyError(f"Malformed policy JSON: {e}") from e
    if not isinstance(obj, dict):
        raise LayerPolicyError(f"Policy JSON root must be an object, got {type(obj).__name__}")
    return obj


def load_policy_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise LayerPolicyError(f"Could not read layer policy file {path}: {e}") from e
    return parse_policy_json(text)


def resolve_layer_policy(num_hidden_layers, global_k_bits, global_v_bits, policy_obj=None):
    """Resolve a sparse policy object (or None) into exactly
    num_hidden_layers LayerPolicyEntry values, indexed by layer_idx.

    policy_obj=None reproduces today's global behavior exactly:
    every layer gets (global_k_bits, global_v_bits, "kivi").
    """
    if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        raise LayerPolicyError(f"num_hidden_layers must be a positive int, got {num_hidden_layers!r}")

    if policy_obj is None:
        global_entry = _validate_cell(
            {"k_bits": global_k_bits, "v_bits": global_v_bits, "family": "kivi"}, "global fallback"
        )
        return ResolvedLayerPolicy(
            policy_name="global",
            layers=tuple(global_entry for _ in range(num_hidden_layers)),
            source="global_fallback",
        )

    required_top = {"policy_name", "default", "overrides"}
    missing_top = required_top - set(policy_obj)
    if missing_top:
        raise LayerPolicyError(f"Policy JSON missing required top-level field(s): {sorted(missing_top)}")
    extra_top = set(policy_obj) - required_top
    if extra_top:
        raise LayerPolicyError(f"Policy JSON has unexpected top-level field(s): {sorted(extra_top)}")

    raw_name = policy_obj["policy_name"]
    if not isinstance(raw_name, str) or not _POLICY_NAME_RE.match(raw_name):
        raise LayerPolicyError(
            f"policy_name={raw_name!r} is invalid; must be a 1-128 character string matching "
            f"[A-Za-z0-9_.-]+ (it is used verbatim in output directory names)"
        )

    default_entry = _validate_cell(policy_obj["default"], "default")

    overrides_raw = policy_obj["overrides"]
    if not isinstance(overrides_raw, dict):
        raise LayerPolicyError(f"overrides must be an object keyed by layer index string, got {type(overrides_raw).__name__}")

    layers = [default_entry] * num_hidden_layers
    seen_indices = set()
    for key, cell in overrides_raw.items():
        if not isinstance(key, str) or not _LAYER_KEY_RE.match(key):
            raise LayerPolicyError(f"overrides key {key!r} is not a valid integer layer index string")
        idx = int(key)
        if idx < 0:
            raise LayerPolicyError(
                f"overrides layer index {idx} is negative; layer indices must be in [0, {num_hidden_layers})"
            )
        if idx >= num_hidden_layers:
            raise LayerPolicyError(
                f"overrides layer index {idx} is out of range; must be in [0, {num_hidden_layers}) "
                f"for a {num_hidden_layers}-layer model"
            )
        if idx in seen_indices:
            # Catches e.g. "17" and "017" both mapping to layer 17, which
            # are distinct JSON keys but must not silently overwrite.
            raise LayerPolicyError(f"duplicate override for layer {idx} (key {key!r})")
        seen_indices.add(idx)
        layers[idx] = _validate_cell(cell, f"overrides[{idx}]")

    return ResolvedLayerPolicy(policy_name=raw_name, layers=tuple(layers), source="policy_file")


def load_and_resolve(path, num_hidden_layers, global_k_bits, global_v_bits):
    """Convenience wrapper: load a policy file from disk and resolve it."""
    policy_obj = load_policy_file(path)
    return resolve_layer_policy(num_hidden_layers, global_k_bits, global_v_bits, policy_obj)


def canonical_policy_dict(resolved):
    """Deterministic, fully-expanded (non-sparse) JSON-serializable
    representation -- used both for embedding into run_config.json and for
    computing policy_hash(). Includes the derived quantize_key/quantize_value
    per layer so the resolved policy is self-describing without re-deriving
    anything downstream."""
    return {
        "policy_name": resolved.policy_name,
        "source": resolved.source,
        "layers": [
            {
                "layer_idx": i,
                "k_bits": entry.k_bits,
                "v_bits": entry.v_bits,
                "family": entry.family,
                "quantize_key": entry.k_bits < 16,
                "quantize_value": entry.v_bits < 16,
            }
            for i, entry in enumerate(resolved.layers)
        ],
    }


def policy_hash(resolved):
    """Stable short content hash of the fully-resolved (non-sparse) policy.
    Two policy files that resolve to the same per-layer bits get the same
    hash regardless of policy_name; two policies with the same policy_name
    but different overrides get different hashes. Used both in directory
    naming (collision resistance beyond the human-chosen policy_name) and
    as the run_config.json resume-identity key."""
    canonical = canonical_policy_dict(resolved)
    # policy_name/source excluded from the hash input on purpose: hash
    # identifies *behavior* (the resolved per-layer bits), not the label.
    blob = json.dumps(canonical["layers"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def layers_as_config_list(resolved):
    """The minimal per-layer list consumed by models/llama_kivi.py at model
    construction time (config.layer_kv_policy)."""
    return [{"k_bits": e.k_bits, "v_bits": e.v_bits, "family": e.family} for e in resolved.layers]
