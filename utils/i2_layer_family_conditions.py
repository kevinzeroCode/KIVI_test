"""Deterministic policy generation/discovery for Stage I2A (layer x
quantizer-family sensitivity). Dependency-light: no torch/transformers
import, so dry-run and CPU-only tests never need the model stack. Mirrors
utils/pilot_policy.py's design exactly, adapted along one axis:

  - Stage D0 (pilot_policy.py) varied (layer, axis) at fixed family="kivi"
    (K2/V16 for the key axis, K16/V2 for the value axis).
  - Stage I2A varies (layer, family) at a FIXED precision, K2/V16, for
    whichever family is under test -- the scientific question is "does
    family (kivi vs rotation_kivi) matter, holding K2/V16 and layer fixed",
    not "which axis is more sensitive".

Locked design (do not silently change without updating
docs/stage_i2_pre_registration.md):
  - 8 layers (identical set to Stage D0's PILOT_LAYERS) x 2 families = 16
    conditions.
  - Every condition: all layers K16/V16 family="kivi" except the target
    layer, which is K2/V16 under the condition's family.
  - 6 tasks (trec, lcc, passage_retrieval_en, 2wikimqa, multifieldqa_en,
    samsum), full LongBench test-split counts (no sub-sampling) -- reusing
    Phase-1's already-validated EXPECTED_TASK_COUNTS
    (analysis/analyze_kv_ablation.py) rather than re-deriving or inventing
    sample counts here.

C1 (Stage I1) observed, on lcc/sample 122/layer 0/K2-V16 only, that
Rotation-KIVI had higher Key decode distortion than standard KIVI. That
single-sample, single-layer distortion measurement is NOT an I2 endpoint,
does NOT determine I2's task selection, and does NOT determine any expected
I2 downstream direction -- see docs/stage_i2_pre_registration.md Section 12.
No layer here is removed or reselected because of the C1 result.
"""
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.analyze_kv_ablation import EXPECTED_TASK_COUNTS as _ALL_LONGBENCH_TASK_COUNTS  # noqa: E402
from utils.layer_policy import (  # noqa: E402
    LayerPolicyError,
    SUPPORTED_FAMILIES,
    load_and_resolve,
    policy_hash,
)

EXPECTED_NUM_LAYERS = 32  # LongChat-7b-v1.5-32k, confirmed from its HF config.json (Stage A/C).

# Identical 8-layer set as Stage D0's utils.pilot_policy.PILOT_LAYERS.
# LOCKED for Stage I2A: do not add/remove/reselect layers after looking at
# I2 results (docs/stage_i2_pre_registration.md Section 3).
I2_LAYERS = (0, 4, 9, 13, 18, 22, 27, 31)

# Both families this pilot's schema (utils.layer_policy.SUPPORTED_FAMILIES)
# currently makes executable. "polar" remains out of scope for I2A.
I2_FAMILIES = ("kivi", "rotation_kivi")
assert set(I2_FAMILIES) == set(SUPPORTED_FAMILIES), (
    "I2_FAMILIES must track utils.layer_policy.SUPPORTED_FAMILIES exactly -- "
    "I2A tests every family the schema makes executable, no more, no less."
)

# Fixed target-layer precision for every I2A condition. No K4, no Value
# quantization, no mixed-family multi-layer policy, one target layer per
# condition only.
I2_K_BITS = 2
I2_V_BITS = 16
I2_GROUP_SIZE = 32
I2_RESIDUAL_LENGTH = 128

# The six already-established LongBench workloads, in the project's
# canonical EXPECTED_TASK_COUNTS order (never the caller's order), each
# count taken verbatim from analysis/analyze_kv_ablation.py -- never
# re-derived or hand-typed here.
I2_TASKS = ("trec", "lcc", "passage_retrieval_en", "2wikimqa", "multifieldqa_en", "samsum")
I2_TASK_COUNTS = OrderedDict((t, _ALL_LONGBENCH_TASK_COUNTS[t]) for t in I2_TASKS)

TOTAL_CONDITIONS = len(I2_LAYERS) * len(I2_FAMILIES)  # 8 x 2 = 16


class I2ConditionError(RuntimeError):
    """Any I2A condition generation/discovery/validation failure. Fails closed."""


def condition_id(layer_idx, family):
    return f"layer{layer_idx:02d}_{family}"


def policy_filename(layer_idx, family):
    return f"{condition_id(layer_idx, family)}.json"


def build_policy_obj(layer_idx, family):
    """Deterministic sparse policy dict matching utils/layer_policy.py's
    schema: every layer K16/V16 family="kivi" except `layer_idx`, which is
    K2/V16 under `family`."""
    if family not in I2_FAMILIES:
        raise I2ConditionError(f"Unknown family {family!r}; must be one of {I2_FAMILIES}")
    name = policy_filename(layer_idx, family)[: -len(".json")]
    return {
        "policy_name": name,
        "default": {"k_bits": 16, "v_bits": 16, "family": "kivi"},
        "overrides": {str(layer_idx): {"k_bits": I2_K_BITS, "v_bits": I2_V_BITS, "family": family}},
    }


def all_i2_specs(layers=I2_LAYERS, families=I2_FAMILIES):
    """Deterministic, ordered list of the 16 (default) I2A condition specs:
    outer loop over layers (in the given order), inner loop over families
    (kivi, then rotation_kivi) -- this exact nesting is the canonical
    condition order everywhere (policy generation, dry-run printout,
    manifest, sequential execution)."""
    specs = []
    for layer_idx in layers:
        for family in families:
            specs.append(
                {
                    "condition_id": condition_id(layer_idx, family),
                    "layer_idx": layer_idx,
                    "family": family,
                    "k_bits": I2_K_BITS,
                    "v_bits": I2_V_BITS,
                    "policy_filename": policy_filename(layer_idx, family),
                    "policy_obj": build_policy_obj(layer_idx, family),
                }
            )
    return specs


def write_i2_policies(policies_dir, specs=None):
    """Idempotently materialize the policy JSON files on disk. If a file
    already exists, its content must match exactly what would be generated
    now (byte-for-byte JSON structure) -- fails closed on any drift rather
    than silently overwriting a possibly-hand-edited file."""
    policies_dir = Path(policies_dir)
    policies_dir.mkdir(parents=True, exist_ok=True)
    specs = specs if specs is not None else all_i2_specs()
    written, unchanged = [], []
    for spec in specs:
        path = policies_dir / spec["policy_filename"]
        text = json.dumps(spec["policy_obj"], indent=2) + "\n"
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if json.loads(existing) != spec["policy_obj"]:
                raise I2ConditionError(
                    f"{path} already exists with different content than the deterministic "
                    "I2A policy generator would produce. Refusing to overwrite -- inspect "
                    "and resolve manually."
                )
            unchanged.append(path)
            continue
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written, unchanged


def _assert_policy_shape(resolved, layer_idx, k_bits, v_bits, family, num_hidden_layers, where):
    """Verifies a resolved policy's per-layer content matches the I2A
    one-target-layer contract exactly: the target layer is (k_bits, v_bits,
    family); every OTHER layer is (16, 16, "kivi"). This is the same
    assertion shape as scripts/h2_attention_decode_parity_canary.py's
    load_canary_model post-construction check, applied here at policy-
    discovery time (CPU-only, before any model is ever loaded)."""
    if len(resolved.layers) != num_hidden_layers:
        raise I2ConditionError(f"{where}: resolved {len(resolved.layers)} layers, expected {num_hidden_layers}")
    for i, entry in enumerate(resolved.layers):
        if i == layer_idx:
            got = (entry.k_bits, entry.v_bits, entry.family)
            want = (k_bits, v_bits, family)
            if got != want:
                raise I2ConditionError(f"{where}: target layer {layer_idx} resolved to {got}, expected {want}")
        else:
            got = (entry.k_bits, entry.v_bits, entry.family)
            if got != (16, 16, "kivi"):
                raise I2ConditionError(f"{where}: non-target layer {i} resolved to {got}, expected (16, 16, 'kivi')")


def discover_and_validate_i2_policies(policies_dir, layers=I2_LAYERS, families=I2_FAMILIES, num_hidden_layers=EXPECTED_NUM_LAYERS):
    """Loads, resolves, and validates every I2A policy file from disk
    (never trusts in-memory specs alone), computes each one's hash, fails
    closed if any of the 16 hashes collide, and fails closed if any
    resolved policy does not match the one-target-layer / all-others-K16-
    V16-kivi contract. Returns the ordered list of condition dicts (same
    order as all_i2_specs), each augmented with 'resolved_policy_hash' and
    'resolved_layer_policy' (the fully expanded per-layer bits, straight
    from utils.layer_policy)."""
    policies_dir = Path(policies_dir)
    specs = all_i2_specs(layers, families)
    conditions = []
    seen_hashes = {}

    for spec in specs:
        path = policies_dir / spec["policy_filename"]
        if not path.exists():
            raise I2ConditionError(f"Expected I2A policy file missing: {path}")
        try:
            resolved = load_and_resolve(str(path), num_hidden_layers, global_k_bits=16, global_v_bits=16)
        except LayerPolicyError as e:
            raise I2ConditionError(f"{path}: {e}") from e

        _assert_policy_shape(
            resolved, spec["layer_idx"], spec["k_bits"], spec["v_bits"], spec["family"], num_hidden_layers, str(path)
        )

        h = policy_hash(resolved)
        if h in seen_hashes:
            raise I2ConditionError(f"Policy hash collision: {spec['condition_id']} and {seen_hashes[h]} both hash to {h}")
        seen_hashes[h] = spec["condition_id"]

        cond = dict(spec)
        cond["policy_path"] = str(path)
        cond["resolved_policy_hash"] = h
        cond["resolved_layer_policy"] = resolved
        conditions.append(cond)

    return conditions


def output_dir_name(condition):
    """<condition_id>_<hash12> -- e.g. layer00_kivi_ad4b68111f21. Collision
    with a different condition would require both the same condition_id
    (impossible -- each (layer, family) pair is unique by construction) AND
    the same hash (impossible unless the policies are identical, which
    discover_and_validate_i2_policies already rejects)."""
    return f"{condition['condition_id']}_{condition['resolved_policy_hash']}"


def total_example_count(task_counts=I2_TASK_COUNTS, num_conditions=TOTAL_CONDITIONS):
    return num_conditions * sum(task_counts.values())


# 16 conditions x 1450 samples/condition = 23,200 planned generation rows.
TOTAL_PLANNED_ROWS = total_example_count()
