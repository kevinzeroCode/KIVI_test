"""Deterministic policy generation/discovery for the layer-sensitivity pilot
(Stage D). Dependency-light: no torch/transformers import, so dry-run and
CPU-only tests never need the model stack.

Pilot design (fixed for Stage D0 -- do not silently change without updating
the corresponding docs/report):
  - 8 probed layers x 2 axes (key, value) = 16 conditions.
  - Every condition: all layers K16/V16 except the probed layer, which is
    K2/V16 (key axis) or K16/V2 (value axis).
  - 4 screening tasks, full LongBench test-split counts (no sub-sampling).

Task-count design (fixed after the Stage-F0 dry-run bug): this module
deliberately separates two distinct concepts that select_task_counts()
previously conflated into a single 4-task dict:
  - DEFAULT_PILOT_TASK_COUNTS: the Stage-D screening default (trec, lcc,
    passage_retrieval_en, 2wikimqa) -- what a bare `--tasks`-less invocation
    of scripts/run_layer_sensitivity_pilot.py runs. Must never change
    silently; the completed Stage-D experiment depends on this default.
  - SUPPORTED_TASK_COUNTS: every LongBench task this pilot infrastructure is
    *allowed* to run when explicitly requested via --tasks (e.g. Stage F0's
    multifieldqa_en/samsum), reusing Phase-1's already-validated
    EXPECTED_TASK_COUNTS (analysis/analyze_kv_ablation.py) rather than
    re-deriving or inventing sample counts here.
"""
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.analyze_kv_ablation import EXPECTED_TASK_COUNTS as SUPPORTED_TASK_COUNTS  # noqa: E402
from utils.layer_policy import (  # noqa: E402
    LayerPolicyError,
    load_and_resolve,
    policy_hash,
    resolve_layer_policy,
)

EXPECTED_NUM_LAYERS = 32  # LongChat-7b-v1.5-32k, confirmed from its HF config.json (Stage A/C).

PILOT_LAYERS = (0, 4, 9, 13, 18, 22, 27, 31)
PILOT_AXES = ("key", "value")  # per-layer order: key probe before value probe

DEFAULT_PILOT_TASK_COUNTS = OrderedDict(
    [
        ("trec", 200),
        ("lcc", 500),
        ("passage_retrieval_en", 200),
        ("2wikimqa", 200),
    ]
)
# Backward-compatible alias: every existing caller of PILOT_TASK_COUNTS
# (the completed Stage-D analysis/validation scripts) only ever means the
# Stage-D default 4 tasks -- keep it pointing at exactly that, unchanged.
PILOT_TASK_COUNTS = DEFAULT_PILOT_TASK_COUNTS
PILOT_TASKS = list(PILOT_TASK_COUNTS)

AXIS_BITS = {"key": {"k_bits": 2, "v_bits": 16}, "value": {"k_bits": 16, "v_bits": 2}}
AXIS_FILE_SUFFIX = {"key": "k2", "value": "v2"}


class PilotPolicyError(RuntimeError):
    """Any pilot policy generation/discovery/validation failure. Fails closed."""


def condition_id(layer_idx, axis):
    return f"layer{layer_idx:02d}_{axis}"


def policy_filename(layer_idx, axis):
    return f"{condition_id(layer_idx, axis)}_{AXIS_FILE_SUFFIX[axis]}.json"


def build_policy_obj(layer_idx, axis):
    """Deterministic sparse policy dict matching utils/layer_policy.py's schema."""
    if axis not in AXIS_BITS:
        raise PilotPolicyError(f"Unknown axis {axis!r}; must be one of {list(AXIS_BITS)}")
    bits = AXIS_BITS[axis]
    name = policy_filename(layer_idx, axis)[: -len(".json")]
    return {
        "policy_name": name,
        "default": {"k_bits": 16, "v_bits": 16, "family": "kivi"},
        "overrides": {str(layer_idx): {"k_bits": bits["k_bits"], "v_bits": bits["v_bits"], "family": "kivi"}},
    }


def all_pilot_specs(layers=PILOT_LAYERS, axes=PILOT_AXES):
    """Deterministic, ordered list of the 16 (default) pilot condition specs:
    outer loop over layers (in the given order), inner loop over axes (key,
    then value) -- this exact nesting is the pilot's canonical condition
    order everywhere (policy generation, dry-run printout, manifest,
    sequential execution)."""
    specs = []
    for layer_idx in layers:
        for axis in axes:
            specs.append(
                {
                    "condition_id": condition_id(layer_idx, axis),
                    "layer_idx": layer_idx,
                    "axis": axis,
                    "k_bits": AXIS_BITS[axis]["k_bits"],
                    "v_bits": AXIS_BITS[axis]["v_bits"],
                    "policy_filename": policy_filename(layer_idx, axis),
                    "policy_obj": build_policy_obj(layer_idx, axis),
                }
            )
    return specs


def write_pilot_policies(policies_dir, specs=None):
    """Idempotently materialize the policy JSON files on disk. If a file
    already exists, its content must match exactly what would be generated
    now (byte-for-byte JSON structure) -- fails closed on any drift rather
    than silently overwriting a possibly-hand-edited file."""
    policies_dir = Path(policies_dir)
    policies_dir.mkdir(parents=True, exist_ok=True)
    specs = specs if specs is not None else all_pilot_specs()
    written, unchanged = [], []
    for spec in specs:
        path = policies_dir / spec["policy_filename"]
        text = json.dumps(spec["policy_obj"], indent=2) + "\n"
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if json.loads(existing) != spec["policy_obj"]:
                raise PilotPolicyError(
                    f"{path} already exists with different content than the deterministic "
                    "pilot policy generator would produce. Refusing to overwrite -- inspect "
                    "and resolve manually."
                )
            unchanged.append(path)
            continue
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written, unchanged


def discover_and_validate_pilot_policies(policies_dir, layers=PILOT_LAYERS, axes=PILOT_AXES, num_hidden_layers=EXPECTED_NUM_LAYERS):
    """Loads, resolves, and validates every pilot policy file from disk
    (never trusts in-memory specs alone), computes each one's hash, and
    fails closed if any of the 16 hashes collide. Returns the ordered list
    of condition dicts (same order as all_pilot_specs), each augmented with
    'resolved_policy_hash' and 'resolved_layers' (the fully expanded
    per-layer bits, straight from utils.layer_policy)."""
    policies_dir = Path(policies_dir)
    specs = all_pilot_specs(layers, axes)
    conditions = []
    seen_hashes = {}

    for spec in specs:
        path = policies_dir / spec["policy_filename"]
        if not path.exists():
            raise PilotPolicyError(f"Expected pilot policy file missing: {path}")
        try:
            resolved = load_and_resolve(str(path), num_hidden_layers, global_k_bits=16, global_v_bits=16)
        except LayerPolicyError as e:
            raise PilotPolicyError(f"{path}: {e}") from e

        h = policy_hash(resolved)
        if h in seen_hashes:
            raise PilotPolicyError(
                f"Policy hash collision: {spec['condition_id']} and {seen_hashes[h]} both hash to {h}"
            )
        seen_hashes[h] = spec["condition_id"]

        cond = dict(spec)
        cond["policy_path"] = str(path)
        cond["resolved_policy_hash"] = h
        cond["resolved_layer_policy"] = resolved
        conditions.append(cond)

    return conditions


def output_dir_name(condition):
    """<condition_id>_<hash12> -- e.g. layer00_key_ad4b68111f21. Collision
    with a different condition would require both the same condition_id
    (impossible -- each (layer, axis) pair is unique by construction) AND
    the same hash (impossible unless the policies are identical, which
    discover_and_validate_pilot_policies already rejects)."""
    return f"{condition['condition_id']}_{condition['resolved_policy_hash']}"


def total_example_count(task_counts=PILOT_TASK_COUNTS, num_conditions=16):
    return num_conditions * sum(task_counts.values())


def select_task_counts(task_names, supported_task_counts=SUPPORTED_TASK_COUNTS, default_task_counts=DEFAULT_PILOT_TASK_COUNTS):
    """task_names=None (or empty/falsy) preserves the Stage-D default
    exactly -- returns default_task_counts unchanged, never validated
    against anything (there is nothing to validate).

    An explicit task_names list is validated against supported_task_counts
    (every LongBench task this pilot is allowed to run, not just the 4
    Stage-D defaults) and returned as an OrderedDict restricted to those
    tasks, in supported_task_counts' canonical order (not the caller's
    argument order, so downstream iteration/printing stays deterministic).
    Fails closed on any task name not present in supported_task_counts.
    """
    if not task_names:
        return OrderedDict(default_task_counts)
    unknown = [t for t in task_names if t not in supported_task_counts]
    if unknown:
        raise PilotPolicyError(
            f"Unknown task(s) {unknown}; must be a subset of {list(supported_task_counts)}"
        )
    return OrderedDict((t, supported_task_counts[t]) for t in supported_task_counts if t in set(task_names))
