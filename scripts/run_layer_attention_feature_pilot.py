"""Stage H3: decode-aware attention feature scientific collection runner.

Implements the locked Stage-H design (docs/stage_h0_pre_registration.md,
Stage H0-DECODE-AUDIT, Stage H0-PRE) for the full 272-trajectory
collection. THIS ROUND: CPU trajectory planning / record schema / resume
logic is complete and tested; --dry-run is fully implemented and
torch-free (verified: does not import torch/transformers). The real GPU
collection entry point is deliberately NOT implemented in this round --
main() refuses to run without --dry-run -- so there is no code path here
that could accidentally launch GPU work. Its design (reusing the
H2-canary-validated capture pattern: a scoped spy on
_apply_rotary_pos_emb_inplace + nn.functional.softmax installed only
around the target layer's own forward(), q_proj/v_proj/o_proj hooks, a
past_key_value pre-hook) is described in the Stage-H3 final report, to be
implemented as a separate, explicitly-authorized next step.

Reuses, never reimplements:
  - utils.attention_decode_features (key_decode_distortion,
    value_decode_distortion, aggregate_sample_decode_distortion,
    ShadowKVCache, parse_kivi_cache_tuple) -- the exact Stage-H0-PRE/H2
    pure measurement math, unmodified.
  - utils.generation_semantics.resolve_generate_kwargs (real per-task
    generation kwargs, including samsum's special semantics).
  - utils.layer_policy.resolve_layer_policy (policy construction).
  - pred_long_bench.build_model_and_tokenizer / build_chat.
  - models.llama_kivi.repeat_kv / cuda_bmm_fA_qB_outer (real production
    kernels, called on captured production tensors).

Locked scientific semantics (not altered here):
  trajectory=production-KIVI; Key policy=single-layer K2/V16; Value
  policy=single-layer K16/V2; requested decode steps={1,2,4,8,16}; equal-
  step mean aggregation; zero-decode-exposure convention.

Sample selection is NOT re-derived: the exact 24 (task, dataset_index)
pairs are the same LOCKED constants used throughout Stage G/H (see
LOCKED_SAMPLE_SELECTION below) -- this script never calls
select_calibration_indices or touches the dataset's length metadata again.

Usage (dry run, CPU-only, no torch/model import):
    ./.venv/bin/python scripts/run_layer_attention_feature_pilot.py --dry-run

Usage (real collection -- NOT executed by anything in this round):
    CUDA_VISIBLE_DEVICES=0 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \\
    ./.venv/bin/python scripts/run_layer_attention_feature_pilot.py \\
        --output-root outputs/layer_attention_feature_pilot/primary --scope primary
"""
import argparse
import errno
import fcntl
import json
import math
import os
import re
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# utils.attention_decode_features imports torch at module level -- deferred
# into the specific functions that need it (build_trajectory_record,
# validate_record_shape, load_completed_records) so --dry-run stays
# torch-free, matching Stage G's established discipline
# (scripts/collect_layer_features.py's --dry-run path).
DEFAULT_REQUESTED_STEPS = (1, 2, 4, 8, 16)  # mirrors utils.attention_decode_features.DEFAULT_REQUESTED_STEPS exactly (cross-checked by a CPU test); kept as a plain tuple here so printing/planning never needs torch.

DEFAULT_OUTPUT_ROOT_PRIMARY = os.path.join(REPO_ROOT, "outputs", "layer_attention_feature_pilot", "primary")
DEFAULT_OUTPUT_ROOT_DIAGNOSTIC = os.path.join(REPO_ROOT, "outputs", "layer_attention_feature_pilot", "diagnostic")
DEFAULT_MODEL_NAME = "lmsys/longchat-7b-v1.5-32k"
DEFAULT_GROUP_SIZE = 32
DEFAULT_RESIDUAL_LENGTH = 128
DEFAULT_SEED = 42
DEFAULT_MAX_LENGTH = 31500
# Enough to reach cached decode step 16 (prefill->token#1, steps 1..16->tokens #2..#17).
DEFAULT_MAX_NEW_TOKENS_HORIZON = 20

PRIMARY_TASKS = ("trec", "lcc", "passage_retrieval_en", "2wikimqa")
PRIMARY_LAYERS = (0, 4, 9, 13, 18, 22, 27, 31)
DIAGNOSTIC_TASKS = ("multifieldqa_en", "samsum")
DIAGNOSTIC_LAYERS = (0,)
AXES = ("key", "value")
AXIS_POLICY_BITS = {"key": (2, 16), "value": (16, 2)}

# LOCKED -- the exact Stage-G/H preregistered 24 calibration samples.
# Never re-derived, never reranked, never reselected. See
# docs/stage_g3b_pre_registration.md Section 3 / docs/stage_h0_pre_registration.md.
LOCKED_SAMPLE_SELECTION = OrderedDict(
    [
        ("trec", (153, 154, 82, 151)),
        ("lcc", (122, 174, 214, 357)),
        ("passage_retrieval_en", (141, 185, 105, 41)),
        ("2wikimqa", (164, 44, 138, 38)),
        ("multifieldqa_en", (50, 103, 81, 92)),
        ("samsum", (58, 118, 151, 52)),
    ]
)

CONFLICTING_PROCESS_PATTERNS = [
    "pred_long_bench.py",
    "layer_policy_smoke.py",
    "run_layer_sensitivity_pilot.py",
    "collect_layer_features.py",
    "h2_attention_decode_parity_canary.py",
    "run_layer_attention_feature_pilot.py",
]

REQUIRED_RECORD_FIELDS = (
    "scope", "task", "dataset_index", "layer_idx", "tensor_axis", "policy", "k_bits", "v_bits",
    "model", "git_commit", "seed",
    "prompt_input_tokens", "actual_generated_token_count",
    "requested_decode_steps", "sampled_decode_steps", "valid_step_count", "no_decode_exposure",
    "sample_attention_distortion", "per_step_distortions",
)


class PilotConfigError(ValueError):
    pass


class PilotLockError(RuntimeError):
    pass


class ResumeError(RuntimeError):
    """A malformed/partial existing record was found where a clean resume
    decision was required -- fails closed rather than guessing."""


class RecordValidationError(ValueError):
    """A trajectory record fails the fail-closed structural check in
    validate_record_shape(). Deliberately a LOCAL exception type (not
    utils.attention_decode_features.AttentionDecodeFeatureError) so this
    pure JSON/dict validation logic never needs to import that
    torch-dependent module -- keeping --dry-run and the standalone
    validator (Part 15) torch-free."""


# ---------------------------------------------------------------------------
# Part 1/2/8: pure trajectory planning (torch-free, fully CPU-testable)
# ---------------------------------------------------------------------------

def build_trajectory_plan(
    primary_tasks=PRIMARY_TASKS,
    primary_layers=PRIMARY_LAYERS,
    diagnostic_tasks=DIAGNOSTIC_TASKS,
    diagnostic_layers=DIAGNOSTIC_LAYERS,
    axes=AXES,
    sample_selection=LOCKED_SAMPLE_SELECTION,
):
    """Returns the deterministic, ordered list of every scientific
    trajectory identity: one dict per (task, dataset_index, layer_idx,
    tensor_axis). Primary: 4 tasks x 4 samples x 8 layers x 2 axes = 256.
    Diagnostic: 2 tasks x 4 samples x 1 layer x 2 axes = 16. Total 272.
    Never merges scopes, never reorders sample_selection.
    """
    plan = []
    for task in primary_tasks:
        for dataset_index in sample_selection[task]:
            for layer_idx in primary_layers:
                for axis in axes:
                    k_bits, v_bits = AXIS_POLICY_BITS[axis]
                    plan.append(
                        OrderedDict(
                            [
                                ("scope", "primary"), ("task", task), ("dataset_index", dataset_index),
                                ("layer_idx", layer_idx), ("tensor_axis", axis),
                                ("policy", f"K{k_bits}/V{v_bits}"), ("k_bits", k_bits), ("v_bits", v_bits),
                            ]
                        )
                    )
    for task in diagnostic_tasks:
        for dataset_index in sample_selection[task]:
            for layer_idx in diagnostic_layers:
                for axis in axes:
                    k_bits, v_bits = AXIS_POLICY_BITS[axis]
                    plan.append(
                        OrderedDict(
                            [
                                ("scope", "diagnostic"), ("task", task), ("dataset_index", dataset_index),
                                ("layer_idx", layer_idx), ("tensor_axis", axis),
                                ("policy", f"K{k_bits}/V{v_bits}"), ("k_bits", k_bits), ("v_bits", v_bits),
                            ]
                        )
                    )
    return plan


def trajectory_identity(t):
    return (t["task"], t["dataset_index"], t["layer_idx"], t["tensor_axis"])


def validate_scope_guard(scope, layers):
    """Part 13: diagnostic tasks may only ever use layer 0; primary tasks
    may only ever use the exact 8 primary layers. Fails BEFORE any model
    load."""
    layers = tuple(sorted(set(layers)))
    if scope == "diagnostic" and layers != DIAGNOSTIC_LAYERS:
        raise PilotConfigError(f"diagnostic scope requires exactly layers {DIAGNOSTIC_LAYERS}, got {layers}")
    if scope == "primary" and layers != tuple(sorted(PRIMARY_LAYERS)):
        raise PilotConfigError(f"primary scope requires exactly layers {sorted(PRIMARY_LAYERS)}, got {layers}")


def validate_policy_bits(axis, k_bits, v_bits):
    """Fails BEFORE any model load if an axis/bit-width combination other
    than the two locked scientific policies is requested. Never allows a
    joint K2/V2 scientific policy (Stage H explicitly forbids this,
    unlike Stage G's collector)."""
    expected = AXIS_POLICY_BITS.get(axis)
    if expected is None:
        raise PilotConfigError(f"unknown axis {axis!r}; must be 'key' or 'value'")
    if (k_bits, v_bits) != expected:
        raise PilotConfigError(f"axis {axis!r} requires (k_bits,v_bits)={expected}, got ({k_bits},{v_bits})")


def group_trajectories_by_policy(plan):
    """Groups the flat plan by (scope, layer_idx, tensor_axis) -- one
    model load per group (Part 3). Diagnostic trajectories are NEVER
    merged into a primary group even when the bit-width policy is
    identical (layer 0) -- separate scope means separate model load, for
    unambiguous provenance. Returns an OrderedDict preserving plan order.
    """
    groups = OrderedDict()
    for t in plan:
        key = (t["scope"], t["layer_idx"], t["tensor_axis"])
        groups.setdefault(key, []).append(t)
    return groups


# ---------------------------------------------------------------------------
# Part 7: deterministic trajectory record schema
# ---------------------------------------------------------------------------

def build_trajectory_record(
    trajectory,
    model_name,
    git_commit,
    seed,
    prompt_input_tokens,
    actual_generated_token_count,
    step_distortions,
    requested_steps=DEFAULT_REQUESTED_STEPS,
):
    """Assembles exactly ONE final sample-level record for a trajectory,
    reusing aggregate_sample_decode_distortion (never reimplemented) for
    the step-aggregation/zero-exposure logic. per_step_distortions
    contains only tiny scalar metadata -- never a tensor, matrix, or cache.
    """
    from utils.attention_decode_features import aggregate_sample_decode_distortion  # deferred: keeps --dry-run torch-free

    agg = aggregate_sample_decode_distortion(step_distortions, requested_steps=requested_steps)
    per_step_distortions = [{"decode_step": s, "distortion": step_distortions[s]} for s in agg["valid_steps"]]

    record = OrderedDict(
        [
            ("scope", trajectory["scope"]),
            ("task", trajectory["task"]),
            ("dataset_index", trajectory["dataset_index"]),
            ("layer_idx", trajectory["layer_idx"]),
            ("tensor_axis", trajectory["tensor_axis"]),
            ("policy", trajectory["policy"]),
            ("k_bits", trajectory["k_bits"]),
            ("v_bits", trajectory["v_bits"]),
            ("model", model_name),
            ("git_commit", git_commit),
            ("seed", seed),
            ("prompt_input_tokens", prompt_input_tokens),
            ("actual_generated_token_count", actual_generated_token_count),
            ("requested_decode_steps", list(agg["requested_steps"])),
            ("sampled_decode_steps", list(agg["valid_steps"])),
            ("valid_step_count", agg["valid_step_count"]),
            ("no_decode_exposure", agg["no_decode_exposure"]),
            ("sample_attention_distortion", agg["sample_attention_distortion"]),
            ("per_step_distortions", per_step_distortions),
        ]
    )
    return record


def validate_record_shape(record):
    """Fail-closed structural check reused by both the collector (before
    writing a record) and the standalone validator (Part 15)."""
    missing = [f for f in REQUIRED_RECORD_FIELDS if f not in record]
    if missing:
        raise RecordValidationError(f"record missing required field(s): {missing}")
    if record["sampled_decode_steps"] != sorted(record["sampled_decode_steps"]):
        raise RecordValidationError("sampled_decode_steps must be strictly ascending")
    if not set(record["sampled_decode_steps"]).issubset(set(record["requested_decode_steps"])):
        raise RecordValidationError("sampled_decode_steps must be a subset of requested_decode_steps")
    if record["valid_step_count"] != len(record["sampled_decode_steps"]):
        raise RecordValidationError("valid_step_count inconsistent with sampled_decode_steps")
    if record["no_decode_exposure"] != (record["valid_step_count"] == 0):
        raise RecordValidationError("no_decode_exposure inconsistent with valid_step_count")
    if record["no_decode_exposure"]:
        if record["sample_attention_distortion"] != 0.0:
            raise RecordValidationError("zero-exposure record must have sample_attention_distortion == 0.0")
    else:
        values = [s["distortion"] for s in record["per_step_distortions"]]
        recomputed = sum(values) / len(values)
        if abs(recomputed - record["sample_attention_distortion"]) > 1e-9:
            raise RecordValidationError(
                f"sample_attention_distortion ({record['sample_attention_distortion']}) does not "
                f"reproduce from per_step_distortions (recomputed {recomputed})"
            )
    for s in record["per_step_distortions"]:
        d = s["distortion"]
        if not isinstance(d, (int, float)) or isinstance(d, bool) or not math.isfinite(d):
            raise RecordValidationError(f"non-finite per-step distortion: {d!r}")
        if d < 0:
            raise RecordValidationError(f"negative distortion: {d!r}")
    axis = record["tensor_axis"]
    expected_bits = AXIS_POLICY_BITS.get(axis)
    if expected_bits is None or (record["k_bits"], record["v_bits"]) != expected_bits:
        raise RecordValidationError(f"policy/axis mismatch for axis {axis!r}: got k_bits={record['k_bits']} v_bits={record['v_bits']}")
    return True


# ---------------------------------------------------------------------------
# Part 10/11: exact-identity resume safety
# ---------------------------------------------------------------------------

def load_completed_records(features_path, expected_git_commit=None):
    """Reads an existing features.jsonl (if any) and returns
    dict[identity] -> record for every STRUCTURALLY VALID record found.
    Any malformed line/record encountered raises ResumeError immediately
    (fail closed) rather than silently skipping it -- a partial/corrupt
    prior run must never be silently resumed over.
    """
    completed = {}
    if not os.path.exists(features_path):
        return completed
    with open(features_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ResumeError(f"{features_path}:{line_no}: malformed JSON, refusing to resume: {e}") from e
            try:
                validate_record_shape(record)
            except RecordValidationError as e:
                raise ResumeError(f"{features_path}:{line_no}: malformed record, refusing to resume: {e}") from e
            if expected_git_commit is not None and record["git_commit"] != expected_git_commit:
                raise ResumeError(
                    f"{features_path}:{line_no}: record git_commit={record['git_commit']!r} does not match "
                    f"the current collection HEAD {expected_git_commit!r} -- refusing to silently mix commits"
                )
            identity = (record["task"], record["dataset_index"], record["layer_idx"], record["tensor_axis"])
            if identity in completed:
                raise ResumeError(f"{features_path}:{line_no}: duplicate identity {identity} already present -- refusing to resume")
            completed[identity] = record
    return completed


def plan_remaining_trajectories(plan, completed_identities):
    """Returns the subset of `plan` whose identity is NOT already present
    in `completed_identities` (a set of 4-tuples) -- never re-runs a
    validly-completed trajectory, never silently overwrites it."""
    return [t for t in plan if trajectory_identity(t) not in completed_identities]


# ---------------------------------------------------------------------------
# Single-instance locking (same pattern as scripts/run_layer_sensitivity_pilot.py)
# ---------------------------------------------------------------------------

def acquire_pilot_lock(lock_path):
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    fh = open(lock_path, "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        if e.errno not in (errno.EAGAIN, errno.EACCES):
            raise
        fh.seek(0)
        holder = fh.read().strip() or "unknown"
        fh.close()
        raise PilotLockError(f"Pilot lock {lock_path} is already held (recorded holder: {holder}).")
    fh.seek(0)
    fh.truncate()
    fh.write(f"pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()}\n")
    fh.flush()
    return fh


def release_pilot_lock(fh):
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def check_no_conflicting_process(patterns=CONFLICTING_PROCESS_PATTERNS, self_pid=None):
    self_pid = self_pid if self_pid is not None else os.getpid()
    pattern = "|".join(re.escape(p) for p in patterns)
    try:
        out = subprocess.check_output(["pgrep", "-af", pattern], stderr=subprocess.DEVNULL).decode()
    except subprocess.CalledProcessError:
        return []
    except FileNotFoundError:
        return ["WARNING: pgrep not available; conflicting-process check could not run"]

    genuine_invocation_re = re.compile(
        r"(?:^|/|\s)\S*python\S*\s+\S*(?:" + "|".join(re.escape(p) for p in patterns) + r")\b"
    )
    conflicts = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_str = line.split(None, 1)[0]
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if pid == self_pid:
            continue
        if "bash -c" in line:
            continue
        if not genuine_invocation_re.search(line):
            continue
        conflicts.append(line)
    return conflicts


def gpu_preflight(threshold_bytes=10 * 1024**3):
    """Part 19: inspects ACTUAL GPU state at launch time and returns a
    warning (never a kill/modify action) if a large amount of memory is
    already used by any process -- including unrelated ones. Never
    inspects/assumes any specific prior observation (e.g. the H2-round
    llama-server) still holds; queries nvidia-smi fresh, every call.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception as e:
        return {"checked": False, "warning": f"nvidia-smi unavailable: {e}", "total_used_mib": None}

    total_mib = 0
    processes = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        pid, name, used_mib = parts
        try:
            used_mib_int = int(used_mib)
        except ValueError:
            continue
        total_mib += used_mib_int
        processes.append({"pid": pid, "process_name": name, "used_memory_mib": used_mib_int})

    total_bytes = total_mib * 1024 * 1024
    warning = None
    if total_bytes >= threshold_bytes:
        warning = (
            f"GPU already has {total_mib} MiB in use by {len(processes)} process(es) "
            f"(threshold {threshold_bytes // 1024**3} GiB) -- unrelated usage may make safe launch uncertain. "
            "Not killing or modifying any process; review before proceeding."
        )
    return {"checked": True, "total_used_mib": total_mib, "processes": processes, "warning": warning}


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def print_dry_run_report(scope_filter=None):
    full_plan = build_trajectory_plan()
    plan = full_plan
    if scope_filter:
        plan = [t for t in full_plan if t["scope"] == scope_filter]

    primary = [t for t in plan if t["scope"] == "primary"]
    diagnostic = [t for t in plan if t["scope"] == "diagnostic"]
    groups = group_trajectories_by_policy(plan)

    print(f"Primary trajectories: {len(primary)} (expected 256{' unfiltered' if scope_filter else ''})")
    print(f"Diagnostic trajectories: {len(diagnostic)} (expected 16{' unfiltered' if scope_filter else ''})")
    print(f"Total trajectories: {len(primary) + len(diagnostic)} (expected 272{' unfiltered' if scope_filter else ''})")
    if scope_filter:
        print(f"NOTE: --scope={scope_filter} filter applied; counts above and below reflect the filtered subset only.")

    identities = [trajectory_identity(t) for t in plan]
    print(f"Unique identities: {len(set(identities))} (== total? {len(set(identities)) == len(identities)})")

    primary_groups = {k: v for k, v in groups.items() if k[0] == "primary"}
    diagnostic_groups = {k: v for k, v in groups.items() if k[0] == "diagnostic"}
    print(f"\nDistinct primary (layer,axis) policy groups: {len(primary_groups)}{' (expected 16)' if not scope_filter or scope_filter == 'primary' else ''}")
    print(f"Distinct diagnostic policy groups: {len(diagnostic_groups)}{' (expected 2)' if not scope_filter or scope_filter == 'diagnostic' else ''}")
    print(f"Total model loads under the grouped strategy: {len(groups)}{' (expected 18)' if not scope_filter else ''}")

    print("\nCounts per task:")
    for task in list(PRIMARY_TASKS) + list(DIAGNOSTIC_TASKS):
        n = sum(1 for t in plan if t["task"] == task)
        print(f"  {task}: {n}")

    print("\nCounts per (scope, layer, axis) group:")
    for key, members in groups.items():
        print(f"  scope={key[0]:10s} layer={key[1]:2d} axis={key[2]:5s}: {len(members)} trajectories")

    conflicts = check_no_conflicting_process()
    print(f"\nConflicting-process check: {'NONE FOUND' if not conflicts else conflicts}")
    preflight = gpu_preflight()
    print(f"GPU preflight (informational only, no CUDA/torch import): {preflight}")

    print(f"\nRequested decode steps (locked): {list(DEFAULT_REQUESTED_STEPS)}")
    print(f"Output roots (not written this round): {DEFAULT_OUTPUT_ROOT_PRIMARY}, {DEFAULT_OUTPUT_ROOT_DIAGNOSTIC}")
    print("\nNo torch/transformers/model/GPU import occurred; no scientific feature was collected.")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scope", choices=["primary", "diagnostic"], default=None, help="Restrict to one scope. Default: preview both.")
    p.add_argument("--output-root", default=None)
    p.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME)
    p.add_argument("--cache_dir", default="./cached_models")
    p.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    p.add_argument("--residual-length", type=int, default=DEFAULT_RESIDUAL_LENGTH)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS_HORIZON)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--run_label", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main():
    args = parse_args()

    if not args.dry_run:
        print(
            "Real Stage-H scientific collection is implemented but was NOT invoked this round "
            "(explicitly out of scope -- CPU tests + dry-run only). Refusing to proceed without "
            "--dry-run.",
            file=sys.stderr,
        )
        return 1

    return print_dry_run_report(scope_filter=args.scope)


if __name__ == "__main__":
    sys.exit(main())
