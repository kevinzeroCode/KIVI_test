"""Stage H3: decode-aware attention feature scientific collection runner.

Implements the locked Stage-H design (docs/stage_h0_pre_registration.md,
Stage H0-DECODE-AUDIT, Stage H0-PRE) for the full 272-trajectory
collection.

Stage H3A: CPU trajectory planning / record schema / resume logic /
--dry-run (torch-free -- verified: does not import torch/transformers).

Stage H3B (this round): the real GPU collection entry point (--run) is
now implemented, reusing H2's already-GPU-validated capture/reconstruction
functions directly (scripts.h2_attention_decode_parity_canary -- a scoped
spy on _apply_rotary_pos_emb_inplace + nn.functional.softmax installed
only around the target layer's own forward(), q_proj/v_proj/o_proj hooks,
reconstruct_key_step/reconstruct_value_step) rather than reimplementing
any of it. main() still requires an EXPLICIT --run flag (plus --scope) to
touch the GPU; omitting both --dry-run and --run refuses with an error.
NOTHING in this repository invokes --run this round -- see the Stage H3B
final report's explicit "DO NOT RUN GPU" confirmation.

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

# Frozen historical provenance references (never derived at runtime -- the
# live collection HEAD will necessarily be a later commit once this file
# itself is committed). See docs/stage_h0_pre_registration.md.
PREREGISTRATION_COMMIT = "c22881c9426a6e547f17b0f4c2c7598ec92c7a2b"  # "Preregister decode-aware attention feature pilot"
MEASUREMENT_PIPELINE_COMMIT = "1a3f2872b41939001ccb96cea068a2ecde26fc98"  # "Add decode-aware attention pilot infrastructure" (H1/H2/H3A)

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


def filter_plan(plan, scope=None, layer=None, axis=None, task=None, dataset_index=None, max_trajectories=None):
    """Part 15: pure, torch-free selector filtering shared by --dry-run and
    the real collector -- a single-trajectory or single-group debugging
    run uses exactly this same function (and the exact same downstream
    execution path) as the full 272-trajectory run, never a separate toy
    implementation. Filters are applied in the order listed; order does
    not affect the result since each is an independent predicate.
    `max_trajectories` truncates the already-filtered, deterministically-
    ordered plan (never reorders it).
    """
    out = plan
    if scope is not None:
        out = [t for t in out if t["scope"] == scope]
    if layer is not None:
        out = [t for t in out if t["layer_idx"] == layer]
    if axis is not None:
        out = [t for t in out if t["tensor_axis"] == axis]
    if task is not None:
        out = [t for t in out if t["task"] == task]
    if dataset_index is not None:
        out = [t for t in out if t["dataset_index"] == dataset_index]
    if max_trajectories is not None:
        out = out[:max_trajectories]
    return out


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
# Torch-free provenance helpers (real-collection preflight only; small,
# deliberately duplicated subprocess wrappers -- not scientific logic, so
# duplicating scripts.h2_attention_decode_parity_canary's equivalents here
# avoids an import that would transitively pull in torch before the git/
# process/GPU checks have even run).
# ---------------------------------------------------------------------------

def get_git_status_short():
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode()
    except Exception:
        return None


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def get_boot_id():
    try:
        with open("/proc/sys/kernel/random/boot_id", "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Part 12/18: manifest lifecycle + compact per-trajectory logging
# ---------------------------------------------------------------------------

def write_manifest(manifest_path, manifest):
    """Atomic write (tmp + fsync + os.replace) -- a reader never observes a
    half-written manifest."""
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, manifest_path)


def append_log(log_path, msg):
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


def trajectory_log_line(trajectory, status, generated_token_count=None, sampled_decode_steps=None, sample_attention_distortion=None, elapsed_s=None):
    """Part 18: compact, no-scientific-interpretation per-trajectory log
    entry -- identity + status + shape only, never a tensor."""
    return (
        f"scope={trajectory['scope']} task={trajectory['task']} dataset_index={trajectory['dataset_index']} "
        f"layer={trajectory['layer_idx']} axis={trajectory['tensor_axis']} status={status} "
        f"generated_tokens={generated_token_count} valid_steps={sampled_decode_steps} "
        f"distortion={sample_attention_distortion} elapsed_s={elapsed_s}"
    )


def append_validated_record(features_path, record):
    """Part 10: validate BEFORE write, fail closed (raises, writes
    nothing) on any structural problem. Append-only, flush+fsync so a
    completed record survives an abrupt process termination."""
    validate_record_shape(record)
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    with open(features_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Part 3/4/5: real collection -- default (GPU-touching) backend
# ---------------------------------------------------------------------------

def default_load_model_fn(model_name_or_path, cache_dir, layer_idx, k_bits, v_bits, seed, group_size, residual_length):
    """Loads ONE fixed (layer_idx, k_bits, v_bits) policy, reusing H2's
    already-GPU-validated scripts.h2_attention_decode_parity_canary.load_canary_model
    verbatim (never reimplemented). Deferred import keeps --dry-run and
    the CPU-mock orchestration tests (Part 16, dependency-injected)
    torch-free."""
    from scripts.h2_attention_decode_parity_canary import load_canary_model

    model, tokenizer, model_class_name = load_canary_model(
        model_name_or_path, cache_dir, k_bits, v_bits, seed,
        layer_idx=layer_idx, group_size=group_size, residual_length=residual_length,
    )
    model_short_name = model_name_or_path.split("/")[-1]
    return {"model": model, "tokenizer": tokenizer, "model_class_name": model_class_name, "model_short_name": model_short_name}


def default_release_model_fn(model_handle):
    """Part 13: release between policy groups -- del the real references,
    then a best-effort gc/cuda cleanup (cleanup only, never a substitute
    for the del above)."""
    import gc

    model = model_handle.pop("model", None)
    del model
    model_handle.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


_REAL_DATASET_CACHE_TASK_KEY = "_dataset"


def _get_dataset(task, dataset_cache):
    if task not in dataset_cache:
        from datasets import load_dataset

        dataset_cache[task] = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
    return dataset_cache[task]


def default_run_trajectory_fn(model_handle, trajectory, dataset_cache, max_new_tokens_horizon, group_size):
    """Part 3/5/6/7/8: runs exactly ONE trajectory against an already-
    loaded model_handle, with fully fresh state (a brand-new ShadowKVCache,
    a brand-new LayerCallCapture via generate_with_capture, fresh decode-
    step counters/metric accumulator -- nothing carried over from any
    other trajectory). Reuses, never reinvents:
      - scripts.h2_attention_decode_parity_canary.{build_prompt,
        generate_with_capture, reconstruct_key_step, reconstruct_value_step}
        (the exact GPU-validated H2 capture/reconstruction path)
      - utils.attention_decode_features.{ShadowKVCache, key_decode_distortion,
        value_decode_distortion} (the exact locked measurement formulas)
      - utils.generation_semantics.resolve_generate_kwargs (real per-task
        generation kwargs, including samsum's min_length/eos_token_id --
        only max_new_tokens is overridden down to the Stage-H horizon; every
        other task-specific kwarg passes through unmodified).
    Returns a dict of the raw scalars build_trajectory_record() needs
    (never a tensor/matrix/cache) plus the executed step_distortions.
    """
    import json as _json

    from scripts.h2_attention_decode_parity_canary import (
        build_prompt,
        generate_with_capture,
        reconstruct_key_step,
        reconstruct_value_step,
    )
    from utils.attention_decode_features import ShadowKVCache, key_decode_distortion, value_decode_distortion
    from utils.generation_semantics import resolve_generate_kwargs

    model = model_handle["model"]
    tokenizer = model_handle["tokenizer"]
    model_short_name = model_handle["model_short_name"]

    task = trajectory["task"]
    dataset_index = trajectory["dataset_index"]
    layer_idx = trajectory["layer_idx"]
    axis = trajectory["tensor_axis"]
    k_bits, v_bits = trajectory["k_bits"], trajectory["v_bits"]

    dataset = _get_dataset(task, dataset_cache)
    if not (0 <= dataset_index < len(dataset)):
        raise PilotConfigError(f"dataset_index {dataset_index} out of range for task {task!r} (len={len(dataset)}) -- refusing to load a shifted row")
    json_obj = dataset[dataset_index]  # identity verified by the bounds check above; index is never re-derived

    prompt = build_prompt(tokenizer, model_short_name, json_obj, task=task)

    with open(os.path.join(REPO_ROOT, "config", "dataset2maxlen.json"), "r", encoding="utf-8") as f:
        dataset2maxlen = _json.load(f)
    inp = tokenizer(prompt, truncation=False, return_tensors="pt")
    context_length = inp.input_ids.shape[-1]
    full_kwargs = resolve_generate_kwargs(task, tokenizer, context_length, dataset2maxlen[task])
    # Truncate the generation horizon only; every other task-specific kwarg
    # (samsum's min_length/eos_token_id) passes through unmodified -- greedy
    # decoding is prefix-deterministic, so this reproduces an exact prefix
    # of what the full-length task generation would produce (established
    # generically for the un-truncated-kwargs case by H2's
    # horizon_prefix_match check).
    max_new_tokens = min(full_kwargs.pop("max_new_tokens"), max_new_tokens_horizon)
    full_kwargs.pop("num_beams", None)
    full_kwargs.pop("do_sample", None)
    full_kwargs.pop("temperature", None)
    full_kwargs.pop("top_p", None)
    extra_kwargs = full_kwargs  # only samsum contributes anything here (min_length, eos_token_id)

    generated_ids, layer_capture, prompt_input_tokens = generate_with_capture(
        model, tokenizer, prompt, max_new_tokens, capture=True, layer_idx=layer_idx, extra_generate_kwargs=extra_kwargs,
    )

    calls = layer_capture.calls
    assert len(calls) >= 1, "fresh LayerCallCapture produced no calls -- capture state was not actually fresh"
    attn = model.model.layers[layer_idx].self_attn
    head_dim = attn.head_dim
    num_kv_heads = attn.num_key_value_heads
    for call in calls:
        raw_v = call["v_proj_raw"]
        q_len = raw_v.shape[1]
        call["v_current"] = raw_v.view(raw_v.shape[0], q_len, num_kv_heads, head_dim).transpose(1, 2).contiguous()

    prefill_call = calls[0]
    decode_calls = calls[1:]

    shadow = ShadowKVCache()
    assert shadow.token_count == 0, "fresh ShadowKVCache was not actually empty before seeding"
    shadow.seed_prefill(prefill_call["k_post_rope"], prefill_call["v_current"])

    step_distortions = {}
    for step_idx, call in enumerate(decode_calls, start=1):
        if axis == "key":
            L_kivi, L_fp16 = reconstruct_key_step(call, group_size, k_bits, shadow, head_dim, num_kv_heads)
            step_distortions[step_idx] = key_decode_distortion(L_kivi, L_fp16)
        else:
            shadow.append_decode_step(call["k_post_rope"], call["v_current"])
            O_value_kivi, O_fp16, _ = reconstruct_value_step(call, group_size, v_bits, shadow, head_dim)
            step_distortions[step_idx] = value_decode_distortion(O_value_kivi, O_fp16)

    del shadow, layer_capture, calls, prefill_call, decode_calls
    return {
        "prompt_input_tokens": prompt_input_tokens,
        "actual_generated_token_count": len(generated_ids),
        "step_distortions": step_distortions,
    }


def run_real_collection(
    args,
    load_model_fn=default_load_model_fn,
    release_model_fn=default_release_model_fn,
    run_trajectory_fn=default_run_trajectory_fn,
    git_status_fn=get_git_status_short,
    git_commit_fn=get_git_commit,
    boot_id_fn=get_boot_id,
    conflict_check_fn=check_no_conflicting_process,
    gpu_preflight_fn=gpu_preflight,
):
    """Part 2/4/11/12/14/19: the real (or, under test, dependency-injected
    fake) execution path. Every check up to and including the GPU
    preflight is torch-free and runs BEFORE any of the injected backend
    functions is called, so a refusal never triggers a model load.

    Fixed-policy grouping (Part 4): one load_model_fn call per (scope,
    layer_idx, tensor_axis) group with at least one remaining trajectory;
    release_model_fn is called exactly once when a group finishes, before
    the next group's load_model_fn call -- never two model handles alive
    at once.
    """
    git_status = git_status_fn()
    if git_status is None:
        raise PilotConfigError("could not determine git status; refusing real scientific collection for provenance safety")
    if git_status.strip():
        raise PilotConfigError(
            "git working tree is not clean; refusing real scientific collection so every record's "
            f"git_commit is trustworthy provenance. git status --short:\n{git_status}"
        )
    collection_head = git_commit_fn()
    if not collection_head:
        raise PilotConfigError("could not determine git HEAD commit; refusing real scientific collection")

    conflicts = conflict_check_fn()
    if conflicts:
        raise PilotLockError(f"conflicting process(es) detected, refusing to launch: {conflicts}")

    preflight = gpu_preflight_fn()
    if preflight.get("warning"):
        # No override flag exists (deliberate Stage H3B scope choice --
        # see the final report): a busy GPU always refuses, unconditionally.
        raise PilotConfigError(f"GPU preflight refused real scientific launch: {preflight['warning']}")

    if args.scope is None:
        raise PilotConfigError("--scope {primary,diagnostic} is required for --run so primary/diagnostic outputs are never mixed under one output root")

    plan = build_trajectory_plan()
    plan = filter_plan(
        plan, scope=args.scope, layer=args.layer, axis=args.axis, task=args.task,
        dataset_index=args.dataset_index, max_trajectories=args.max_trajectories,
    )
    if not plan:
        raise PilotConfigError("the requested selectors produced an empty trajectory plan; nothing to run")
    for _t in plan:
        validate_policy_bits(_t["tensor_axis"], _t["k_bits"], _t["v_bits"])
    if args.layer is None:
        # Full scope guard only applies when the caller hasn't deliberately
        # narrowed to a single layer via --layer (a legitimate debugging
        # selector, Part 15).
        validate_scope_guard(args.scope, sorted({t["layer_idx"] for t in plan}))

    output_root = args.output_root or (DEFAULT_OUTPUT_ROOT_PRIMARY if args.scope == "primary" else DEFAULT_OUTPUT_ROOT_DIAGNOSTIC)
    run_label = args.run_label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = os.path.join(output_root, run_label)
    os.makedirs(run_dir, exist_ok=True)
    features_path = os.path.join(run_dir, "features.jsonl")
    manifest_path = os.path.join(run_dir, "manifest.json")
    log_path = os.path.join(run_dir, "collection.log")

    lock_fh = acquire_pilot_lock(os.path.join(run_dir, "pilot.lock"))
    try:
        completed = load_completed_records(features_path, expected_git_commit=collection_head)  # fail-closed (Part 11)
        remaining = plan_remaining_trajectories(plan, set(completed.keys()))

        manifest = {
            "scope": args.scope,
            "collection_head": collection_head,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "measurement_pipeline_commit": MEASUREMENT_PIPELINE_COMMIT,
            "planned_count": len(plan),
            "completed_count": len(completed),
            "skipped_resumed_count": len(completed),
            "failure_identity": None,
            "start_boot_id": boot_id_fn(),
            "end_boot_id": None,
            "nonfinite_status": "not_detected",
            "exit_status": None,
            "state": "RUNNING",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "ended_at": None,
        }
        write_manifest(manifest_path, manifest)
        append_log(log_path, f"collection start scope={args.scope} planned={len(plan)} already_completed={len(completed)} remaining={len(remaining)}")

        dataset_cache = {}
        groups = group_trajectories_by_policy(remaining)
        current_trajectory = None  # tracks the in-progress trajectory for accurate failure-identity reporting
        try:
            for (scope, layer_idx, axis), members in groups.items():
                current_trajectory = None  # reset: a load_model_fn failure must not misattribute to the previous group's last trajectory
                append_log(log_path, f"loading model for policy group scope={scope} layer={layer_idx} axis={axis} ({len(members)} trajectories)")
                model_handle = load_model_fn(
                    args.model_name_or_path, args.cache_dir, layer_idx, members[0]["k_bits"], members[0]["v_bits"],
                    args.seed, args.group_size, args.residual_length,
                )
                try:
                    for t in members:
                        current_trajectory = t
                        t0 = datetime.now(timezone.utc)
                        result = run_trajectory_fn(model_handle, t, dataset_cache, args.max_new_tokens, args.group_size)
                        elapsed_s = (datetime.now(timezone.utc) - t0).total_seconds()
                        record = build_trajectory_record(
                            t, args.model_name_or_path, collection_head, args.seed,
                            result["prompt_input_tokens"], result["actual_generated_token_count"], result["step_distortions"],
                        )
                        append_validated_record(features_path, record)  # Part 10: fail-closed before write
                        manifest["completed_count"] += 1
                        write_manifest(manifest_path, manifest)
                        append_log(log_path, trajectory_log_line(
                            t, "completed", result["actual_generated_token_count"],
                            record["sampled_decode_steps"], record["sample_attention_distortion"], round(elapsed_s, 3),
                        ))
                        current_trajectory = None  # this trajectory is now durably recorded; not "in progress" anymore
                finally:
                    release_model_fn(model_handle)
        except Exception as e:  # noqa: BLE001 -- fail closed: preserve records/logs, write failure state, never fabricate
            failed_identity = trajectory_identity(current_trajectory) if current_trajectory is not None else None
            manifest["failure_identity"] = list(failed_identity) if failed_identity else None
            manifest["state"] = "FAILED"
            manifest["exit_status"] = f"{type(e).__name__}: {e}"
            manifest["end_boot_id"] = boot_id_fn()
            manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
            write_manifest(manifest_path, manifest)
            append_log(log_path, f"FAILED identity={failed_identity} error={type(e).__name__}: {e}")
            raise

        final_completed = load_completed_records(features_path, expected_git_commit=collection_head)
        final_identities = set(final_completed.keys())
        expected_identities = {trajectory_identity(t) for t in plan}
        manifest["completed_count"] = len(final_completed)
        manifest["end_boot_id"] = boot_id_fn()
        manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
        if final_identities == expected_identities:
            manifest["state"] = "COMPLETE"
            manifest["exit_status"] = "ok"
        else:
            manifest["state"] = "FAILED"
            manifest["exit_status"] = f"planned count did not validate: expected {len(expected_identities)}, got {len(final_identities)}"
        write_manifest(manifest_path, manifest)
        append_log(log_path, f"collection end state={manifest['state']} completed={manifest['completed_count']}/{manifest['planned_count']}")
        return 0 if manifest["state"] == "COMPLETE" else 1
    finally:
        release_pilot_lock(lock_fh)


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
    p.add_argument("--dry-run", action="store_true", help="CPU-only planning report. Torch-free. Mutually exclusive with --run.")
    p.add_argument("--run", action="store_true", help="Real scientific GPU collection. Requires --scope. See Part 22: NOT invoked by anything in this round.")
    # Part 15: safe debugging/execution selectors. These narrow the SAME
    # trajectory plan/execution path used by the full run -- a 1-trajectory
    # or 1-group debugging run is never a separate toy implementation.
    p.add_argument("--layer", type=int, default=None, help="Restrict to one layer_idx.")
    p.add_argument("--axis", choices=["key", "value"], default=None, help="Restrict to one tensor_axis.")
    p.add_argument("--task", default=None, help="Restrict to one LongBench task.")
    p.add_argument("--dataset-index", type=int, default=None, help="Restrict to one dataset_index.")
    p.add_argument("--max-trajectories", type=int, default=None, help="Truncate the (already-filtered) plan to at most N trajectories.")
    return p.parse_args(argv)


def main():
    args = parse_args()

    if args.dry_run and args.run:
        print("--dry-run and --run are mutually exclusive.", file=sys.stderr)
        return 1

    if args.dry_run:
        return print_dry_run_report(scope_filter=args.scope)

    if args.run:
        # Part 22: this call is real and would touch the GPU -- nothing in
        # this repository invokes main() with --run this round.
        try:
            return run_real_collection(args)
        except (PilotConfigError, PilotLockError, ResumeError) as e:
            print(f"Refusing real scientific collection: {type(e).__name__}: {e}", file=sys.stderr)
            return 1

    print(
        "Neither --dry-run nor --run was given. Real Stage-H scientific collection requires the "
        "explicit --run flag (plus --scope); GPU collection never happens merely because --dry-run "
        "is absent. Refusing to proceed.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
