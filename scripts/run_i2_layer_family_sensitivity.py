"""Stage I2A: infrastructure for the layer x quantizer-family sensitivity
experiment.

16 conditions (8 layers x 2 families: kivi, rotation_kivi), all at a FIXED
target-layer precision K2/V16, 6 LongBench tasks at full test-split size,
sequential single-instance execution. This module builds the full I2
driver -- policy generation/validation, resume-safe per-task generation,
single-instance locking, crash-safe manifest, host monitoring -- plus a
separate, explicit I2B GPU route-validity canary mode. Running either mode
for real (GPU generation) is a separate, explicit later step (I2B, then
I2C); --dry-run and --canary-dry-run exercise everything except that.

Never touches outputs/layer_sensitivity_pilot/, outputs/
layer_attention_feature_pilot/, outputs/i1_rotation_kivi_canary/, or pred/.
Writes exclusively under --output-root (default
outputs/i2_layer_family_sensitivity/).

Scientific question, condition/task lock, aggregation, bootstrap, and
crossover definitions are frozen in docs/stage_i2_pre_registration.md --
this module implements the mechanics only and must not redefine any of
those.

Usage (dry run, safe, CPU-only, no model import, no GPU):
    ./.venv/bin/python scripts/run_i2_layer_family_sensitivity.py --dry-run

Usage (I2B canary dry run, safe, CPU-only, reports the canary plan without
touching the GPU):
    ./.venv/bin/python scripts/run_i2_layer_family_sensitivity.py --mode canary --dry-run

Usage (I2C formal generation -- NOT executed in Stage I2A; requires an
explicit later authorization and I2B to have passed first):
    CUDA_VISIBLE_DEVICES=0 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \\
    ./.venv/bin/python scripts/run_i2_layer_family_sensitivity.py --run

Usage (I2B canary -- NOT executed in Stage I2A):
    CUDA_VISIBLE_DEVICES=0 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \\
    ./.venv/bin/python scripts/run_i2_layer_family_sensitivity.py --mode canary --run
"""
import argparse
import errno
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from utils.i2_layer_family_conditions import (  # noqa: E402
    I2_FAMILIES,
    I2_GROUP_SIZE,
    I2_K_BITS,
    I2_LAYERS,
    I2_RESIDUAL_LENGTH,
    I2_TASK_COUNTS,
    I2_V_BITS,
    TOTAL_CONDITIONS,
    TOTAL_PLANNED_ROWS,
    discover_and_validate_i2_policies,
    output_dir_name,
    total_example_count,
    write_i2_policies,
)
from utils.jsonl_integrity import inspect_jsonl  # noqa: E402

DEFAULT_POLICIES_DIR = os.path.join(REPO_ROOT, "analysis", "policies", "i2_layer_family_sensitivity")
DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "outputs", "i2_layer_family_sensitivity")

# I2B canary constants (Section 20 of docs/stage_i2_pre_registration.md).
# NOT used for efficacy conclusions -- runtime/cache-shape route validity
# only. Reuses the same already-fixed lcc calibration sample as the
# established H2/I1 canaries (never a new, unvetted sample), specifically
# because it is a long-context sample and this canary's whole purpose is to
# run long enough to observe a real quantized Key prefix rollover -- unlike
# H2/C1's short 6-token canaries, which never exercised one (this is the
# exact reporting gap Section 18 of the pre-registration closes).
CANARY_LAYERS = (0, 18, 31)
CANARY_FAMILIES = I2_FAMILIES
CANARY_TASK = "lcc"
CANARY_DATASET_INDEX = 122
CANARY_SEED = 42
# Deliberately > I2_RESIDUAL_LENGTH (128) so at least one Key-cache rollover
# (bulk, ~114-step cadence after prefill -- docs/stage_h0_pre_registration.md
# Section 0) has a chance to fire during this canary; NOT bound by lcc's
# normal task max_gen (64), since this is a structural route-validity check,
# not a scored generation. If a real run does not observe a quantized
# prefix, the canary must be reported inconclusive/blocked -- never silently
# passed (see compute_i2b_canary_gate).
CANARY_MAX_NEW_TOKENS = 140

# Stage I2C0: the pinned formal LongBench revision. docs/stage_i2_pre_registration.md
# Section 6 flagged that no existing code path in this repo pins a
# `revision=` kwarg, so the installed `datasets` package's current default
# HEAD revision could silently drift out from under a multi-day, resumable
# 23,200-row formal run. This value is a fixed, previously-OBSERVED
# revision hash (recorded in environment/README.md / EXPERIMENT_STATUS.md
# from an earlier one-off smoke test) -- frozen here for I2C formal
# generation only. Historical H2/I1/Stage-D runners are deliberately left
# unpinned/unmodified; this constant is consumed ONLY by the I2C formal
# path (run_condition's default dataset_revision, i2c_dataset_identity_preflight).
I2_DATASET_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"

# Filenames for the top-level formal-run lock/manifest, both written
# directly under --output-root (a sibling of canary/, never inside it --
# the canary subtree is never interpreted as a formal condition).
I2C_FORMAL_LOCK_NAME = ".i2c_formal.lock"
I2C_FORMAL_MANIFEST_NAME = "i2c_formal_manifest.json"

# Any of these process name patterns running (other than this invocation
# itself) means a conflicting generation writer might already be active.
CONFLICTING_PROCESS_PATTERNS = [
    "pred_long_bench.py",
    "run_layer_sensitivity_pilot.py",
    "run_layer_attention_feature_pilot.py",
    "i1_rotation_kivi_parity_canary.py",
    "run_i2_layer_family_sensitivity.py",
]

# dataset_revision participates in the resume-critical config keys
# (Stage I2C0, Section 4): a condition directory generated under a
# different (or missing -- pre-I2C0 legacy) dataset revision must never be
# silently resumed as the same formal condition.
I2_CONDITION_CORE_KEYS = [
    "model_name_or_path", "max_length", "group_size", "residual_length", "seed", "policy_hash", "dataset_revision",
]


class I2LockError(RuntimeError):
    pass


class I2ConflictError(RuntimeError):
    pass


class I2ResumeError(RuntimeError):
    pass


class I2ConfigError(RuntimeError):
    pass


class I2CanaryError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Single-instance safety -- identical pattern to
# scripts/run_layer_sensitivity_pilot.py's acquire_pilot_lock (no shared
# lock util exists in this repo; copied per established per-script
# convention rather than introducing a new shared module for this alone).
# ---------------------------------------------------------------------------

def acquire_i2_lock(lock_path):
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
        raise I2LockError(
            f"I2A lock {lock_path} is already held (recorded holder: {holder}). "
            "Refusing to start a second writer against the same output root."
        )
    fh.seek(0)
    fh.truncate()
    fh.write(f"pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()}\n")
    fh.flush()
    return fh


def release_i2_lock(fh):
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def check_no_conflicting_process(patterns=CONFLICTING_PROCESS_PATTERNS, self_pid=None):
    """Identical false-positive-avoidance logic to
    run_layer_sensitivity_pilot.check_no_conflicting_process (see that
    function's docstring for the concrete false positives this guards
    against: bash -c tool wrapper shells and stale pgrep-loop artifacts)."""
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


# ---------------------------------------------------------------------------
# Resume safety -- identical exact-count-only semantics to
# run_layer_sensitivity_pilot.resolve_pilot_task_resume_plan (deliberately
# NOT reused via import to keep this module's error type (I2ResumeError)
# distinct and self-contained, but the logic is byte-for-byte the same
# contract: a final file must have EXACTLY the expected row count, never
# >=; any invalid row fails closed; an over-count partial fails closed).
# ---------------------------------------------------------------------------

def resolve_i2_task_resume_plan(task, out_path, partial_path, expected):
    """Returns (action, active_path, done): action in {"skip", "finalize",
    "resume", "start"}."""
    if os.path.exists(out_path):
        info = inspect_jsonl(out_path)
        if info.invalid_rows:
            raise I2ResumeError(
                f"{task}: final file {out_path} has {info.invalid_rows} invalid JSON row(s), "
                f"first at line {info.first_invalid_line}. Refusing to skip a corrupted final file."
            )
        if info.valid_rows == expected:
            return "skip", out_path, info.valid_rows
        raise I2ResumeError(
            f"{task}: final file {out_path} has {info.valid_rows} valid rows, expected exactly "
            f"{expected} (over-count is exactly as unsafe as under-count -- both indicate an "
            "inconsistent state that must be inspected manually, never silently accepted)."
        )

    info = inspect_jsonl(partial_path)
    if info.invalid_rows:
        raise I2ResumeError(
            f"{task}: partial file {partial_path} has {info.invalid_rows} invalid JSON line(s), "
            f"first invalid line {info.first_invalid_line}, valid prefix {info.valid_prefix_rows}. "
            "Resume start_idx cannot be safely determined from a corrupted partial."
        )
    if info.valid_rows > expected:
        raise I2ResumeError(
            f"{task}: partial file {partial_path} has {info.valid_rows} valid rows, more than "
            f"expected {expected}. Refusing to resume from an over-count partial."
        )
    if info.valid_rows == expected:
        return "finalize", partial_path, info.valid_rows
    if info.valid_rows > 0:
        return "resume", partial_path, info.valid_rows
    return "start", partial_path, 0


def compute_resume_report(conditions, task_counts, output_root):
    report = []
    for c in conditions:
        condition_dir = os.path.join(output_root, output_dir_name(c))
        for task, expected in task_counts.items():
            out_path = os.path.join(condition_dir, f"{task}.jsonl")
            partial_path = out_path + ".partial"
            try:
                action, _active_path, done = resolve_i2_task_resume_plan(task, out_path, partial_path, expected)
                report.append({"condition_id": c["condition_id"], "task": task, "action": action, "done": done, "expected": expected})
            except I2ResumeError as e:
                report.append({"condition_id": c["condition_id"], "task": task, "action": "ERROR", "done": None, "expected": expected, "error": str(e)})
    return report


def finalize_task_if_ready(action, active_path, out_path, expected):
    info = inspect_jsonl(active_path)
    if info.invalid_rows or info.valid_rows != expected:
        raise I2ResumeError(
            f"Refusing to finalize {active_path}: valid_rows={info.valid_rows} "
            f"invalid_rows={info.invalid_rows}, expected exactly {expected} valid rows."
        )
    os.replace(active_path, out_path)


# ---------------------------------------------------------------------------
# Per-condition directory / config
# ---------------------------------------------------------------------------

def build_condition_run_config(condition, model_name_or_path, max_length, group_size, residual_length, seed, task_counts=I2_TASK_COUNTS, dataset_revision=I2_DATASET_REVISION):
    return {
        "model_name_or_path": model_name_or_path,
        "condition_id": condition["condition_id"],
        "layer_idx": condition["layer_idx"],
        "family": condition["family"],
        "k_bits": condition["k_bits"],
        "v_bits": condition["v_bits"],
        "policy_path": condition["policy_path"],
        "policy_hash": condition["resolved_policy_hash"],
        "max_length": max_length,
        "group_size": group_size,
        "residual_length": residual_length,
        "seed": seed,
        "dataset_revision": dataset_revision,
        "tasks": list(task_counts),
        "expected_task_counts": dict(task_counts),
        "experiment": "i2_layer_family_sensitivity_stage_i2a",
    }


def prepare_condition_directory(condition_dir, run_config):
    """Analogue of run_layer_sensitivity_pilot.prepare_condition_directory:
    creates the directory + writes run_config.json on first use; on a
    resumed run, fails closed if any I2_CONDITION_CORE_KEYS entry
    (crucially including policy_hash) doesn't match exactly."""
    run_config_path = os.path.join(condition_dir, "run_config.json")
    if not os.path.exists(condition_dir):
        os.makedirs(condition_dir)
        with open(run_config_path, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)
        return "created"

    if not os.path.exists(run_config_path):
        raise I2ConfigError(
            f"{condition_dir} exists but has no run_config.json -- unverified state, refusing to reuse."
        )
    with open(run_config_path, "r", encoding="utf-8") as f:
        existing = json.load(f)
    mismatches = {
        key: (existing.get(key), run_config.get(key))
        for key in I2_CONDITION_CORE_KEYS
        if existing.get(key) != run_config.get(key)
    }
    if mismatches:
        raise I2ConfigError(
            f"Refusing to resume {condition_dir}: run_config.json does not match the current "
            f"condition. Mismatched keys (existing, requested): {mismatches}"
        )
    return "validated"


# ---------------------------------------------------------------------------
# I2A-wide manifest -- crash-safe via write-temp + os.replace.
# ---------------------------------------------------------------------------

def build_initial_i2_manifest(conditions, args, task_counts=I2_TASK_COUNTS):
    return {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "experiment": "i2_layer_family_sensitivity_stage_i2a",
        "num_conditions": len(conditions),
        "tasks": list(task_counts),
        "expected_task_counts": dict(task_counts),
        "total_examples": total_example_count(task_counts, len(conditions)),
        "conditions": [
            {
                "condition_id": c["condition_id"],
                "layer_idx": c["layer_idx"],
                "family": c["family"],
                "k_bits": c["k_bits"],
                "v_bits": c["v_bits"],
                "policy_path": c["policy_path"],
                "policy_hash": c["resolved_policy_hash"],
                "tasks": list(task_counts),
                "expected_task_counts": dict(task_counts),
                "status": "pending",
                "output_dir": c["output_dir"],
            }
            for c in conditions
        ],
    }


def write_i2_manifest_atomic(path, manifest):
    path = str(path)
    tmp_path = f"{path}.tmp{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def read_i2_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_condition_status(manifest, condition_id_, status):
    for c in manifest["conditions"]:
        if c["condition_id"] == condition_id_:
            c["status"] = status
            return manifest
    raise I2ConfigError(f"condition_id {condition_id_} not found in manifest")


# ---------------------------------------------------------------------------
# Host/GPU monitoring -- copied per established per-script convention (no
# shared util module exists in this repo for this concern; see
# scripts/run_layer_sensitivity_pilot.py / scripts/i1_rotation_kivi_parity_canary.py).
# ---------------------------------------------------------------------------

def sample_host_metrics():
    ts = datetime.now(timezone.utc).astimezone().isoformat()
    try:
        uptime = subprocess.check_output(["uptime"]).decode().strip()
    except Exception:
        uptime = None
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,power.draw", "--format=csv,noheader"]
        ).decode().strip()
    except Exception:
        gpu = None
    return {"timestamp": ts, "uptime": uptime, "gpu": gpu}


def get_boot_id():
    try:
        with open("/proc/sys/kernel/random/boot_id", "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def get_git_status_short():
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode()
    except Exception:
        return None


def gpu_snapshot():
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        procs = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return {"gpu": gpu, "compute_processes": procs}
    except Exception as e:
        return {"error": str(e)}


def gpu_preflight(threshold_bytes=10 * 1024**3):
    """Same threshold/shape as scripts/i1_rotation_kivi_parity_canary.py's
    gpu_preflight: refuses to proceed if pre-existing GPU memory usage looks
    substantial (>10 GiB) rather than assuming the device is free."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception as e:
        return {"checked": False, "error": str(e)}
    total_used = 0
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
            used_bytes = int(used_mib) * 1024 * 1024
        except ValueError:
            continue
        total_used += used_bytes
        processes.append({"pid": pid, "process_name": name, "used_mib": used_mib})
    warning = None
    if total_used > threshold_bytes:
        warning = f"pre-existing GPU memory usage {total_used} bytes exceeds threshold {threshold_bytes} bytes"
    return {"checked": True, "total_used_mib": total_used // (1024 * 1024), "processes": processes, "warning": warning}


# ---------------------------------------------------------------------------
# Real per-condition generation (NOT invoked by --dry-run; not executed in
# this round -- see docs/stage_i2_pre_registration.md for the staged
# I2B -> I2C authorization plan).
# ---------------------------------------------------------------------------

def run_condition(condition, args, dataset2prompt, dataset2maxlen, task_counts=I2_TASK_COUNTS, dataset_revision=I2_DATASET_REVISION):  # pragma: no cover - GPU path, exercised only by explicit --mode generation --run with the exact formal universe
    """Runs one I2C formal condition end to end: build the KIVI model with
    this condition's one-target-layer policy (target layer =
    condition['family'] at K2/V16, all other layers kivi K16/V16), generate
    exactly `task_counts` (the 6-task formal set) with durable per-row
    fsync + exact-count resume, host-monitor sampling every 30s, then
    finalize. Deferred (heavy) imports happen here only -- --dry-run never
    reaches this function. Structurally identical to
    run_layer_sensitivity_pilot.run_condition, generalized from
    (layer, axis) to (layer, family).

    `dataset_revision` is pinned (Stage I2C0) and forwarded verbatim to
    every load_dataset() call and into run_config.json/manifest.json --
    never silently defaulted to the installed `datasets` package's current
    HEAD revision."""
    import gc
    import threading

    import torch
    from datasets import load_dataset

    import pred_long_bench as plb
    from utils.generation_semantics import NO_BUILD_CHAT_DATASETS, resolve_generate_kwargs

    condition_dir = os.path.join(args.output_root, output_dir_name(condition))
    run_config = build_condition_run_config(
        condition, args.model_name_or_path, args.max_length, args.group_size, args.residual_length, args.seed,
        task_counts=task_counts, dataset_revision=dataset_revision,
    )
    status = prepare_condition_directory(condition_dir, run_config)

    monitor_log_path = os.path.join(condition_dir, "host_monitor.log")
    stop_monitor = threading.Event()

    def _monitor_loop():
        with open(monitor_log_path, "a", encoding="utf-8") as mf:
            while not stop_monitor.is_set():
                mf.write(json.dumps(sample_host_metrics()) + "\n")
                mf.flush()
                stop_monitor.wait(30)

    monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
    monitor_thread.start()

    boot_id_start = get_boot_id()
    start_time = datetime.now(timezone.utc).astimezone().isoformat()
    exit_code = 1
    hook_handle = None
    nan_inf_flags = {"seen_nonfinite": False}
    try:
        plb.seed_everything(args.seed)

        class _ModelArgs:
            def __init__(self):
                self.model_name_or_path = args.model_name_or_path
                self.k_bits = 16
                self.v_bits = 16
                self.group_size = args.group_size
                self.residual_length = args.residual_length

        class _TrainingArgs:
            def __init__(self):
                self.cache_dir = args.cache_dir

        model, tokenizer, model_class_name = plb.build_model_and_tokenizer(
            _ModelArgs(), _TrainingArgs(), torch.float16, use_kivi_model=True,
            resolved_layer_policy=condition["resolved_layer_policy"],
        )
        model.eval()
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_short_name = args.model_name_or_path.split("/")[-1]

        def _lm_head_hook(module, inp, output):
            if not torch.isfinite(output).all():
                nan_inf_flags["seen_nonfinite"] = True

        hook_handle = model.lm_head.register_forward_hook(_lm_head_hook)

        for task, expected in task_counts.items():
            out_path = os.path.join(condition_dir, f"{task}.jsonl")
            partial_path = out_path + ".partial"
            action, active_path, done = resolve_i2_task_resume_plan(task, out_path, partial_path, expected)
            if action == "skip":
                continue
            if action == "finalize":
                finalize_task_if_ready(action, active_path, out_path, expected)
                continue

            data = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True, revision=dataset_revision)
            prompt_format = dataset2prompt[task]
            max_gen = dataset2maxlen[task]

            with open(active_path, "a", encoding="utf-8") as f:
                for idx in range(done, expected):
                    json_obj = data[idx]
                    prompt = prompt_format.format(**json_obj)
                    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
                    if len(tokenized_prompt) > args.max_length:
                        half = int(args.max_length / 2)
                        prompt = (
                            tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
                            + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
                        )
                    if task not in NO_BUILD_CHAT_DATASETS:
                        prompt = plb.build_chat(tokenizer, prompt, model_short_name)
                    inp = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
                    context_length = inp.input_ids.shape[-1]
                    generate_kwargs = resolve_generate_kwargs(task, tokenizer, context_length, max_gen)
                    with torch.no_grad():
                        output = model.generate(**inp, **generate_kwargs)[0]
                    pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
                    pred = plb.post_process(pred, model_short_name)
                    # dataset_index is written explicitly (beyond the pred/answers/
                    # all_classes/length fields pred_long_bench.py/run_layer_sensitivity_pilot.py
                    # write) so analysis/analyze_i2_layer_family_sensitivity.py can validate
                    # duplicate/missing-index pairing directly instead of relying only on
                    # row order (Section 9's "task, dataset index, layer" pairing contract).
                    row = {"dataset_index": idx, "pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}
                    json.dump(row, f, ensure_ascii=False)
                    f.write("\n")
                    f.flush()
                    os.fsync(f.fileno())
                    del inp, output, tokenized_prompt
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            finalize_task_if_ready("finalize", active_path, out_path, expected)

        exit_code = 0
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        stop_monitor.set()
        monitor_thread.join(timeout=5)
        end_time = datetime.now(timezone.utc).astimezone().isoformat()
        boot_id_end = get_boot_id()
        manifest = {
            "condition_id": condition["condition_id"],
            "start_time": start_time,
            "end_time": end_time,
            "boot_id_start": boot_id_start,
            "boot_id_end": boot_id_end,
            "boot_id_stable": boot_id_start == boot_id_end,
            "exit_code": exit_code,
            "nonfinite_logits_seen": nan_inf_flags["seen_nonfinite"],
            "git_commit": get_git_commit(),
            "dataset_revision": dataset_revision,
        }
        with open(os.path.join(condition_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return exit_code


# ---------------------------------------------------------------------------
# I2B: GPU route-validity canary (Section 20 of the pre-registration).
# Implemented but NOT run in Stage I2A. Proves runtime policy resolution +
# real K2 quantized-Key-prefix presence + valid cache shapes + finite
# generation + hook neutrality for layers {0, 18, 31} x both families, K2/
# V16, using one already-fixed long-context sample. Does NOT require
# Rotation to score better or produce the same tokens as standard KIVI --
# this is a structural/runtime check only, never an efficacy conclusion.
# ---------------------------------------------------------------------------

def compute_i2b_canary_gate(condition_result):
    """Pure function (no torch/GPU dependency), CPU-unit-testable on
    synthetic input -- same discipline as
    scripts/i1_rotation_kivi_parity_canary.py's compute_c0_structural_gate.
    `condition_result` is one (layer, family) canary observation:
        {
            "layer_idx": int, "family": str,
            "policy_resolved_correctly": bool,
            "quantized_prefix_observed": bool,  # key_quant_trans not None at >=1 step
            "cache_shapes_valid": bool,
            "generation_finite": bool,
            "hook_neutral": bool,
            "ran_without_error": bool,
        }
    Returns {..., "condition_pass": bool}. A False
    quantized_prefix_observed means the canary is INCONCLUSIVE for K2 route
    validity at this condition -- it must not be silently treated as a
    pass; report it explicitly rather than weakening this requirement."""
    criteria = {
        "policy_resolved_correctly": bool(condition_result["policy_resolved_correctly"]),
        "quantized_prefix_observed": bool(condition_result["quantized_prefix_observed"]),
        "cache_shapes_valid": bool(condition_result["cache_shapes_valid"]),
        "generation_finite": bool(condition_result["generation_finite"]),
        "hook_neutral": bool(condition_result["hook_neutral"]),
        "ran_without_error": bool(condition_result["ran_without_error"]),
    }
    result = dict(condition_result)
    result.update(criteria)
    result["condition_pass"] = all(criteria.values())
    return result


def i2b_canary_conditions(layers=CANARY_LAYERS, families=CANARY_FAMILIES):
    """Deterministic, ordered list of the 6 I2B canary conditions: 3 layers
    x 2 families. Layers/families intentionally reuse I2_FAMILIES and a
    strict subset of I2_LAYERS (first, middle, last of the 8) rather than
    inventing a separate canary-only vocabulary."""
    for layer in layers:
        if layer not in I2_LAYERS:
            raise I2CanaryError(f"canary layer {layer} is not one of the 8 locked I2_LAYERS {I2_LAYERS}")
    return [{"layer_idx": l, "family": f} for l in layers for f in families]


def _validate_quantized_key_cache_shapes(state):
    """Structural (not merely non-None) validation of one decode step's
    real KIVI Key cache tuple. Reuses
    utils.attention_decode_features.parse_kivi_cache_tuple's field
    semantics unchanged -- never invents a new cache format.

    Per the triton_quantize_and_pack_along_last_dim contract
    (quant/new_pack.py, read but not modified): for input shape
    (B, nh, head_dim, T), key_quant_trans, key_scale_trans, and
    key_mn_trans all come back with the SAME leading (B, nh, head_dim)
    shape -- only their trailing dimension differs (packed sequence length
    for key_quant_trans vs. number of quantization groups for scale/mn).
    That leading-3-dims agreement, plus "every present tensor has
    strictly positive dimensions everywhere", is exactly what this checks.

    key_full (the FP16 residual) is deliberately NOT required merely
    because key_quant_trans is present: models/llama_kivi.py's prefill
    branch legitimately sets key_states_full = None when the prefill
    length is an exact multiple of residual_length (the whole prefill is
    quantized, no residual remains) -- requiring key_full unconditionally
    would be inventing a stricter cache format than production actually
    guarantees. When key_full IS present, its shape is still checked for
    positive dimensions like every other present tensor.
    """
    if state.key_quant_trans is None:
        return True  # nothing quantized yet at this step -- not a shape failure
    quantized_tensors = [state.key_quant_trans, state.key_scale_trans, state.key_mn_trans]
    if any(t is None for t in quantized_tensors):
        return False  # a quantized prefix without its scale/mn is malformed
    tensors_to_check = list(quantized_tensors)
    if state.key_full is not None:
        tensors_to_check.append(state.key_full)
    for t in tensors_to_check:
        if len(t.shape) == 0 or any(d <= 0 for d in t.shape):
            return False
    leading_shapes = {tuple(t.shape[:3]) for t in quantized_tensors}
    if len(leading_shapes) != 1:
        return False
    return True


def _make_finite_check_hook():
    """Returns (hook_fn, flags): flags['seen_nonfinite'] flips to True the
    first time a forward output is not fully finite (NaN/Inf). Extracted
    as a standalone function (rather than an inline closure) so its
    control logic is independently CPU-testable without a GPU or a real
    model -- see tests/test_i2_layer_family_sensitivity.py.

    Read-only by construction: `_hook` returns None, so
    register_forward_hook leaves the module's real output completely
    unchanged (nn.Module.register_forward_hook only replaces the output if
    the hook returns a non-None value). This is the exact same pattern
    already established, unmodified, in
    run_condition()/run_layer_sensitivity_pilot.py's `_lm_head_hook` --
    reused here, not reinvented. Because it never mutates output, it can
    be installed identically on both the hooks-on and hooks-off
    generate_with_capture() calls without biasing the hook-neutrality
    comparison between them."""
    import torch

    flags = {"seen_nonfinite": False}

    def _hook(module, inp, output):
        if not torch.isfinite(output).all():
            flags["seen_nonfinite"] = True

    return _hook, flags


def run_i2b_canary_condition(layer_idx, family, model_name_or_path, cache_dir):  # pragma: no cover - GPU path, exercised only by explicit --mode canary --run
    """One (layer, family) I2B canary observation. Reuses
    scripts/h2_attention_decode_parity_canary.py's load_canary_model
    (already generalized over layer_idx/family/group_size/residual_length
    and already asserts the target-layer-vs-all-others policy shape) and
    generate_with_capture directly -- never reimplemented. Deferred (heavy)
    imports happen here only.

    `generation_finite` requires BOTH that every generated token id is a
    well-formed int AND that no lm_head forward output was non-finite
    (NaN/Inf) during either generate() call -- reusing, unmodified, the
    exact same read-only forward-hook pattern already established in
    run_condition()/run_layer_sensitivity_pilot.py's `_lm_head_hook`
    (never a production-code change). The hook only reads `output`; since
    its callback returns None, register_forward_hook leaves the module's
    real output completely unchanged, so installing it identically on both
    the hooks-on and hooks-off generate() calls cannot bias the
    hook-neutrality comparison between them -- it is present symmetrically
    on both sides. Installed immediately before, removed immediately
    after, this condition's two generate() calls only."""
    import torch
    from datasets import load_dataset

    from scripts.h2_attention_decode_parity_canary import (
        build_prompt,
        generate_with_capture,
        load_canary_model,
    )
    from utils.attention_decode_features import parse_kivi_cache_tuple

    result = {
        "layer_idx": layer_idx, "family": family,
        "policy_resolved_correctly": False, "quantized_prefix_observed": False,
        "cache_shapes_valid": False, "generation_finite": False, "hook_neutral": False,
        "ran_without_error": False, "error": None,
    }
    model = None
    finite_hook_handle = None
    try:
        model, tokenizer, model_class_name = load_canary_model(
            model_name_or_path, cache_dir, I2_K_BITS, I2_V_BITS, CANARY_SEED,
            layer_idx=layer_idx, group_size=I2_GROUP_SIZE, residual_length=I2_RESIDUAL_LENGTH, family=family,
        )
        # load_canary_model already asserts target-layer/other-layers shape
        # internally (raises CanaryError on mismatch); reaching here means
        # policy resolution was correct.
        result["policy_resolved_correctly"] = True

        finite_check_hook, nonfinite_flags = _make_finite_check_hook()
        finite_hook_handle = model.lm_head.register_forward_hook(finite_check_hook)

        data = load_dataset("THUDM/LongBench", CANARY_TASK, split="test", trust_remote_code=True)
        json_obj = data[CANARY_DATASET_INDEX]
        model_short_name = model_name_or_path.split("/")[-1]
        prompt = build_prompt(tokenizer, model_short_name, json_obj, task=CANARY_TASK)

        ids_with_hooks, layer_capture, _prompt_len = generate_with_capture(
            model, tokenizer, prompt, CANARY_MAX_NEW_TOKENS, capture=True, layer_idx=layer_idx
        )
        ids_without_hooks, _, _ = generate_with_capture(
            model, tokenizer, prompt, CANARY_MAX_NEW_TOKENS, capture=False, layer_idx=layer_idx
        )
        result["generated_token_count"] = len(ids_with_hooks)
        result["hook_neutral"] = ids_with_hooks == ids_without_hooks
        result["generation_finite"] = all(isinstance(t, int) for t in ids_with_hooks) and not nonfinite_flags["seen_nonfinite"]

        quantized_prefix_observed = False
        cache_shapes_valid = True
        cache_observations = []
        for call in layer_capture.calls[1:]:  # decode calls only; calls[0] is prefill
            past_in = call.get("past_key_value_in")
            if past_in is None:
                continue
            state = parse_kivi_cache_tuple(past_in)
            if state.key_quant_trans is not None:
                quantized_prefix_observed = True
            cache_shapes_valid = cache_shapes_valid and _validate_quantized_key_cache_shapes(state)
            cache_observations.append(
                {
                    "key_quant_trans_present": state.key_quant_trans is not None,
                    "key_quant_trans_shape": None if state.key_quant_trans is None else list(state.key_quant_trans.shape),
                    "key_full_shape": None if state.key_full is None else list(state.key_full.shape),
                    "key_scale_shape": None if state.key_scale_trans is None else list(state.key_scale_trans.shape),
                    "key_mn_shape": None if state.key_mn_trans is None else list(state.key_mn_trans.shape),
                }
            )
        result["quantized_prefix_observed"] = quantized_prefix_observed
        result["cache_shapes_valid"] = cache_shapes_valid
        result["cache_observations"] = cache_observations
        result["ran_without_error"] = True
    except Exception as e:  # noqa: BLE001 -- canary must record, not crash uncaught
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        if finite_hook_handle is not None:
            finite_hook_handle.remove()
        if model is not None:
            del model
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


# I2B writes only under this dedicated subtree -- never a formal condition
# output directory (output_dir_name() always produces
# "<condition_id>_<hash12>", which can never equal or collide with
# "canary" or anything under it), and never
# outputs/i1_rotation_kivi_canary/, outputs/layer_sensitivity_pilot/,
# outputs/layer_attention_feature_pilot/, or pred/.
I2B_OUTPUT_ROOT = os.path.join(DEFAULT_OUTPUT_ROOT, "canary")


def i2b_preflight_checks():
    """Torch-free safety gate mirroring
    scripts/i1_rotation_kivi_parity_canary.py::preflight_checks exactly
    (same checks, same order, same fail-closed behavior): clean git tree,
    known HEAD, no conflicting generator process, fresh GPU preflight. Run
    BEFORE any heavy import. Raises on any failure; never kills/modifies a
    process."""
    git_status = get_git_status_short()
    if git_status is None:
        raise I2ConfigError("could not determine git status; refusing to run for provenance safety")
    if git_status.strip():
        raise I2ConfigError(f"git working tree is not clean; refusing GPU canary. git status --short:\n{git_status}")
    collection_head = get_git_commit()
    if not collection_head:
        raise I2ConfigError("could not determine git HEAD commit; refusing to run")
    conflicts = check_no_conflicting_process()
    if conflicts:
        raise I2ConflictError(f"conflicting process(es) detected, refusing to launch: {conflicts}")
    preflight = gpu_preflight()
    if preflight.get("warning"):
        raise I2ConfigError(f"GPU preflight refused launch: {preflight['warning']}")
    return collection_head, preflight


def run_i2b_canary(args):  # pragma: no cover - GPU path, exercised only by explicit --mode canary --run
    """Top-level I2B orchestration. Executes exactly the 6 canary
    conditions (i2b_canary_conditions(): layers 0/18/31 x kivi/
    rotation_kivi, K2/V16) SEQUENTIALLY -- run_i2b_canary_condition
    releases its model (del + gc.collect + torch.cuda.empty_cache, in a
    finally, regardless of success/failure) before this loop starts the
    next condition, so two 7B models are never resident simultaneously.
    Reuses run_i2b_canary_condition/compute_i2b_canary_gate unmodified;
    this function only orchestrates + writes provenance. Writes only under
    I2B_OUTPUT_ROOT (outputs/i2_layer_family_sensitivity/canary/<run_label>/),
    mirroring scripts/i1_rotation_kivi_parity_canary.py::main's provenance
    shape (per-run UTC-timestamped directory, canary.lock, host_monitor.log,
    a single JSON summary)."""
    import torch

    collection_head, preflight = i2b_preflight_checks()

    if not torch.cuda.is_available():
        raise I2CanaryError("CUDA is not available; the I2B canary requires a real GPU.")

    run_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = os.path.join(I2B_OUTPUT_ROOT, run_label)
    os.makedirs(out_dir, exist_ok=True)
    lock_fh = acquire_i2_lock(os.path.join(out_dir, "canary.lock"))
    log_path = os.path.join(out_dir, "host_monitor.log")

    def log(msg):
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line)

    try:
        boot_id_start = get_boot_id()
        gpu_before = gpu_snapshot()
        log(f"start mode=canary run_label={run_label} boot_id={boot_id_start} HEAD={collection_head}")

        summary = {
            "mode": "canary",
            "run_label": run_label,
            "collection_head": collection_head,
            "boot_id_start": boot_id_start,
            "git_status_short": get_git_status_short(),
            "cuda_device": torch.cuda.get_device_name(0),
            "triton_ptxas_path": os.environ.get("TRITON_PTXAS_PATH"),
            "gpu_before": gpu_before,
            "gpu_preflight": preflight,
            "canary_task": CANARY_TASK,
            "canary_dataset_index": CANARY_DATASET_INDEX,
            "canary_seed": CANARY_SEED,
            "group_size": I2_GROUP_SIZE,
            "residual_length": I2_RESIDUAL_LENGTH,
            "max_new_tokens": CANARY_MAX_NEW_TOKENS,
            "model_name_or_path": args.model_name_or_path,
        }

        exit_code = 0
        condition_results = []
        try:
            for cond in i2b_canary_conditions():
                log(f"condition start layer={cond['layer_idx']} family={cond['family']}")
                raw_result = run_i2b_canary_condition(cond["layer_idx"], cond["family"], args.model_name_or_path, args.cache_dir)
                gated = compute_i2b_canary_gate(raw_result)
                condition_results.append(gated)
                log(
                    f"condition end layer={cond['layer_idx']} family={cond['family']} "
                    f"condition_pass={gated['condition_pass']} "
                    f"quantized_prefix_observed={gated['quantized_prefix_observed']} "
                    f"generated_token_count={gated.get('generated_token_count')} "
                    f"error={gated.get('error')}"
                )
            summary["conditions"] = condition_results
            summary["num_conditions"] = len(condition_results)
            summary["overall_pass"] = len(condition_results) == 6 and all(c["condition_pass"] for c in condition_results)
        except Exception as e:  # noqa: BLE001 -- canary must record, not crash uncaught
            summary["error"] = f"{type(e).__name__}: {e}"
            summary["conditions"] = condition_results
            summary["overall_pass"] = False
            exit_code = 1

        boot_id_end = get_boot_id()
        summary["boot_id_end"] = boot_id_end
        summary["boot_id_stable"] = boot_id_start == boot_id_end
        summary["gpu_after"] = gpu_snapshot()
        summary["exit_code"] = exit_code

        with open(os.path.join(out_dir, "canary_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        log(f"end exit_code={exit_code} overall_pass={summary.get('overall_pass')}")
        print(json.dumps(summary, indent=2, default=str))
        print(f"\nArtifacts written to: {out_dir}")
        return exit_code
    finally:
        release_i2_lock(lock_fh)


# ---------------------------------------------------------------------------
# I2C0: formal generation execution-path infrastructure (Stage I2C0).
# Everything below is CPU-only / file-I/O-only except run_i2c_formal_generation
# itself and the dataset-identity preflight's network call -- neither is
# invoked by --dry-run or by anything else in this module.
# ---------------------------------------------------------------------------

def validate_formal_i2c_arguments(args):
    """Fail-closed formal-argument lock (Section 6 of the I2C0 task): real
    I2C execution requires EXACTLY the preregistered universe -- no
    subset/superset run may masquerade as I2C. Only the real --run formal
    path calls this; --dry-run remains flexible for exploration."""
    errors = []
    if args.model_name_or_path != "lmsys/longchat-7b-v1.5-32k":
        errors.append(f"model_name_or_path must be exactly 'lmsys/longchat-7b-v1.5-32k', got {args.model_name_or_path!r}")
    if sorted(args.layers) != sorted(I2_LAYERS):
        errors.append(f"layers must be exactly {sorted(I2_LAYERS)}, got {sorted(args.layers)}")
    if set(args.families) != set(I2_FAMILIES):
        errors.append(f"families must be exactly {sorted(I2_FAMILIES)}, got {sorted(set(args.families))}")
    if args.max_length != 31500:
        errors.append(f"max_length must be exactly 31500, got {args.max_length}")
    if args.group_size != I2_GROUP_SIZE:
        errors.append(f"group_size must be exactly {I2_GROUP_SIZE}, got {args.group_size}")
    if args.residual_length != I2_RESIDUAL_LENGTH:
        errors.append(f"residual_length must be exactly {I2_RESIDUAL_LENGTH}, got {args.residual_length}")
    if args.seed != 42:
        errors.append(f"seed must be exactly 42, got {args.seed}")
    if os.path.abspath(args.output_root) != os.path.abspath(DEFAULT_OUTPUT_ROOT):
        errors.append(f"output_root must be exactly {DEFAULT_OUTPUT_ROOT!r}, got {args.output_root!r}")
    if os.path.abspath(args.policies_dir) != os.path.abspath(DEFAULT_POLICIES_DIR):
        errors.append(f"policies_dir must be exactly {DEFAULT_POLICIES_DIR!r}, got {args.policies_dir!r}")
    if errors:
        raise I2ConfigError(
            "Formal I2C argument lock violated -- refusing a subset/superset run masquerading "
            "as I2C:\n" + "\n".join(errors)
        )


def i2c_dataset_identity_preflight(revision=I2_DATASET_REVISION, task_counts=I2_TASK_COUNTS):  # pragma: no cover - network path, exercised only by explicit --mode generation --run
    """CPU/network preflight (Section 5) -- no model, no GPU: for each of
    the 6 formal tasks, resolves THUDM/LongBench at the pinned revision and
    verifies the test split has EXACTLY the preregistered row count (these
    six tasks' sizes are already validated project-wide in
    analysis.analyze_kv_ablation.EXPECTED_TASK_COUNTS, so an exact match is
    the correct, stronger check here rather than merely '>='). Deferred
    `datasets` import -- never invoked by --dry-run. Raises I2ConfigError
    (never silently falls back to another revision) if the revision cannot
    be resolved or any task's row count doesn't match. Returns the
    per-task report list (task, requested_revision, available_row_count,
    planned_row_count)."""
    from datasets import load_dataset

    report = []
    for task, expected in task_counts.items():
        try:
            data = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True, revision=revision)
        except Exception as e:  # noqa: BLE001 -- must report, never silently fall back to another revision
            raise I2ConfigError(
                f"Could not resolve THUDM/LongBench task={task!r} at pinned revision={revision!r}: "
                f"{type(e).__name__}: {e}. Refusing to fall back to another revision -- STOP."
            ) from e
        available = len(data)
        report.append(
            {"task": task, "requested_revision": revision, "available_row_count": available, "planned_row_count": expected}
        )
        if available != expected:
            raise I2ConfigError(
                f"THUDM/LongBench task={task!r} at revision={revision!r} has {available} test rows, "
                f"expected exactly {expected} (the preregistered Stage-I2 count). Refusing to launch."
            )
    return report


def find_passing_i2b_canary_for_head(collection_head, canary_root=None):
    """Section 12: formal I2C must not rely forever on a stale canary.
    Scans every I2B canary run directory for a machine-readable
    canary_summary.json with overall_pass=True, exactly 6 conditions, every
    condition_pass=True, AND collection_head equal to the CURRENT git HEAD
    -- never "some previous commit passed". Returns the matching (most
    recent, by run-label timestamp) summary dict; raises I2ConfigError if
    none qualifies. CPU/file-I/O only."""
    canary_root = canary_root if canary_root is not None else I2B_OUTPUT_ROOT
    if not os.path.isdir(canary_root):
        raise I2ConfigError(
            f"No I2B canary runs found under {canary_root}; the I2B canary must PASS on the "
            "current HEAD before formal I2C generation is authorized."
        )
    candidates = []
    for run_label in sorted(os.listdir(canary_root)):
        summary_path = os.path.join(canary_root, run_label, "canary_summary.json")
        if not os.path.exists(summary_path):
            continue
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        candidates.append((run_label, summary))
    if not candidates:
        raise I2ConfigError(f"No readable I2B canary_summary.json found under {canary_root}.")

    matching = [
        (run_label, s)
        for run_label, s in candidates
        if s.get("collection_head") == collection_head
        and s.get("overall_pass") is True
        and len(s.get("conditions", [])) == 6
        and all(c.get("condition_pass") is True for c in s.get("conditions", []))
    ]
    if not matching:
        raise I2ConfigError(
            f"No I2B canary run under {canary_root} has overall_pass=True with all 6/6 "
            f"condition_pass=True AND collection_head == current HEAD ({collection_head}). A "
            "stale-HEAD or partial-pass canary does not authorize formal I2C generation -- rerun "
            "the I2B canary on this HEAD first."
        )
    matching.sort(key=lambda item: item[0])  # run_label is a sortable UTC timestamp string
    return matching[-1][1]


def build_initial_i2c_manifest(conditions, args, collection_head, preflight, dataset_report, task_counts=I2_TASK_COUNTS):
    """Top-level formal manifest (Section 13). Written once (via
    write_i2_manifest_atomic) before the first condition starts; updated
    in place (record_i2c_condition_result) as conditions progress."""
    now = datetime.now(timezone.utc).astimezone().isoformat()
    return {
        "experiment": "i2c_formal_layer_family_generation",
        "collection_head": collection_head,
        "dataset_revision": I2_DATASET_REVISION,
        "dataset_identity_preflight": dataset_report,
        "model_name_or_path": args.model_name_or_path,
        "seed": args.seed,
        "max_length": args.max_length,
        "group_size": args.group_size,
        "residual_length": args.residual_length,
        "tasks": list(task_counts),
        "expected_task_counts": dict(task_counts),
        "total_planned_rows": total_example_count(task_counts, len(conditions)),
        "num_conditions": len(conditions),
        "boot_id_start": get_boot_id(),
        "gpu_preflight": preflight,
        "triton_ptxas_path": os.environ.get("TRITON_PTXAS_PATH"),
        "start_time": now,
        "last_update_time": now,
        "conditions": [
            {
                "condition_id": c["condition_id"],
                "layer_idx": c["layer_idx"],
                "family": c["family"],
                "k_bits": c["k_bits"],
                "v_bits": c["v_bits"],
                "policy_path": c["policy_path"],
                "policy_hash": c["resolved_policy_hash"],
                "output_dir": c["output_dir"],
                "tasks": list(task_counts),
                "expected_task_counts": dict(task_counts),
                "status": "pending",
                "task_counts_actual": None,
                "exit_code": None,
                "nonfinite_logits_seen": None,
            }
            for c in conditions
        ],
    }


def record_i2c_condition_result(manifest, condition_id_, status, task_counts_actual=None, exit_code=None, nonfinite_logits_seen=None):
    """Updates one condition's entry in-place (status in {pending, running,
    complete, failed} -- Section 13) and bumps last_update_time. Fields
    left as None by the caller are left unchanged (not overwritten with
    None) -- only explicitly-provided values are recorded."""
    for c in manifest["conditions"]:
        if c["condition_id"] == condition_id_:
            c["status"] = status
            if task_counts_actual is not None:
                c["task_counts_actual"] = task_counts_actual
            if exit_code is not None:
                c["exit_code"] = exit_code
            if nonfinite_logits_seen is not None:
                c["nonfinite_logits_seen"] = nonfinite_logits_seen
            manifest["last_update_time"] = datetime.now(timezone.utc).astimezone().isoformat()
            return manifest
    raise I2ConfigError(f"condition_id {condition_id_} not found in formal manifest")


def validate_condition_completion(condition_dir, task_counts=I2_TASK_COUNTS):
    """Section 14: never mark a condition complete merely because
    run_condition() returned. Verifies ALL SIX final task JSONLs exist,
    are valid, have the exact expected row count, and have no leftover
    .partial; plus the per-condition manifest.json has exit_code==0 and
    nonfinite_logits_seen==False. Read-only; CPU/file-I/O only. Returns
    (is_complete, report)."""
    report = {"tasks": {}, "problems": []}
    for task, expected in task_counts.items():
        out_path = os.path.join(condition_dir, f"{task}.jsonl")
        partial_path = out_path + ".partial"
        info = inspect_jsonl(out_path)
        task_ok = info.exists and info.invalid_rows == 0 and info.valid_rows == expected and not os.path.exists(partial_path)
        report["tasks"][task] = {
            "exists": info.exists,
            "valid_rows": info.valid_rows,
            "invalid_rows": info.invalid_rows,
            "expected": expected,
            "leftover_partial": os.path.exists(partial_path),
            "ok": task_ok,
        }
        if not task_ok:
            report["problems"].append(f"{task}: not complete ({report['tasks'][task]})")

    manifest_path = os.path.join(condition_dir, "manifest.json")
    condition_manifest = None
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            condition_manifest = json.load(f)

    exit_code = condition_manifest.get("exit_code") if condition_manifest else None
    nonfinite_logits_seen = condition_manifest.get("nonfinite_logits_seen") if condition_manifest else None
    if condition_manifest is None:
        report["problems"].append("condition manifest.json missing")
    if exit_code != 0:
        report["problems"].append(f"condition manifest exit_code={exit_code!r}, expected 0")
    if nonfinite_logits_seen is not False:
        report["problems"].append(f"condition manifest nonfinite_logits_seen={nonfinite_logits_seen!r}, expected False")

    report["exit_code"] = exit_code
    report["nonfinite_logits_seen"] = nonfinite_logits_seen
    report["total_rows"] = sum(t["valid_rows"] for t in report["tasks"].values())
    report["is_complete"] = not report["problems"]
    return report["is_complete"], report


def validate_formal_generation_complete(conditions, output_root, task_counts=I2_TASK_COUNTS):
    """Section 15: aggregate integrity check across all 16 conditions --
    16/16 conditions complete, 96/96 condition-task files complete,
    23,200/23,200 rows -- required before I2C_FORMAL_GENERATION = PASS.
    Read-only; CPU/file-I/O only."""
    complete_conditions = 0
    complete_task_files = 0
    total_rows = 0
    for c in conditions:
        condition_dir = os.path.join(output_root, output_dir_name(c))
        is_complete, report = validate_condition_completion(condition_dir, task_counts)
        complete_task_files += sum(1 for t in report["tasks"].values() if t["ok"])
        total_rows += report["total_rows"]
        if is_complete:
            complete_conditions += 1
    expected_task_files = len(conditions) * len(task_counts)
    expected_total_rows = total_example_count(task_counts, len(conditions))
    all_complete = (
        complete_conditions == len(conditions)
        and complete_task_files == expected_task_files
        and total_rows == expected_total_rows
    )
    return {
        "conditions_complete": complete_conditions,
        "conditions_total": len(conditions),
        "task_files_complete": complete_task_files,
        "task_files_total": expected_task_files,
        "rows_complete": total_rows,
        "rows_total": expected_total_rows,
        "all_complete": all_complete,
    }


def run_i2c_formal_generation(args):  # pragma: no cover - GPU path, exercised only by explicit --mode generation --run with the exact formal universe
    """Top-level I2C formal-generation orchestration (Sections 7/8). Runs
    ALL validation/preflight (argument lock, git/process/GPU safety,
    dataset-identity, current-HEAD I2B prerequisite) before acquiring the
    top-level formal lock and starting any model load. Executes exactly
    the 16 preregistered conditions in the locked layer-then-family order
    (utils.i2_layer_family_conditions.all_i2_specs' canonical nesting, via
    discover_and_validate_i2_policies) SEQUENTIALLY via the existing
    run_condition(), one model at a time -- never parallelized. Stops
    immediately (does not attempt later conditions) the moment any
    condition fails to complete. Never invokes
    analysis/analyze_i2_layer_family_sensitivity.py -- I2D is a separate,
    later, explicitly-authorized review stage (Section 16)."""
    import torch

    validate_formal_i2c_arguments(args)
    collection_head, preflight = i2b_preflight_checks()

    if not torch.cuda.is_available():
        raise I2ConfigError("CUDA is not available; formal I2C generation requires a real GPU.")

    dataset_report = i2c_dataset_identity_preflight()
    find_passing_i2b_canary_for_head(collection_head)  # raises I2ConfigError if no qualifying I2B run exists

    write_i2_policies(args.policies_dir)
    conditions = discover_and_validate_i2_policies(args.policies_dir, list(I2_LAYERS), list(I2_FAMILIES))
    if len(conditions) != 16:
        raise I2ConfigError(f"Expected exactly 16 formal I2C conditions, got {len(conditions)}.")
    for c in conditions:
        c["output_dir"] = os.path.join(args.output_root, output_dir_name(c))
    output_dirs = [c["output_dir"] for c in conditions]
    if len(set(output_dirs)) != len(output_dirs):
        raise I2ConfigError("Output directory collision detected among I2C formal conditions.")
    hashes = [c["resolved_policy_hash"] for c in conditions]
    if len(set(hashes)) != len(hashes):
        raise I2ConfigError("Policy hash collision detected among I2C formal conditions.")

    os.makedirs(args.output_root, exist_ok=True)
    lock_fh = acquire_i2_lock(os.path.join(args.output_root, I2C_FORMAL_LOCK_NAME))
    manifest_path = os.path.join(args.output_root, I2C_FORMAL_MANIFEST_NAME)

    try:
        if os.path.exists(manifest_path):
            manifest = read_i2_manifest(manifest_path)
            if manifest.get("dataset_revision") != I2_DATASET_REVISION:
                raise I2ConfigError(
                    f"Existing formal manifest at {manifest_path} has dataset_revision="
                    f"{manifest.get('dataset_revision')!r}, expected {I2_DATASET_REVISION!r}. Refusing "
                    "to resume a formal run under a different dataset revision."
                )
        else:
            manifest = build_initial_i2c_manifest(conditions, args, collection_head, preflight, dataset_report)
            write_i2_manifest_atomic(manifest_path, manifest)

        dataset2prompt = json.load(open(os.path.join(REPO_ROOT, "config/dataset2prompt.json"), "r"))
        dataset2maxlen = json.load(open(os.path.join(REPO_ROOT, "config/dataset2maxlen.json"), "r"))

        for c in conditions:
            manifest = read_i2_manifest(manifest_path)
            record_i2c_condition_result(manifest, c["condition_id"], status="running")
            write_i2_manifest_atomic(manifest_path, manifest)

            exit_code = run_condition(
                c, args, dataset2prompt, dataset2maxlen, task_counts=I2_TASK_COUNTS, dataset_revision=I2_DATASET_REVISION
            )

            is_complete, completion_report = validate_condition_completion(c["output_dir"], I2_TASK_COUNTS)
            task_counts_actual = {t: v["valid_rows"] for t, v in completion_report["tasks"].items()}

            manifest = read_i2_manifest(manifest_path)
            if exit_code == 0 and is_complete:
                record_i2c_condition_result(
                    manifest, c["condition_id"], status="complete", task_counts_actual=task_counts_actual,
                    exit_code=exit_code, nonfinite_logits_seen=completion_report["nonfinite_logits_seen"],
                )
                write_i2_manifest_atomic(manifest_path, manifest)
            else:
                record_i2c_condition_result(
                    manifest, c["condition_id"], status="failed", task_counts_actual=task_counts_actual,
                    exit_code=exit_code, nonfinite_logits_seen=completion_report["nonfinite_logits_seen"],
                )
                write_i2_manifest_atomic(manifest_path, manifest)
                raise I2ConfigError(
                    f"Condition {c['condition_id']} did not complete successfully "
                    f"(exit_code={exit_code}, problems={completion_report['problems']}); stopping the "
                    "formal run -- later conditions are not attempted."
                )

        aggregate = validate_formal_generation_complete(conditions, args.output_root, I2_TASK_COUNTS)
        manifest = read_i2_manifest(manifest_path)
        manifest["aggregate"] = aggregate
        manifest["last_update_time"] = datetime.now(timezone.utc).astimezone().isoformat()
        write_i2_manifest_atomic(manifest_path, manifest)
        print(json.dumps(aggregate, indent=2))
        return 0 if aggregate["all_complete"] else 1
    finally:
        release_i2_lock(lock_fh)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name_or_path", default="lmsys/longchat-7b-v1.5-32k")
    p.add_argument("--cache_dir", default="./cached_models")
    p.add_argument("--mode", choices=["generation", "canary"], default="generation")
    p.add_argument("--layers", type=int, nargs="+", default=list(I2_LAYERS))
    p.add_argument("--families", nargs="+", default=list(I2_FAMILIES), choices=list(I2_FAMILIES))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run", action="store_true", help="Required to actually execute GPU generation or the I2B canary. Not passed in Stage I2A.")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--max-length", type=int, default=31500)
    p.add_argument("--group-size", type=int, default=I2_GROUP_SIZE)
    p.add_argument("--residual-length", type=int, default=I2_RESIDUAL_LENGTH)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--policies-dir", default=DEFAULT_POLICIES_DIR)
    return p.parse_args(argv)


def _print_generation_dry_run(conditions, task_counts, output_dirs, hashes, args):
    print(f"I2A conditions: {len(conditions)} (deterministic order)")
    for c in conditions:
        print(f"  {c['condition_id']:24s} layer={c['layer_idx']:2d} family={c['family']:14s} "
              f"K{c['k_bits']}/V{c['v_bits']:<3d} hash={c['resolved_policy_hash']} -> {c['output_dir']}")
    n_tasks = len(task_counts)
    n_examples = total_example_count(task_counts, len(conditions))
    print(f"\nSelected tasks: {list(task_counts)}")
    print(f"Dataset evaluations: {len(conditions)} conditions x {n_tasks} tasks = {len(conditions) * n_tasks}")
    print(f"Total examples: {len(conditions)} x {sum(task_counts.values())} = {n_examples}")
    print(f"\nOutput directory collisions: {'NONE' if len(set(output_dirs)) == len(output_dirs) else 'FOUND'}")
    print(f"Policy hash collisions: {'NONE' if len(set(hashes)) == len(hashes) else 'FOUND'}")
    resume_report = compute_resume_report(conditions, task_counts, args.output_root)
    non_start = [r for r in resume_report if r["action"] != "start"]
    print(f"\nResume inspection against {args.output_root} ({len(resume_report)} condition-task pairs checked):")
    if non_start:
        for r in non_start:
            extra = f" ({r['done']}/{r['expected']})" if r["done"] is not None else f" -- {r.get('error')}"
            print(f"  {r['condition_id']}/{r['task']}: {r['action']}{extra}")
    else:
        print("  all pending (action=start) -- no existing output found under this root")
    conflicts = check_no_conflicting_process()
    print(f"\nConflicting-process check ({', '.join(CONFLICTING_PROCESS_PATTERNS)}): "
          f"{'NONE FOUND' if not conflicts else conflicts}")
    print(f"\nFormal dataset revision (pinned, I2C only): {I2_DATASET_REVISION}")
    print("\nNo GPU process will be launched (--dry-run); torch/transformers/datasets were not imported.")


def _print_canary_dry_run():
    conditions = i2b_canary_conditions()
    print(f"I2B canary conditions: {len(conditions)} (3 layers x 2 families)")
    for c in conditions:
        print(f"  layer={c['layer_idx']:2d} family={c['family']:14s} K{I2_K_BITS}/V{I2_V_BITS} "
              f"task={CANARY_TASK} dataset_index={CANARY_DATASET_INDEX} max_new_tokens={CANARY_MAX_NEW_TOKENS}")
    print("\nThis canary is NOT used for efficacy conclusions -- runtime/cache-shape route validity only.")
    print("Gate (per condition, compute_i2b_canary_gate): policy_resolved_correctly, "
          "quantized_prefix_observed, cache_shapes_valid, generation_finite, hook_neutral, "
          "ran_without_error -- all must be True for condition_pass.")
    print("\nNo GPU process will be launched (--dry-run); torch/transformers/datasets were not imported.")


def main():
    args = parse_args()

    if args.mode == "canary":
        if args.dry_run or not args.run:
            _print_canary_dry_run()
            return 0
        # Stage I2B: the real GPU route-validity canary is now authorized
        # and wired. This is the ONLY path in this module that touches the
        # GPU -- --mode generation --run (I2C formal generation) below is
        # unaffected and still refuses unconditionally.
        return run_i2b_canary(args)

    write_i2_policies(args.policies_dir)
    conditions = discover_and_validate_i2_policies(args.policies_dir, args.layers, args.families)
    for c in conditions:
        c["output_dir"] = os.path.join(args.output_root, output_dir_name(c))

    output_dirs = [c["output_dir"] for c in conditions]
    if len(set(output_dirs)) != len(output_dirs):
        raise I2ConfigError("Output directory collision detected among I2A conditions.")
    hashes = [c["resolved_policy_hash"] for c in conditions]
    if len(set(hashes)) != len(hashes):
        raise I2ConfigError("Policy hash collision detected among I2A conditions.")

    if args.dry_run or not args.run:
        _print_generation_dry_run(conditions, I2_TASK_COUNTS, output_dirs, hashes, args)
        return 0

    # Stage I2C0: the real formal generation orchestration is now wired.
    # validate_formal_i2c_arguments (called first, inside
    # run_i2c_formal_generation) fails closed unless the exact
    # preregistered universe was requested -- a subset/superset run can
    # never masquerade as formal I2C.
    return run_i2c_formal_generation(args)


if __name__ == "__main__":
    sys.exit(main())
