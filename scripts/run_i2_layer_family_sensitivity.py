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

# Any of these process name patterns running (other than this invocation
# itself) means a conflicting generation writer might already be active.
CONFLICTING_PROCESS_PATTERNS = [
    "pred_long_bench.py",
    "run_layer_sensitivity_pilot.py",
    "run_layer_attention_feature_pilot.py",
    "i1_rotation_kivi_parity_canary.py",
    "run_i2_layer_family_sensitivity.py",
]

I2_CONDITION_CORE_KEYS = [
    "model_name_or_path", "max_length", "group_size", "residual_length", "seed", "policy_hash",
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

def build_condition_run_config(condition, model_name_or_path, max_length, group_size, residual_length, seed, task_counts=I2_TASK_COUNTS):
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

def run_condition(condition, args, dataset2prompt, dataset2maxlen, task_counts=I2_TASK_COUNTS):  # pragma: no cover - GPU path, not exercised in Stage I2A
    """Runs one I2A condition end to end: build the KIVI model with this
    condition's one-target-layer policy (target layer = condition['family']
    at K2/V16, all other layers kivi K16/V16), generate exactly
    `task_counts` (the 6-task formal set) with durable per-row fsync +
    exact-count resume, host-monitor sampling every 30s, then finalize.
    Deferred (heavy) imports happen here only -- --dry-run never reaches
    this function. Structurally identical to
    run_layer_sensitivity_pilot.run_condition, generalized from
    (layer, axis) to (layer, family)."""
    import gc
    import threading

    import torch
    from datasets import load_dataset

    import pred_long_bench as plb
    from utils.generation_semantics import NO_BUILD_CHAT_DATASETS, resolve_generate_kwargs

    condition_dir = os.path.join(args.output_root, output_dir_name(condition))
    run_config = build_condition_run_config(
        condition, args.model_name_or_path, args.max_length, args.group_size, args.residual_length, args.seed,
        task_counts=task_counts,
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

            data = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
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


def run_i2b_canary_condition(layer_idx, family, model_name_or_path, cache_dir):  # pragma: no cover - GPU path, not exercised in Stage I2A
    """One (layer, family) I2B canary observation. Reuses
    scripts/h2_attention_decode_parity_canary.py's load_canary_model
    (already generalized over layer_idx/family/group_size/residual_length
    and already asserts the target-layer-vs-all-others policy shape) and
    generate_with_capture directly -- never reimplemented. Deferred (heavy)
    imports happen here only."""
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
    try:
        model, tokenizer, model_class_name = load_canary_model(
            model_name_or_path, cache_dir, I2_K_BITS, I2_V_BITS, CANARY_SEED,
            layer_idx=layer_idx, group_size=I2_GROUP_SIZE, residual_length=I2_RESIDUAL_LENGTH, family=family,
        )
        # load_canary_model already asserts target-layer/other-layers shape
        # internally (raises CanaryError on mismatch); reaching here means
        # policy resolution was correct.
        result["policy_resolved_correctly"] = True

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
        result["hook_neutral"] = ids_with_hooks == ids_without_hooks
        result["generation_finite"] = all(isinstance(t, int) for t in ids_with_hooks)

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
                shapes_ok = (
                    state.key_scale_trans is not None
                    and state.key_mn_trans is not None
                    and state.key_quant_trans.shape[0] == state.key_scale_trans.shape[0] == state.key_mn_trans.shape[0]
                )
                cache_shapes_valid = cache_shapes_valid and shapes_ok
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

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:  # noqa: BLE001 -- canary must record, not crash uncaught
        result["error"] = f"{type(e).__name__}: {e}"
    return result


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
        raise I2CanaryError(
            "I2B canary execution is not authorized by this task (Stage I2A is infrastructure/"
            "preregistration only). Re-run with --dry-run, or obtain explicit I2B authorization first."
        )

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

    raise I2ConfigError(
        "I2C formal generation execution is not authorized by this task (Stage I2A is "
        "infrastructure/preregistration only; I2B must pass first). Re-run with --dry-run, "
        "or obtain explicit I2C authorization first."
    )


if __name__ == "__main__":
    sys.exit(main())
