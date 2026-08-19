"""Stage D0: infrastructure for the layer-wise KV sensitivity pilot.

16 conditions (8 probed layers x 2 axes), 4 screening LongBench tasks at
full test-split size, sequential single-instance execution. This module
builds the full pilot driver -- policy generation/validation, resume-safe
per-task generation, single-instance locking, crash-safe manifest, host
monitoring -- but running it for real (GPU generation) is a separate,
explicit later step; --dry-run exercises everything except that.

Never touches pred/ or the formal resume system (pred_long_bench.py). Writes
exclusively under --output-root (default outputs/layer_sensitivity_pilot/).

Usage (dry run, safe, CPU-only, no model import):
    ./.venv/bin/python scripts/run_layer_sensitivity_pilot.py --dry-run

Usage (real run -- NOT executed in this round, see docs/... for the exact
command and go/no-go writeup):
    CUDA_VISIBLE_DEVICES=0 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \\
    ./.venv/bin/python scripts/run_layer_sensitivity_pilot.py
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

from utils.jsonl_integrity import inspect_jsonl  # noqa: E402
from utils.pilot_policy import (  # noqa: E402
    PILOT_AXES,
    PILOT_LAYERS,
    PILOT_TASK_COUNTS,
    discover_and_validate_pilot_policies,
    output_dir_name,
    select_task_counts,
    total_example_count,
    write_pilot_policies,
)

DEFAULT_POLICIES_DIR = os.path.join(REPO_ROOT, "analysis", "policies", "layer_sensitivity_pilot")
DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "outputs", "layer_sensitivity_pilot")
DEFAULT_FP16_BASELINE_DIR = os.path.join(
    REPO_ROOT, "pred", "longchat-7b-v1.5-32k_31500_16bits_group32_residual128"
)

# Any of these process name patterns running (other than this invocation
# itself) means a conflicting generation writer might already be active.
CONFLICTING_PROCESS_PATTERNS = [
    "pred_long_bench.py",
    "layer_policy_smoke.py",
    "run_layer_sensitivity_pilot.py",
]

PILOT_CONDITION_CORE_KEYS = [
    "model_name_or_path", "max_length", "group_size", "residual_length", "seed", "policy_hash",
]


class PilotLockError(RuntimeError):
    pass


class PilotConflictError(RuntimeError):
    pass


class PilotResumeError(RuntimeError):
    pass


class PilotConfigError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Single-instance safety (section 4)
# ---------------------------------------------------------------------------

def acquire_pilot_lock(lock_path):
    """Real single-instance mechanism via flock(LOCK_EX | LOCK_NB). Returns
    an open file handle that must be kept alive for the lock's duration (do
    not let it get garbage-collected). A second concurrent invocation
    targeting the same lock path fails immediately with PilotLockError,
    reporting the PID recorded by the current holder if available."""
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
        raise PilotLockError(
            f"Pilot lock {lock_path} is already held (recorded holder: {holder}). "
            "Refusing to start a second pilot writer against the same output root."
        )
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
    """Scans running processes for a *genuine* python invocation of one of
    the given scripts -- e.g. ".../python scripts/run_layer_sensitivity_pilot.py".

    `pgrep -af <pattern>` alone is not enough: it matches the full command
    line text of *any* process, including this tool session's own bash-tool
    wrapper processes (`bash -c "... eval '... run_layer_sensitivity_pilot.py
    --dry-run' ..."`), which merely *mention* the script name inside a
    quoted argument without ever actually running it, and stale leftover
    `pgrep -f "layer_policy_smoke.py..."` polling-loop wrapper shells from
    earlier turns in this same session. Both are real false positives
    observed while building this check. To avoid them, a match only counts
    if the command line looks like an actual interpreter invocation of the
    script (a `python`-like token immediately followed by a path ending in
    the script name) and is not itself a `bash -c` wrapper shell.

    Returns the list of conflicting process lines found (empty if none).
    Never raises on its own -- callers decide whether an empty/non-empty
    result blocks execution, so --dry-run can report this check's *result*
    without treating "found nothing" as a hard requirement of dry-run itself.
    """
    self_pid = self_pid if self_pid is not None else os.getpid()
    pattern = "|".join(re.escape(p) for p in patterns)
    try:
        out = subprocess.check_output(["pgrep", "-af", pattern], stderr=subprocess.DEVNULL).decode()
    except subprocess.CalledProcessError:
        return []  # pgrep exit 1 == no matches
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
            continue  # this session's own tool wrapper shells / stale pgrep-loop artifacts
        if not genuine_invocation_re.search(line):
            continue
        conflicts.append(line)
    return conflicts


# ---------------------------------------------------------------------------
# Resume safety (section 6) -- stricter than the formal runner's
# resolve_dataset_resume_plan(): a final file must have *exactly* the
# expected row count, never >=. This intentionally does not reuse or modify
# pred_long_bench.resolve_dataset_resume_plan.
# ---------------------------------------------------------------------------

def resolve_pilot_task_resume_plan(task, out_path, partial_path, expected):
    """Returns (action, active_path, done):
      action in {"skip", "finalize", "resume", "start"}.
      "finalize": the partial already has exactly `expected` valid rows
        (e.g. the process crashed between the last write and the atomic
        rename) -- ready to os.replace() with no further generation.
      "skip": the final file already has exactly `expected` valid rows.

    Fails closed (PilotResumeError) on: any invalid JSON row, a final file
    with valid_rows != expected (over OR under count -- this is the
    explicit fix for the old formal-run bug where valid_rows >= expected
    was silently treated as complete), or a partial with more rows than
    expected.
    """
    if os.path.exists(out_path):
        info = inspect_jsonl(out_path)
        if info.invalid_rows:
            raise PilotResumeError(
                f"{task}: final file {out_path} has {info.invalid_rows} invalid JSON row(s), "
                f"first at line {info.first_invalid_line}. Refusing to skip a corrupted final file."
            )
        if info.valid_rows == expected:
            return "skip", out_path, info.valid_rows
        raise PilotResumeError(
            f"{task}: final file {out_path} has {info.valid_rows} valid rows, expected exactly "
            f"{expected} (an over-count final file is exactly as unsafe as an under-count one -- "
            "both indicate an inconsistent state that must be inspected manually, never silently "
            "accepted)."
        )

    info = inspect_jsonl(partial_path)
    if info.invalid_rows:
        raise PilotResumeError(
            f"{task}: partial file {partial_path} has {info.invalid_rows} invalid JSON line(s), "
            f"first invalid line {info.first_invalid_line}, valid prefix {info.valid_prefix_rows}. "
            "Resume start_idx cannot be safely determined from a corrupted partial."
        )
    if info.valid_rows > expected:
        raise PilotResumeError(
            f"{task}: partial file {partial_path} has {info.valid_rows} valid rows, more than "
            f"expected {expected}. Refusing to resume from an over-count partial."
        )
    if info.valid_rows == expected:
        return "finalize", partial_path, info.valid_rows
    if info.valid_rows > 0:
        return "resume", partial_path, info.valid_rows
    return "start", partial_path, 0


def compute_resume_report(conditions, task_counts, output_root):
    """For every (condition, task) pair, inspects any existing output under
    output_root and reports the resume decision resolve_pilot_task_resume_plan
    would make -- without generating anything. Used by --dry-run to prove a
    future full invocation would skip already-complete work (and by the
    canary's own post-hoc verification). A PilotResumeError for one
    (condition, task) is captured as an "ERROR" entry rather than raised, so
    one corrupt condition doesn't stop the report for the other 15."""
    report = []
    for c in conditions:
        condition_dir = os.path.join(output_root, output_dir_name(c))
        for task, expected in task_counts.items():
            out_path = os.path.join(condition_dir, f"{task}.jsonl")
            partial_path = out_path + ".partial"
            try:
                action, _active_path, done = resolve_pilot_task_resume_plan(task, out_path, partial_path, expected)
                report.append({"condition_id": c["condition_id"], "task": task, "action": action, "done": done, "expected": expected})
            except PilotResumeError as e:
                report.append({"condition_id": c["condition_id"], "task": task, "action": "ERROR", "done": None, "expected": expected, "error": str(e)})
    return report


def finalize_task_if_ready(action, active_path, out_path, expected):
    """Atomically renames a fully-written partial to its final name once
    resolve_pilot_task_resume_plan reports "finalize" or once a fresh
    generation loop just completed the last row. Re-validates exact count
    immediately before the rename (never trust a stale decision)."""
    info = inspect_jsonl(active_path)
    if info.invalid_rows or info.valid_rows != expected:
        raise PilotResumeError(
            f"Refusing to finalize {active_path}: valid_rows={info.valid_rows} "
            f"invalid_rows={info.invalid_rows}, expected exactly {expected} valid rows."
        )
    os.replace(active_path, out_path)


# ---------------------------------------------------------------------------
# Per-condition directory / config (section 2, 6)
# ---------------------------------------------------------------------------

def build_condition_run_config(condition, model_name_or_path, max_length, group_size, residual_length, seed, task_counts=PILOT_TASK_COUNTS):
    return {
        "model_name_or_path": model_name_or_path,
        "condition_id": condition["condition_id"],
        "layer_idx": condition["layer_idx"],
        "axis": condition["axis"],
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
        "pilot": "layer_sensitivity_pilot_stage_d0",
    }


def prepare_condition_directory(condition_dir, run_config):
    """Pilot-specific analogue of pred_long_bench.prepare_run_directory:
    creates the directory + writes run_config.json on first use; on a
    resumed run, fails closed if any PILOT_CONDITION_CORE_KEYS entry
    (crucially including policy_hash) doesn't match exactly -- two
    different policies must never be considered resume-compatible."""
    run_config_path = os.path.join(condition_dir, "run_config.json")
    if not os.path.exists(condition_dir):
        os.makedirs(condition_dir)
        with open(run_config_path, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)
        return "created"

    if not os.path.exists(run_config_path):
        raise PilotConfigError(
            f"{condition_dir} exists but has no run_config.json -- unverified state, refusing to reuse."
        )
    with open(run_config_path, "r", encoding="utf-8") as f:
        existing = json.load(f)
    mismatches = {
        key: (existing.get(key), run_config.get(key))
        for key in PILOT_CONDITION_CORE_KEYS
        if existing.get(key) != run_config.get(key)
    }
    if mismatches:
        raise PilotConfigError(
            f"Refusing to resume {condition_dir}: run_config.json does not match the current "
            f"condition. Mismatched keys (existing, requested): {mismatches}"
        )
    return "validated"


# ---------------------------------------------------------------------------
# Pilot-wide manifest (section 9) -- crash-safe via write-temp + os.replace.
# ---------------------------------------------------------------------------

def build_initial_pilot_manifest(conditions, args, task_counts=PILOT_TASK_COUNTS):
    return {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "pilot": "layer_sensitivity_pilot_stage_d0",
        "num_conditions": len(conditions),
        "tasks": list(task_counts),
        "expected_task_counts": dict(task_counts),
        "total_examples": total_example_count(task_counts, len(conditions)),
        "conditions": [
            {
                "condition_id": c["condition_id"],
                "layer_idx": c["layer_idx"],
                "axis": c["axis"],
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


def write_pilot_manifest_atomic(path, manifest):
    path = str(path)
    tmp_path = f"{path}.tmp{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def read_pilot_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_condition_status(manifest, condition_id_, status):
    for c in manifest["conditions"]:
        if c["condition_id"] == condition_id_:
            c["status"] = status
            return manifest
    raise PilotConfigError(f"condition_id {condition_id_} not found in manifest")


# ---------------------------------------------------------------------------
# Baseline reuse audit (section 7) -- read-only, no GPU, no torch import.
# ---------------------------------------------------------------------------

def audit_fp16_baseline_reuse(baseline_dir=DEFAULT_FP16_BASELINE_DIR, task_counts=PILOT_TASK_COUNTS):
    """Checks whether the existing formal K16/V16 predictions can serve as
    the FP16 reference for the 4 screening tasks: presence, exact expected
    row count, zero invalid rows, no leftover .partial, trailing newline.
    Does NOT independently verify k_bits/v_bits metadata (this legacy
    directory predates run_config.json -- see models/llama_kivi.py /
    pred_long_bench.py's discovery notes) and does NOT by itself prove the
    plain-HF LlamaForCausalLM generation path is numerically/behaviorally
    identical to LlamaForCausalLM_KIVI's all-16/16 pass-through route used
    by every pilot condition's non-probed layers -- see the caller's report
    for that separate, partial (hotpotqa-only) cross-check from Stage C.
    """
    result = {"baseline_dir": baseline_dir, "run_config_present": os.path.exists(os.path.join(baseline_dir, "run_config.json")), "tasks": {}}
    all_ok = True
    for task, expected in task_counts.items():
        path = os.path.join(baseline_dir, f"{task}.jsonl")
        partial = path + ".partial"
        info = inspect_jsonl(path)
        ok = info.exists and info.invalid_rows == 0 and info.valid_rows == expected and info.ends_with_newline and not os.path.exists(partial)
        all_ok = all_ok and ok
        result["tasks"][task] = {
            "exists": info.exists,
            "valid_rows": info.valid_rows,
            "invalid_rows": info.invalid_rows,
            "expected": expected,
            "ends_with_newline": info.ends_with_newline,
            "leftover_partial": os.path.exists(partial),
            "ok": ok,
        }
    result["reusable"] = all_ok
    return result


# ---------------------------------------------------------------------------
# Host monitoring (section 8) -- design/implementation present, only invoked
# during a real condition run (never during --dry-run).
# ---------------------------------------------------------------------------

def sample_host_metrics():
    """One GPU sample line, same fields as the Phase-1 host_monitor.log
    convention (timestamp, uptime, temperature.gpu, utilization.gpu,
    power.draw)."""
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


# ---------------------------------------------------------------------------
# Real per-condition generation (NOT invoked by --dry-run; not executed in
# this round -- see the final report for the exact future launch command).
# ---------------------------------------------------------------------------

def run_condition(condition, args, dataset2prompt, dataset2maxlen, task_counts=PILOT_TASK_COUNTS):  # pragma: no cover - GPU path, not exercised in Stage D0
    """Runs one pilot condition end to end: build the KIVI model with this
    condition's layer policy, generate exactly `task_counts` (a subset or
    the full 4 screening tasks, per --tasks) with durable per-row fsync +
    exact-count resume, host-monitor sampling every 30s, then finalize.
    Deferred (heavy) imports happen here only -- --dry-run never reaches
    this function."""
    import gc
    import threading

    import torch
    from datasets import load_dataset

    import pred_long_bench as plb

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

        # Smoke-only-style diagnostic (same pattern as layer_policy_smoke.py):
        # a forward hook on the real lm_head module, not a production code
        # change, recording whether any logits tensor produced during this
        # condition's generation contains NaN/Inf.
        def _lm_head_hook(module, inp, output):
            if not torch.isfinite(output).all():
                nan_inf_flags["seen_nonfinite"] = True

        hook_handle = model.lm_head.register_forward_hook(_lm_head_hook)

        for task, expected in task_counts.items():
            out_path = os.path.join(condition_dir, f"{task}.jsonl")
            partial_path = out_path + ".partial"
            action, active_path, done = resolve_pilot_task_resume_plan(task, out_path, partial_path, expected)
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
                    if task not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                        prompt = plb.build_chat(tokenizer, prompt, model_short_name)
                    inp = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
                    context_length = inp.input_ids.shape[-1]
                    with torch.no_grad():
                        output = model.generate(
                            **inp, max_new_tokens=max_gen, num_beams=1, do_sample=False, temperature=1.0, top_p=1.0,
                        )[0]
                    pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
                    pred = plb.post_process(pred, model_short_name)
                    row = {"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}
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
            "kernel_log_note": "journalctl inspection is permission-blocked in this environment; not claimed clean.",
        }
        with open(os.path.join(condition_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return exit_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name_or_path", default="lmsys/longchat-7b-v1.5-32k")
    p.add_argument("--cache_dir", default="./cached_models")
    p.add_argument("--layers", type=int, nargs="+", default=list(PILOT_LAYERS))
    p.add_argument("--axes", nargs="+", default=list(PILOT_AXES), choices=list(PILOT_AXES))
    p.add_argument("--tasks", nargs="+", default=list(PILOT_TASK_COUNTS))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help="Explicit acknowledgement that existing condition output dirs may be resumed (default behavior already resumes safely; this flag only affects messaging).")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--max-length", type=int, default=31500)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--residual-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--policies-dir", default=DEFAULT_POLICIES_DIR)
    p.add_argument("--fp16-baseline-dir", default=DEFAULT_FP16_BASELINE_DIR)
    return p.parse_args(argv)


def main():
    args = parse_args()

    task_counts = select_task_counts(args.tasks)

    write_pilot_policies(args.policies_dir)
    conditions = discover_and_validate_pilot_policies(args.policies_dir, args.layers, args.axes)
    for c in conditions:
        c["output_dir"] = os.path.join(args.output_root, output_dir_name(c))

    output_dirs = [c["output_dir"] for c in conditions]
    if len(set(output_dirs)) != len(output_dirs):
        raise PilotConfigError("Output directory collision detected among pilot conditions.")
    hashes = [c["resolved_policy_hash"] for c in conditions]
    if len(set(hashes)) != len(hashes):
        raise PilotConfigError("Policy hash collision detected among pilot conditions.")

    baseline_audit = audit_fp16_baseline_reuse(args.fp16_baseline_dir, task_counts)
    conflicts = check_no_conflicting_process()

    if args.dry_run:
        print(f"Pilot conditions: {len(conditions)} (deterministic order)")
        for c in conditions:
            print(f"  {c['condition_id']:16s} layer={c['layer_idx']:2d} axis={c['axis']:5s} "
                  f"K{c['k_bits']}/V{c['v_bits']:<3d} hash={c['resolved_policy_hash']} -> {c['output_dir']}")
        n_tasks = len(task_counts)
        n_examples = total_example_count(task_counts, len(conditions))
        print(f"\nSelected tasks: {list(task_counts)}")
        print(f"Dataset evaluations: {len(conditions)} conditions x {n_tasks} tasks = {len(conditions) * n_tasks}")
        print(f"Total examples: {len(conditions)} x {sum(task_counts.values())} = {n_examples}")
        print(f"\nOutput directory collisions: {'NONE' if len(set(output_dirs)) == len(output_dirs) else 'FOUND'}")
        print(f"Policy hash collisions: {'NONE' if len(set(hashes)) == len(hashes) else 'FOUND'}")
        print(f"\nFP16 baseline reuse audit ({args.fp16_baseline_dir}):")
        print(f"  run_config.json present: {baseline_audit['run_config_present']} (legacy dir predates it -- expected False)")
        for task, info in baseline_audit["tasks"].items():
            print(f"  {task}: {info}")
        print(f"  reusable (integrity-only check): {baseline_audit['reusable']}")
        resume_report = compute_resume_report(conditions, task_counts, args.output_root)
        non_start = [r for r in resume_report if r["action"] != "start"]
        print(f"\nResume inspection against {args.output_root} ({len(resume_report)} condition-task pairs checked):")
        if non_start:
            for r in non_start:
                extra = f" ({r['done']}/{r['expected']})" if r["done"] is not None else f" -- {r.get('error')}"
                print(f"  {r['condition_id']}/{r['task']}: {r['action']}{extra}")
        else:
            print("  all pending (action=start) -- no existing output found under this root")
        print(f"\nConflicting-process check ({', '.join(CONFLICTING_PROCESS_PATTERNS)}): "
              f"{'NONE FOUND' if not conflicts else conflicts}")
        print("\nNo GPU process will be launched (--dry-run); torch/transformers/datasets were not imported.")
        return 0

    if conflicts:
        raise PilotConflictError(f"Conflicting process(es) detected, refusing to start: {conflicts}")

    lock_path = os.path.join(args.output_root, ".pilot.lock")
    lock_fh = acquire_pilot_lock(lock_path)
    try:
        os.makedirs(args.output_root, exist_ok=True)
        manifest_path = os.path.join(args.output_root, "pilot_manifest.json")
        manifest = build_initial_pilot_manifest(conditions, args, task_counts)
        write_pilot_manifest_atomic(manifest_path, manifest)

        dataset2prompt = json.load(open(os.path.join(REPO_ROOT, "config/dataset2prompt.json"), "r"))
        dataset2maxlen = json.load(open(os.path.join(REPO_ROOT, "config/dataset2maxlen.json"), "r"))

        for c in conditions:
            manifest = read_pilot_manifest(manifest_path)
            update_condition_status(manifest, c["condition_id"], "running")
            write_pilot_manifest_atomic(manifest_path, manifest)

            exit_code = run_condition(c, args, dataset2prompt, dataset2maxlen, task_counts)

            manifest = read_pilot_manifest(manifest_path)
            update_condition_status(manifest, c["condition_id"], "complete" if exit_code == 0 else "failed")
            write_pilot_manifest_atomic(manifest_path, manifest)
            if exit_code != 0:
                raise RuntimeError(f"Condition {c['condition_id']} failed (exit_code={exit_code}); stopping sequential run.")
    finally:
        release_pilot_lock(lock_fh)

    return 0


if __name__ == "__main__":
    sys.exit(main())
