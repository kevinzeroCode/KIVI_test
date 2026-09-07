"""Stage I1-C: Rotation-KIVI GPU production-parity canary harness.

Validates (does NOT scientifically collect) the Stage I1B "rotation_kivi"
family locked in models/llama_kivi.py: a QuaRot-inspired post-RoPE
per-head Hadamard Q/K rotation applied to KIVI Key-cache quantization only
(Value stays standard KIVI). NOT full QuaRot.

Two modes:
  --mode c0  Invariance canary: family="rotation_kivi" K16/V16 vs
             family="kivi" K16/V16 (same layer/prompt). Proves the
             rotation itself preserves model computation (no quantization
             involved on either side at K16/V16).
  --mode c1  Quantized-Key canary: family="rotation_kivi" K2/V16 vs
             family="kivi" K2/V16. Proves the real quantized rotated-Key
             production path executes and measures Key distortion
             separately for each route. Generated tokens are NOT required
             to match between the two quantizers at K2.
  --mode c0diag  Stage I1-C0D numerical attribution diagnostic (read-only
             relative to c0/c1's own code): given the original --mode c0
             FAIL, determines whether the observed differences come from
             ordinary FP16 representation rounding or a genuine production
             mismatch. One rotation_kivi K16/V16 model load only; defines
             no new pass/fail gate, reports raw numbers.

All modes require an explicit --run flag; omitting it (or any git-dirty /
conflicting-process / GPU-busy condition) refuses before any GPU work.
Writes only under outputs/i1_rotation_kivi_canary/{c0,c1,c0diag}/ -- never
outputs/layer_attention_feature_pilot/ or outputs/layer_sensitivity_pilot/
(frozen/reserved Stage-H/Stage-E data), and never overwrites a prior run
(each run gets its own UTC-timestamped subdirectory).

Reuses production code directly wherever possible -- never reimplements:
  - scripts.h2_attention_decode_parity_canary: LayerCallCapture,
    tensor_diff_report, build_prompt, load_canary_model,
    generate_with_capture, get_boot_id, get_git_commit,
    get_git_status_short, gpu_snapshot, reconstruct_key_step
  - utils.hadamard: get_normalized_hadamard, apply_hadamard_rotation
  - utils.attention_decode_features: key_decode_distortion, ShadowKVCache,
    parse_kivi_cache_tuple

Usage (NOT executed by anything in this round):
    CUDA_VISIBLE_DEVICES=0 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \\
    ./.venv/bin/python scripts/i1_rotation_kivi_parity_canary.py --run --mode c0
"""
import argparse
import errno
import fcntl
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Deferred (function-local) imports of utils.hadamard / utils.attention_decode_features
# / scripts.h2_attention_decode_parity_canary keep --dry-run-equivalent /
# argument-parsing paths torch-free, matching this project's established
# discipline (utils.attention_decode_features imports torch at module level).

DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "outputs", "i1_rotation_kivi_canary")
C0_OUTPUT_ROOT = os.path.join(DEFAULT_OUTPUT_ROOT, "c0")
C1_OUTPUT_ROOT = os.path.join(DEFAULT_OUTPUT_ROOT, "c1")
# Stage I1-C0D: numerical attribution diagnostic. Separate root -- never
# overwrites a historical C0 run.
C0DIAG_OUTPUT_ROOT = os.path.join(DEFAULT_OUTPUT_ROOT, "c0diag")

DEFAULT_MODEL_NAME = "lmsys/longchat-7b-v1.5-32k"
DEFAULT_CACHE_DIR = "./cached_models"

# Locked I1-C identity (Section 6/13). Reuses H2's exact task/index/layer/
# seed/group/residual/horizon -- never re-derived.
CANARY_TASK = "lcc"
CANARY_DATASET_INDEX = 122
CANARY_LAYER = 0
CANARY_GROUP_SIZE = 32
CANARY_RESIDUAL_LENGTH = 128
CANARY_MAX_NEW_TOKENS = 6  # reuses H2's horizon: prefill + decode steps 1..5, covers 1/2/4
CANARY_SEED = 42

C0_K_BITS, C0_V_BITS = 16, 16
C1_K_BITS, C1_V_BITS = 2, 16

REQUESTED_INVARIANCE_STEPS = (1, 2, 4)

CONFLICTING_PROCESS_PATTERNS = [
    "h2_attention_decode_parity_canary.py",
    "i1_rotation_kivi_parity_canary.py",
    "run_layer_attention_feature_pilot.py",
    "run_layer_sensitivity_pilot.py",
    "pred_long_bench.py",
]


class CanaryConfigError(ValueError):
    pass


class CanaryLockError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Torch-free provenance/safety helpers (own copies, same pattern as every
# prior script in this project -- see scripts/run_layer_attention_feature_pilot.py)
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
    """Inspects ACTUAL GPU state at launch time -- never a kill/modify
    action, never assumes any prior observation still holds."""
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


def acquire_lock(lock_path):
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
        raise CanaryLockError(f"Canary lock {lock_path} is already held (recorded holder: {holder}).")
    fh.seek(0)
    fh.truncate()
    fh.write(f"pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()}\n")
    fh.flush()
    return fh


def release_lock(fh):
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def preflight_checks():
    """Torch-free safety gate, run BEFORE any heavy import. Raises on any
    failure; never kills/modifies a process."""
    git_status = get_git_status_short()
    if git_status is None:
        raise CanaryConfigError("could not determine git status; refusing to run for provenance safety")
    if git_status.strip():
        raise CanaryConfigError(f"git working tree is not clean; refusing GPU canary. git status --short:\n{git_status}")
    collection_head = get_git_commit()
    if not collection_head:
        raise CanaryConfigError("could not determine git HEAD commit; refusing to run")
    conflicts = check_no_conflicting_process()
    if conflicts:
        raise CanaryLockError(f"conflicting process(es) detected, refusing to launch: {conflicts}")
    preflight = gpu_preflight()
    if preflight.get("warning"):
        raise CanaryConfigError(f"GPU preflight refused launch: {preflight['warning']}")
    return collection_head, preflight


def _fresh_model_load(model_name_or_path, cache_dir, k_bits, v_bits, family):
    """One fresh model load for one policy (Section 6: never hold two
    7B models simultaneously). Deferred import."""
    from scripts.h2_attention_decode_parity_canary import load_canary_model

    return load_canary_model(
        model_name_or_path, cache_dir, k_bits, v_bits, CANARY_SEED,
        layer_idx=CANARY_LAYER, group_size=CANARY_GROUP_SIZE, residual_length=CANARY_RESIDUAL_LENGTH,
        family=family,
    )


def _release_model(model):
    import gc

    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _load_prompt(tokenizer, model_name_or_path):
    from datasets import load_dataset

    from scripts.h2_attention_decode_parity_canary import build_prompt

    data = load_dataset("THUDM/LongBench", CANARY_TASK, split="test", trust_remote_code=True)
    json_obj = data[CANARY_DATASET_INDEX]
    model_short_name = model_name_or_path.split("/")[-1]
    return build_prompt(tokenizer, model_short_name, json_obj, task=CANARY_TASK)


def _generate_and_capture(model, tokenizer, prompt):
    """One fully-instrumented generation: hooks-on run + hooks-off
    hook-neutrality companion run, plus reshaped V attached to every call."""
    from scripts.h2_attention_decode_parity_canary import generate_with_capture

    ids_with_hooks, capture, prompt_len = generate_with_capture(
        model, tokenizer, prompt, CANARY_MAX_NEW_TOKENS, capture=True, layer_idx=CANARY_LAYER
    )
    ids_without_hooks, _, _ = generate_with_capture(
        model, tokenizer, prompt, CANARY_MAX_NEW_TOKENS, capture=False, layer_idx=CANARY_LAYER
    )
    hook_neutral = ids_with_hooks == ids_without_hooks

    attn = model.model.layers[CANARY_LAYER].self_attn
    head_dim = attn.head_dim
    num_kv_heads = attn.num_key_value_heads
    calls = capture.calls
    for call in calls:
        raw_v = call["v_proj_raw"]
        q_len = raw_v.shape[1]
        call["v_current"] = raw_v.view(raw_v.shape[0], q_len, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    return {
        "ids_with_hooks": ids_with_hooks,
        "hook_neutral": hook_neutral,
        "calls": calls,
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
        "prompt_len": prompt_len,
    }


# ---------------------------------------------------------------------------
# C0: invariance canary (K16/V16, no quantization on either route)
# ---------------------------------------------------------------------------

def compute_c0_structural_gate(
    cache_checks, decode_invariance_records, production_comparison_records,
    generated_prefix_equal, hook_neutral_reference, hook_neutral_rotation, finite_checks,
):
    """Pure (no torch/GPU/model dependency beyond whatever already-computed
    tensor_diff_report-shaped dicts and plain booleans the caller supplies)
    combination of the Section 10 structural correctness gate. Deliberately
    factored out of run_c0() so it is CPU-testable with synthetic inputs --
    the actual production tensors still require a real GPU run to produce.

    Per Section 10: torch.equal is NEVER required for the rotation-vs-raw
    or rotation-vs-reference FP16 comparisons (an extra Hadamard matmul
    introduces ordinary rounding) -- only allclose_1e-3/shape/finite checks
    gate the result. generated_prefix_equal and both families' hook
    neutrality ARE required. This function never invents a new scientific
    efficacy threshold; it only combines already-locked structural checks.
    """
    gate = {
        "cache_representation_correct": bool(
            cache_checks["rotation_key_quant_trans_is_none"]
            and cache_checks["reference_key_quant_trans_is_none"]
            and cache_checks["rotation_key_full_vs_K_expected_rot"]["allclose_1e-3"]
            and cache_checks["reference_key_full_vs_raw_post_rope"]["torch_equal"]
        ),
        "qk_invariance_within_tolerance": all(r["allclose_1e-3"] for r in decode_invariance_records),
        "finite_tensors": all(finite_checks),
        "matching_shapes": all(
            r["pre_softmax_logits"]["shape_equal"] and r["attn_output_pre_o_proj"]["shape_equal"] for r in production_comparison_records
        ),
        "no_unexpected_cache_representation": bool(
            cache_checks["rotation_key_quant_trans_is_none"] and cache_checks["reference_key_quant_trans_is_none"]
        ),
        "generated_prefix_equal": bool(generated_prefix_equal),
        "hook_neutral_reference": bool(hook_neutral_reference),
        "hook_neutral_rotation": bool(hook_neutral_rotation),
    }
    gate["overall_pass"] = all(bool(v) for v in gate.values())
    return gate


def run_c0(model_name_or_path=DEFAULT_MODEL_NAME, cache_dir=DEFAULT_CACHE_DIR):
    import torch

    from scripts.h2_attention_decode_parity_canary import tensor_diff_report
    from utils.attention_decode_features import parse_kivi_cache_tuple
    from utils.hadamard import apply_hadamard_rotation, get_normalized_hadamard

    # --- REFERENCE: family="kivi" K16/V16 ---
    ref_model, ref_tokenizer, ref_model_class = _fresh_model_load(model_name_or_path, cache_dir, C0_K_BITS, C0_V_BITS, "kivi")
    prompt = _load_prompt(ref_tokenizer, model_name_or_path)
    ref = _generate_and_capture(ref_model, ref_tokenizer, prompt)
    _release_model(ref_model)

    # --- ROTATION: family="rotation_kivi" K16/V16 (sequential, never simultaneous) ---
    rot_model, rot_tokenizer, rot_model_class = _fresh_model_load(model_name_or_path, cache_dir, C0_K_BITS, C0_V_BITS, "rotation_kivi")
    rot = _generate_and_capture(rot_model, rot_tokenizer, prompt)

    ref_calls, rot_calls = ref["calls"], rot["calls"]
    if len(ref_calls) < 3 or len(rot_calls) < 3:
        _release_model(rot_model)
        raise RuntimeError(f"expected >=3 layer-0 calls per route, got reference={len(ref_calls)} rotation={len(rot_calls)}")

    head_dim = rot["head_dim"]

    # --- Section 7: prefill parity (production path, not simulation) ---
    prefill_check = tensor_diff_report(rot_calls[0]["attn_output_pre_o_proj"], ref_calls[0]["attn_output_pre_o_proj"], "prefill_attn_output_pre_o_proj")

    # --- Section 8: cache representation ---
    H = get_normalized_hadamard(head_dim, rot_calls[0]["k_post_rope"].device, rot_calls[0]["k_post_rope"].dtype)
    K_expected_rot = apply_hadamard_rotation(rot_calls[0]["k_post_rope"], H)
    rot_past = parse_kivi_cache_tuple(rot_calls[1]["past_key_value_in"])
    ref_past = parse_kivi_cache_tuple(ref_calls[1]["past_key_value_in"])
    cache_checks = {
        "rotation_key_quant_trans_is_none": rot_past.key_quant_trans is None,
        "reference_key_quant_trans_is_none": ref_past.key_quant_trans is None,
        "rotation_key_full_vs_K_expected_rot": tensor_diff_report(rot_past.key_full, K_expected_rot, "rotation_key_full_vs_expected"),
        "reference_key_full_vs_raw_post_rope": tensor_diff_report(ref_past.key_full, ref_calls[0]["k_post_rope"], "reference_key_full_vs_raw"),
    }

    # --- Section 12: Value untouched (representation-only claim) ---
    value_check = tensor_diff_report(rot_past.value_full, ref_past.value_full, "value_full_reference_vs_rotation")

    # --- Section 9: real decode rotation invariance (mathematical, pre-quantization) ---
    rot_decode_calls = rot_calls[1:]
    ref_decode_calls = ref_calls[1:]
    n_common = min(len(rot_decode_calls), len(ref_decode_calls))
    decode_invariance_records = []
    production_comparison_records = []
    for step_idx in range(1, n_common + 1):
        rcall = rot_decode_calls[step_idx - 1]
        Q_raw = rcall["q_post_rope"]
        K_raw = rcall["k_post_rope"]
        Hd = get_normalized_hadamard(head_dim, Q_raw.device, Q_raw.dtype)
        Q_rot_indep = apply_hadamard_rotation(Q_raw, Hd)
        K_rot_indep = apply_hadamard_rotation(K_raw, Hd)
        raw_logits = torch.matmul(Q_raw, K_raw.transpose(2, 3))
        rot_logits_indep = torch.matmul(Q_rot_indep, K_rot_indep.transpose(2, 3))
        decode_invariance_records.append(
            {
                "step": step_idx,
                "is_preferred_step": step_idx in REQUESTED_INVARIANCE_STEPS,
                **tensor_diff_report(rot_logits_indep, raw_logits, f"QK_invariance_step{step_idx}"),
            }
        )

        # --- Section 10: production logit/output comparison (diagnostic tolerances) ---
        fcall = ref_decode_calls[step_idx - 1]
        production_comparison_records.append(
            {
                "step": step_idx,
                "pre_softmax_logits": tensor_diff_report(rcall.get("pre_softmax_logits"), fcall.get("pre_softmax_logits"), f"pre_softmax_logits_step{step_idx}"),
                "attn_output_pre_o_proj": tensor_diff_report(rcall.get("attn_output_pre_o_proj"), fcall.get("attn_output_pre_o_proj"), f"attn_output_pre_o_proj_step{step_idx}"),
            }
        )

    _release_model(rot_model)

    # --- Section 11: generated-prefix equality + per-family hook neutrality ---
    generated_prefix_equal = ref["ids_with_hooks"] == rot["ids_with_hooks"]

    def _finite(t):
        return bool(torch.isfinite(t.float()).all().item()) if t is not None else False

    finite_checks = [
        _finite(rot_calls[0]["attn_output_pre_o_proj"]),
        _finite(ref_calls[0]["attn_output_pre_o_proj"]),
        _finite(rot_past.key_full),
        _finite(ref_past.key_full),
    ] + [_finite(r.get("attn_output_pre_o_proj")) for r in rot_decode_calls[:n_common]] + [_finite(r.get("attn_output_pre_o_proj")) for r in ref_decode_calls[:n_common]]

    structural_gate = compute_c0_structural_gate(
        cache_checks=cache_checks,
        decode_invariance_records=decode_invariance_records,
        production_comparison_records=production_comparison_records,
        generated_prefix_equal=generated_prefix_equal,
        hook_neutral_reference=ref["hook_neutral"],
        hook_neutral_rotation=rot["hook_neutral"],
        finite_checks=finite_checks,
    )

    return {
        "mode": "c0",
        "reference_policy": f"K{C0_K_BITS}/V{C0_V_BITS} family=kivi",
        "rotation_policy": f"K{C0_K_BITS}/V{C0_V_BITS} family=rotation_kivi",
        "reference_model_class": ref_model_class,
        "rotation_model_class": rot_model_class,
        "prompt_len": rot["prompt_len"],
        "num_layer0_calls": {"reference": len(ref_calls), "rotation": len(rot_calls)},
        "prefill_check": prefill_check,
        "cache_checks": {
            "rotation_key_quant_trans_is_none": cache_checks["rotation_key_quant_trans_is_none"],
            "reference_key_quant_trans_is_none": cache_checks["reference_key_quant_trans_is_none"],
            "rotation_key_full_vs_K_expected_rot": cache_checks["rotation_key_full_vs_K_expected_rot"],
            "reference_key_full_vs_raw_post_rope": cache_checks["reference_key_full_vs_raw_post_rope"],
        },
        "value_untouched_check": value_check,
        "decode_invariance_records": decode_invariance_records,
        "production_comparison_records": production_comparison_records,
        "generated_ids_reference": ref["ids_with_hooks"],
        "generated_ids_rotation": rot["ids_with_hooks"],
        "generated_prefix_equal": generated_prefix_equal,
        "hook_neutrality": {"reference": ref["hook_neutral"], "rotation": rot["hook_neutral"]},
        "structural_gate": structural_gate,
    }


# ---------------------------------------------------------------------------
# C1: quantized-Key canary (K2/V16) -- implemented, NOT run this round
# ---------------------------------------------------------------------------

def run_c1(model_name_or_path=DEFAULT_MODEL_NAME, cache_dir=DEFAULT_CACHE_DIR):
    from scripts.h2_attention_decode_parity_canary import reconstruct_key_step, tensor_diff_report
    from utils.attention_decode_features import ShadowKVCache, key_decode_distortion
    from utils.hadamard import apply_hadamard_rotation, get_normalized_hadamard

    # --- REFERENCE: family="kivi" K2/V16 -- identical to H2's own key-axis canary ---
    ref_model, ref_tokenizer, ref_model_class = _fresh_model_load(model_name_or_path, cache_dir, C1_K_BITS, C1_V_BITS, "kivi")
    prompt = _load_prompt(ref_tokenizer, model_name_or_path)
    ref = _generate_and_capture(ref_model, ref_tokenizer, prompt)
    head_dim, num_kv_heads = ref["head_dim"], ref["num_kv_heads"]
    ref_calls = ref["calls"]
    _release_model(ref_model)

    ref_shadow = ShadowKVCache()
    ref_shadow.seed_prefill(ref_calls[0]["k_post_rope"], ref_calls[0]["v_current"])
    ref_distortions = {}
    for step_idx, call in enumerate(ref_calls[1:], start=1):
        L_kivi, L_fp16 = reconstruct_key_step(call, CANARY_GROUP_SIZE, C1_K_BITS, ref_shadow, head_dim, num_kv_heads)
        ref_distortions[step_idx] = key_decode_distortion(L_kivi, L_fp16)

    # --- ROTATION: family="rotation_kivi" K2/V16 (sequential load) ---
    rot_model, rot_tokenizer, rot_model_class = _fresh_model_load(model_name_or_path, cache_dir, C1_K_BITS, C1_V_BITS, "rotation_kivi")
    rot = _generate_and_capture(rot_model, rot_tokenizer, prompt)
    rot_calls = rot["calls"]
    _release_model(rot_model)

    H = get_normalized_hadamard(head_dim, rot_calls[0]["k_post_rope"].device, rot_calls[0]["k_post_rope"].dtype)

    # Option B (Section 14): a fully-rotated FP16 shadow, seeded/appended
    # with rotated K throughout -- matches what the REAL rotated-quantized
    # production cache holds, so reconstruct_key_step (reused unmodified)
    # can be fed a rotated "call" without any change to that function.
    rot_shadow_rotated = ShadowKVCache()
    rot_shadow_rotated.seed_prefill(apply_hadamard_rotation(rot_calls[0]["k_post_rope"], H), rot_calls[0]["v_current"])
    # A SEPARATE, purely-diagnostic raw (unrotated) shadow -- used only to
    # verify rotated-vs-raw FP16 invariance holds at this data (never fed
    # into the distortion measurement itself; raw Q is never mixed with
    # rotated K or vice versa).
    rot_shadow_raw = ShadowKVCache()
    rot_shadow_raw.seed_prefill(rot_calls[0]["k_post_rope"], rot_calls[0]["v_current"])

    rot_distortions = {}
    invariance_cross_check_records = []
    for step_idx, call in enumerate(rot_calls[1:], start=1):
        Q_raw = call["q_post_rope"]
        K_raw = call["k_post_rope"]
        Q_rot = apply_hadamard_rotation(Q_raw, H)
        K_rot = apply_hadamard_rotation(K_raw, H)

        rotated_call = dict(call)
        rotated_call["q_post_rope"] = Q_rot
        rotated_call["k_post_rope"] = K_rot
        # reconstruct_key_step reads call["past_key_value_in"] -- the REAL
        # production past_key_value, whose quantized/full Key material was
        # produced from the ACTUAL rotated K (Stage I1B), never reimplemented.
        L_kivi_rot, L_fp16_rot = reconstruct_key_step(rotated_call, CANARY_GROUP_SIZE, C1_K_BITS, rot_shadow_rotated, head_dim, num_kv_heads)
        rot_distortions[step_idx] = key_decode_distortion(L_kivi_rot, L_fp16_rot)

        # Diagnostic-only: raw shadow advanced identically, for the
        # required rotation-invariance cross-check (never used for distortion).
        rot_shadow_raw.append_decode_step(K_raw, call["v_current"])
        import math

        L_fp16_raw = (Q_raw @ rot_shadow_raw.k().transpose(2, 3)) / math.sqrt(head_dim)
        invariance_cross_check_records.append(
            {"step": step_idx, **tensor_diff_report(L_fp16_rot, L_fp16_raw, f"rotated_fp16_vs_raw_fp16_step{step_idx}")}
        )

    return {
        "mode": "c1",
        "reference_policy": f"K{C1_K_BITS}/V{C1_V_BITS} family=kivi",
        "rotation_policy": f"K{C1_K_BITS}/V{C1_V_BITS} family=rotation_kivi",
        "reference_model_class": ref_model_class,
        "rotation_model_class": rot_model_class,
        "num_layer0_calls": {"reference": len(ref_calls), "rotation": len(rot_calls)},
        "reference_key_decode_distortion_by_step": ref_distortions,
        "rotation_key_decode_distortion_by_step": rot_distortions,
        "rotation_invariance_cross_check": invariance_cross_check_records,
        "generated_ids_reference": ref["ids_with_hooks"],
        "generated_ids_rotation": rot["ids_with_hooks"],
        "generated_prefix_required_to_match": False,
        "hook_neutrality": {"reference": ref["hook_neutral"], "rotation": rot["hook_neutral"]},
    }


# ---------------------------------------------------------------------------
# C0D: numerical attribution diagnostic (Stage I1-C0D). Read-only relative
# to run_c0/run_c1/compute_c0_structural_gate -- does not call, wrap, or
# modify any of them, and does not touch models/llama_kivi.py,
# utils/hadamard.py, or the quant/ kernels. Only ONE fresh model load
# (rotation_kivi K16/V16); the reference route is never re-run here --
# Section 10's downstream comparison is satisfied by quoting the ALREADY-
# FROZEN original C0 result rather than recomputing it.
# ---------------------------------------------------------------------------

def run_c0diag(model_name_or_path=DEFAULT_MODEL_NAME, cache_dir=DEFAULT_CACHE_DIR):
    """Determines whether the C0 numerical differences come from ordinary
    FP16 representation rounding (Section 6/7: an FP32 orthogonal oracle
    isolates pure rotation math from any storage precision; an FP16
    representation oracle then isolates the cost of storing/using that
    rotated representation at FP16) or a genuine mismatch between the
    intended rotated computation and the real production path (Section 8:
    production pre_softmax_logits vs the independently reconstructed FP16
    rotated reference). Defines NO new pass/fail gate -- raw numbers only,
    per Section 2/8/11.
    """
    import math

    import torch

    from scripts.h2_attention_decode_parity_canary import tensor_diff_report
    from utils.attention_decode_features import ShadowKVCache
    from utils.hadamard import apply_hadamard_rotation, get_normalized_hadamard, normalized_hadamard_matrix

    model, tokenizer, model_class = _fresh_model_load(model_name_or_path, cache_dir, C0_K_BITS, C0_V_BITS, "rotation_kivi")
    prompt = _load_prompt(tokenizer, model_name_or_path)
    rot = _generate_and_capture(model, tokenizer, prompt)
    head_dim = rot["head_dim"]
    calls = rot["calls"]
    _release_model(model)

    if len(calls) < 3:
        raise RuntimeError(f"expected >=3 layer-0 calls, got {len(calls)}")

    prefill_call = calls[0]
    decode_calls = calls[1:]

    # --- Section 5: raw FP16 shadow history, built ONLY from captured raw
    # post-RoPE tensors, entirely outside the production cache, never fed
    # back into generation (generation already finished above). ---
    raw_shadow = ShadowKVCache()
    raw_shadow.seed_prefill(prefill_call["k_post_rope"], prefill_call["v_current"])

    H32 = normalized_hadamard_matrix(head_dim, dtype=torch.float32).to(prefill_call["k_post_rope"].device)

    def _prob_diff(a, b):
        diff = (a.float() - b.float()).abs()
        diff_norm = torch.linalg.vector_norm((a.float() - b.float()).double())
        ref_norm = torch.clamp(torch.linalg.vector_norm(b.float().double()), min=1e-12)
        return {"max_abs_diff": float(diff.max().item()), "relative_l2": float((diff_norm / ref_norm).item())}

    step_records = []
    for step_idx, call in enumerate(decode_calls, start=1):
        # "K = full raw FP16 shadow cache through the current step" -- this
        # step's own raw K must already be appended before it is used.
        raw_shadow.append_decode_step(call["k_post_rope"], call["v_current"])

        Q = call["q_post_rope"]
        K = raw_shadow.k()

        # --- Section 6: FP32 orthogonal oracle -- no production tensor involved ---
        Qf = Q.float()
        Kf = K.float()
        L_raw32 = torch.matmul(Qf, Kf.transpose(2, 3)) / math.sqrt(head_dim)
        QH32 = torch.matmul(Qf, H32)
        KH32 = torch.matmul(Kf, H32)
        L_rot32 = torch.matmul(QH32, KH32.transpose(2, 3)) / math.sqrt(head_dim)
        fp32_orthogonal_oracle = tensor_diff_report(L_rot32, L_raw32, f"fp32_raw_vs_rotated_step{step_idx}")

        # --- Section 7: FP16 representation oracle -- same effective dtype
        # semantics as production (fp16 tensors, fp16-dtype matmul output,
        # division by a python float keeps the fp16 dtype). ---
        QH16 = QH32.to(torch.float16)
        KH16 = KH32.to(torch.float16)
        L_rot16_ref = torch.matmul(QH16, KH16.transpose(2, 3)) / math.sqrt(head_dim)
        fp16_representation_error = tensor_diff_report(L_rot16_ref, L_rot32, f"fp32_rotated_vs_fp16_representation_step{step_idx}")

        # Diagnostic-only: the CURRENT implementation's own rotation
        # utility against this same manual construction -- proves
        # apply_hadamard_rotation itself is not an additional error source.
        H16 = get_normalized_hadamard(head_dim, Q.device, torch.float16)
        Q_via_apply = apply_hadamard_rotation(Q, H16)
        K_via_apply = apply_hadamard_rotation(K, H16)
        apply_hadamard_vs_manual_Q = tensor_diff_report(Q_via_apply, QH16, f"apply_hadamard_vs_manual_Q_step{step_idx}")
        apply_hadamard_vs_manual_K = tensor_diff_report(K_via_apply, KH16, f"apply_hadamard_vs_manual_K_step{step_idx}")

        # --- Section 8: production attribution (load-bearing) ---
        L_production = call.get("pre_softmax_logits")
        production_attribution = (
            tensor_diff_report(L_production, L_rot16_ref, f"production_vs_fp16_reference_step{step_idx}")
            if L_production is not None else None
        )

        # --- Section 9: softmax probability diagnostic (descriptive only, no gate) ---
        p_raw32 = torch.softmax(L_raw32.float(), dim=-1)
        p_rot32 = torch.softmax(L_rot32.float(), dim=-1)
        p_rot16_ref = torch.softmax(L_rot16_ref.float(), dim=-1)
        softmax_diagnostic = {
            "raw32_vs_rot32": _prob_diff(p_raw32, p_rot32),
            "rot32_vs_rot16_ref": _prob_diff(p_rot32, p_rot16_ref),
        }
        post_softmax_production = call.get("post_softmax_weights")
        if post_softmax_production is not None:
            softmax_diagnostic["rot16_ref_vs_production"] = _prob_diff(p_rot16_ref, post_softmax_production.float())

        step_records.append(
            {
                "step": step_idx,
                "fp32_orthogonal_oracle": fp32_orthogonal_oracle,
                "fp16_representation_error": fp16_representation_error,
                "apply_hadamard_vs_manual_Q": apply_hadamard_vs_manual_Q,
                "apply_hadamard_vs_manual_K": apply_hadamard_vs_manual_K,
                "production_attribution": production_attribution,
                "softmax_diagnostic": softmax_diagnostic,
            }
        )

    return {
        "mode": "c0diag",
        "policy": f"K{C0_K_BITS}/V{C0_V_BITS} family=rotation_kivi",
        "model_class": model_class,
        "num_layer0_calls": len(calls),
        "step_records": step_records,
        "generated_ids": rot["ids_with_hooks"],
        "hook_neutral": rot["hook_neutral"],
        "downstream_output_comparison_note": (
            "attn_output_pre_o_proj vs standard K16/V16 is NOT recomputed here (Section 10) -- "
            "see the frozen original C0 result at "
            "outputs/i1_rotation_kivi_canary/c0/20260907T054459769185Z/parity_summary.json "
            "for that descriptive confirmation (all 5 steps allclose_1e-3=true)."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["c0", "c1", "c0diag"], required=False, default=None)
    p.add_argument("--run", action="store_true", help="Real GPU execution. Required for any mode to touch the GPU.")
    p.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME)
    p.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    return p.parse_args(argv)


def main():
    args = parse_args()

    if not args.run:
        print(
            "Refusing to proceed: --run was not given. Real Stage I1-C GPU canaries "
            "(--mode c0 / --mode c1 / --mode c0diag) never execute without an explicit --run flag.",
            file=sys.stderr,
        )
        return 1
    if args.mode is None:
        print("Refusing to proceed: --mode {c0,c1,c0diag} is required with --run.", file=sys.stderr)
        return 1

    try:
        collection_head, preflight = preflight_checks()
    except (CanaryConfigError, CanaryLockError) as e:
        print(f"Refusing real GPU canary: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    output_root = {"c0": C0_OUTPUT_ROOT, "c1": C1_OUTPUT_ROOT, "c0diag": C0DIAG_OUTPUT_ROOT}[args.mode]
    run_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = os.path.join(output_root, run_label)
    os.makedirs(out_dir, exist_ok=True)
    lock_fh = acquire_lock(os.path.join(out_dir, "canary.lock"))
    log_path = os.path.join(out_dir, "host_monitor.log")

    def log(msg):
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line)

    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; this canary requires a real GPU.")

        boot_id_start = get_boot_id()
        gpu_before = gpu_snapshot()
        log(f"start mode={args.mode} run_label={run_label} boot_id={boot_id_start} HEAD={collection_head}")

        summary = {
            "mode": args.mode,
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
            "canary_layer": CANARY_LAYER,
            "canary_seed": CANARY_SEED,
            "group_size": CANARY_GROUP_SIZE,
            "residual_length": CANARY_RESIDUAL_LENGTH,
            "model_name_or_path": args.model_name_or_path,
        }
        exit_code = 0
        try:
            mode_fn = {"c0": run_c0, "c1": run_c1, "c0diag": run_c0diag}[args.mode]
            summary["result"] = mode_fn(args.model_name_or_path, args.cache_dir)
        except Exception as e:  # noqa: BLE001 -- canary must record, not crash uncaught
            summary["error"] = f"{type(e).__name__}: {e}"
            exit_code = 1

        boot_id_end = get_boot_id()
        summary["boot_id_end"] = boot_id_end
        summary["boot_id_stable"] = boot_id_start == boot_id_end
        summary["gpu_after"] = gpu_snapshot()
        summary["exit_code"] = exit_code

        with open(os.path.join(out_dir, "parity_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        log(f"end exit_code={exit_code}")
        print(json.dumps(summary, indent=2, default=str))
        print(f"\nArtifacts written to: {out_dir}")
        return exit_code
    finally:
        release_lock(lock_fh)


if __name__ == "__main__":
    sys.exit(main())
