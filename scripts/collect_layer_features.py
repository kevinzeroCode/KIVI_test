"""Layer x task K/V feature-collection driver.

Stage G1 built --dry-run configuration validation only (no torch/transformers
import, no model, no forward pass). Stage G3A adds the real collection path,
reusing the exact Stage-G2-validated measurement logic verbatim:
  - Key features use the production-equivalent post-RoPE Key representation
    (k_proj hook -> reshape -> production rotary_emb -> in-place RoPE ->
    residual-aware slice -> transpose(2,3), matching models/llama_kivi.py's
    LlamaFlashAttention_KIVI prefill branch exactly).
  - Value features use the production-equivalent Value representation
    (v_proj hook -> reshape -> residual-aware slice, no transpose, no RoPE).
  - Reconstruction features always use the real Triton
    quantize_and_pack_along_last_dim + the G2-confirmed compatible
    unpack_and_dequant_vcache dequantizer -- never the CPU reference
    quantizer (utils.feature_extraction.cpu_reference_quantize_dequantize
    is not imported by this module at all).
Every captured production-quantizer-input tensor is cross-checked against
the manually reconstructed tensor before a feature record is emitted; a
mismatch raises rather than silently emitting an unverified record.

Does NOT (this round): run LongBench generation, write to pred/ or any
Stage-D/F0 output directory, collect a full calibration set, compute
feature/sensitivity correlations, implement attention geometry.

Usage (dry run, safe, CPU-only, no model import):
    ./.venv/bin/python scripts/collect_layer_features.py --dry-run

Usage (real collection -- tiny canary):
    CUDA_VISIBLE_DEVICES=0 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \\
    ./.venv/bin/python scripts/collect_layer_features.py \\
        --tasks lcc --num-samples 1 --layers 0 31 --axes key value \\
        --output-root outputs/layer_feature_collector_canary
"""
import argparse
import gc
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from utils.pilot_policy import (  # noqa: E402
    EXPECTED_NUM_LAYERS,
    PILOT_AXES,
    SUPPORTED_TASK_COUNTS,
    select_task_counts,
)
from utils.feature_schema import FEATURE_RECORD_FIELDS, IDENTITY_FIELDS  # noqa: E402 -- torch-free
from utils.layer_policy import resolve_layer_policy  # noqa: E402 -- torch-free
from utils.jsonl_integrity import inspect_jsonl  # noqa: E402 -- torch-free

DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "outputs", "layer_feature_pilot")
DEFAULT_TASKS = ("trec", "lcc", "passage_retrieval_en", "2wikimqa")
DEFAULT_NUM_SAMPLES = 20
DEFAULT_PROBE_BITS = 2

# Mirrors run_layer_sensitivity_pilot.py's conflicting-process discipline
# (see that script's check_no_conflicting_process docstring for why the
# genuine-invocation regex and "bash -c" exclusion exist).
CONFLICTING_PROCESS_PATTERNS = [
    "pred_long_bench.py",
    "layer_policy_smoke.py",
    "run_layer_sensitivity_pilot.py",
    "collect_layer_features.py",
]

# Reconstruction/distribution fields that are mathematically non-negative by
# construction; used by validate_feature_jsonl's sanity check.
NONNEGATIVE_FIELDS = (
    "relative_l2", "mse", "max_abs_error", "variance", "std",
    "max_abs", "p50_abs", "p95_abs", "p99_abs", "outlier_fraction",
)


class FeatureCollectionConfigError(ValueError):
    pass


class _Args:
    """Minimal stand-in for utils.process_args.ModelArguments/TrainingArguments
    -- only the fields pred_long_bench.build_model_and_tokenizer() reads."""


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


# ---------------------------------------------------------------------------
# Pure (torch-free) orchestration logic -- CPU-testable without CUDA.
# ---------------------------------------------------------------------------

def validate_layers(layers, num_hidden_layers=EXPECTED_NUM_LAYERS):
    if not layers:
        raise FeatureCollectionConfigError("--layers must be non-empty")
    bad = [l for l in layers if not (0 <= l < num_hidden_layers)]
    if bad:
        raise FeatureCollectionConfigError(f"layer index/indices out of range [0, {num_hidden_layers}): {bad}")
    if len(set(layers)) != len(layers):
        raise FeatureCollectionConfigError(f"duplicate layer indices in --layers: {layers}")
    return list(layers)


def validate_axes(axes, supported_axes=PILOT_AXES):
    if not axes:
        raise FeatureCollectionConfigError("--axes must be non-empty")
    bad = [a for a in axes if a not in supported_axes]
    if bad:
        raise FeatureCollectionConfigError(f"unsupported axis/axes {bad}; supported: {list(supported_axes)}")
    return list(axes)


def validate_num_samples(num_samples):
    if num_samples <= 0:
        raise FeatureCollectionConfigError(f"--num-samples must be positive, got {num_samples}")
    return num_samples


def build_collection_plan(layers, axes, task_counts, num_samples):
    """Returns the deterministic, ordered list of (task, layer, axis) cells
    this configuration WOULD collect features for. Never merges tasks --
    preserves feature(layer, task) granularity per the Stage G0/G1 design
    requirement that layer-index alone cannot explain task-dependent
    response.
    """
    plan = []
    for task in task_counts:
        n = min(num_samples, task_counts[task])
        for layer in layers:
            for axis in axes:
                plan.append({"task": task, "layer_idx": layer, "axis": axis, "num_samples": n})
    return plan


def key_residual_tokens(distribution_tokens, residual_length):
    """Key's KIVI residual is a REMAINDER, not a fixed window: production
    (models/llama_kivi.py's LlamaFlashAttention_KIVI.forward prefill
    branch) keeps exactly (distribution_tokens % residual_length) tokens at
    full precision -- whatever is left over after taking the largest
    multiple of residual_length -- so the quantized portion is always an
    exact multiple of residual_length (itself a multiple of group_size),
    which the token-grouped Key quantizer requires. If distribution_tokens
    < residual_length, nothing is quantized at all and every token is
    residual (production's key_states_quant=None case).

    Verified against the real Stage-G3A lcc sample: distribution_tokens=18062,
    residual_length=128 -> 18062 % 128 == 14.
    """
    if distribution_tokens < residual_length:
        return distribution_tokens
    return distribution_tokens % residual_length


def value_residual_tokens(distribution_tokens, residual_length):
    """Value's KIVI residual is a FIXED-SIZE sliding window: production
    always keeps exactly `residual_length` of the most recent tokens at
    full precision when distribution_tokens > residual_length (or the
    entire sequence if it's not longer than residual_length), independent
    of any alignment -- because V's quantizer groups along the channel
    (head_dim) axis, not the token axis, so there is no token-count
    alignment constraint to satisfy. This is why Key and Value residual
    token counts differ in KIND, not just magnitude.

    Verified against the real Stage-G3A lcc sample: distribution_tokens=18062,
    residual_length=128 -> residual is exactly 128 (not a remainder).
    """
    if distribution_tokens <= residual_length:
        return distribution_tokens
    return residual_length


def build_probe_policy_obj(layers, k_bits, v_bits):
    """Pure dict builder for a utils.layer_policy policy_obj: every layer in
    `layers` gets (k_bits, v_bits); every other layer stays K16/V16
    pass-through. Both axes are always quantized together at a probed layer
    (policy is per-layer, not per-axis) -- --axes only filters which
    feature records get EMITTED, not which tensors get quantized.
    """
    return {
        "policy_name": "layer_feature_collector_canary",
        "default": {"k_bits": 16, "v_bits": 16, "family": "kivi"},
        "overrides": {str(l): {"k_bits": k_bits, "v_bits": v_bits, "family": "kivi"} for l in layers},
    }


def expected_call_order(layers):
    """Deterministic order in which models/llama_kivi.py's decoder stack
    will call the per-axis production quantizer entry points
    (_chunked_key_quantize_and_pack_along_last_dim /
    _chunked_value_quantize_and_pack_along_last_dim): layers are visited in
    ascending index order (sequential decoder-stack traversal). Each of
    these two entry points is called EXACTLY ONCE per layer regardless of
    prompt length -- internal token-range chunking (key_quant_chunk_size /
    value_quant_chunk_size, default 512) happens *inside* them and is
    already concatenated into a single code/scale/mn result before
    returning, so patching at this level (rather than the lower-level
    per-chunk triton_quantize_and_pack_along_last_dim) keeps call count
    independent of sequence length. Used to attribute each captured
    per-axis call to a layer_idx without needing self.layer_idx inside the
    spy."""
    return sorted(layers)


def expected_identities(task_sample_pairs, layers, axes):
    """The exact (task, sample_idx, layer_idx, tensor_axis) identity tuples
    a correct run should produce, given which (task, sample_idx) pairs were
    processed and which layers/axes were requested."""
    return [
        (task, sample_idx, layer, axis)
        for (task, sample_idx) in task_sample_pairs
        for layer in sorted(layers)
        for axis in axes
    ]


def validate_feature_jsonl(path, expected_ids):
    """Torch-free strict validation of a features.jsonl artifact: valid
    JSON, trailing newline, exact identity-set match (no missing/extra/
    duplicate rows), all non-identity fields present/finite, and the
    mathematically-non-negative fields are >= 0. Never asserts one layer's
    error must exceed another's -- only validity, not a scientific claim.
    """
    inspection = inspect_jsonl(path)
    problems = []
    if inspection.invalid_rows != 0:
        problems.append(f"{inspection.invalid_rows} invalid JSON row(s)")
    if not inspection.ends_with_newline:
        problems.append("file does not end with a trailing newline")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    seen_ids = []
    for row in rows:
        ident = (row.get("task"), row.get("sample_idx"), row.get("layer_idx"), row.get("tensor_axis"))
        seen_ids.append(ident)
        for field in FEATURE_RECORD_FIELDS:
            if field in IDENTITY_FIELDS:
                continue
            val = row.get(field)
            if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
                problems.append(f"{ident}: field {field!r} is not a finite number: {val!r}")
        for field in NONNEGATIVE_FIELDS:
            val = row.get(field)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and val < 0:
                problems.append(f"{ident}: field {field!r} is negative: {val!r}")

    if len(set(seen_ids)) != len(seen_ids):
        dupes = sorted({i for i in seen_ids if seen_ids.count(i) > 1})
        problems.append(f"duplicate identity tuple(s): {dupes}")

    if sorted(seen_ids) != sorted(expected_ids):
        problems.append(f"identity set mismatch: expected {sorted(expected_ids)}, got {sorted(seen_ids)}")

    return {"row_count": len(rows), "identities": seen_ids, "problems": problems, "ok": not problems}


# ---------------------------------------------------------------------------
# Real collection (heavy imports deferred into this function only).
# ---------------------------------------------------------------------------

def run_real_collection(args):
    import time

    import torch
    from datasets import load_dataset

    import pred_long_bench as plb
    import models.llama_kivi as llama_kivi
    from utils.layer_policy import canonical_policy_dict, policy_hash
    from utils.feature_extraction import build_feature_record, distribution_stats, reconstruction_stats
    from utils.generation_semantics import NO_BUILD_CHAT_DATASETS
    from quant.new_pack import unpack_and_dequant_vcache

    plb.seed_everything(args.seed)

    run_label = args.run_label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = os.path.join(args.output_root, run_label)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "host_monitor.log")

    def log(msg):
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line)

    boot_id_start = get_boot_id()
    wall_start = time.monotonic()
    log(f"start run_label={run_label} boot_id={boot_id_start}")

    task_names = list(args.tasks) if args.tasks else list(DEFAULT_TASKS)
    task_counts = select_task_counts(task_names)
    num_samples = validate_num_samples(args.num_samples)
    axes = validate_axes(args.axes)

    probe_config = plb.LlamaConfig.from_pretrained(args.model_name_or_path)
    num_hidden_layers = probe_config.num_hidden_layers
    layers = validate_layers(args.layers, num_hidden_layers)

    policy_obj = build_probe_policy_obj(layers, args.k_bits, args.v_bits)
    resolved = resolve_layer_policy(num_hidden_layers, 16, 16, policy_obj)

    conflicts = check_no_conflicting_process()
    if conflicts:
        raise FeatureCollectionConfigError(f"Conflicting process(es) detected, refusing to start: {conflicts}")

    model_args = _Args()
    model_args.model_name_or_path = args.model_name_or_path
    model_args.k_bits = 16
    model_args.v_bits = 16
    model_args.group_size = args.group_size
    model_args.residual_length = args.residual_length
    training_args = _Args()
    training_args.cache_dir = args.cache_dir
    dtype = torch.float16

    log("loading model")
    model, tokenizer, model_class_name = plb.build_model_and_tokenizer(
        model_args, training_args, dtype, use_kivi_model=True, resolved_layer_policy=resolved
    )
    model.eval()
    log(f"model loaded class={model_class_name}")

    for l in layers:
        attn = model.model.layers[l].self_attn
        got = (attn.k_bits, attn.v_bits)
        if got != (args.k_bits, args.v_bits):
            raise FeatureCollectionConfigError(f"layer {l} policy mismatch: expected {(args.k_bits, args.v_bits)}, got {got}")
    for i, dl in enumerate(model.model.layers):
        if i in layers:
            continue
        got = (dl.self_attn.k_bits, dl.self_attn.v_bits)
        if got != (16, 16):
            raise FeatureCollectionConfigError(f"layer {i} is not K16/V16 control: got {got}")

    device = torch.device(args.device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    with open(os.path.join(REPO_ROOT, "config", "dataset2prompt.json"), "r", encoding="utf-8") as f:
        dataset2prompt = json.load(f)

    real_chunked_key = llama_kivi._chunked_key_quantize_and_pack_along_last_dim
    real_chunked_value = llama_kivi._chunked_value_quantize_and_pack_along_last_dim
    sorted_layers = expected_call_order(layers)

    features_path = os.path.join(out_dir, "features.jsonl")
    sample_records = []
    task_sample_pairs = []
    nonfinite_any = False

    with open(features_path, "w", encoding="utf-8") as feat_f:
        for task in task_counts:
            data = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
            n = min(num_samples, task_counts[task], len(data))
            for sample_idx in range(n):
                task_sample_pairs.append((task, sample_idx))
                json_obj = data[sample_idx]
                prompt = dataset2prompt[task].format(**json_obj)
                tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
                truncated = len(tokenized_prompt) > args.max_length
                if truncated:
                    half = int(args.max_length / 2)
                    prompt = (
                        tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
                        + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
                    )
                if task not in NO_BUILD_CHAT_DATASETS:
                    prompt = plb.build_chat(tokenizer, prompt, args.model_name_or_path.split("/")[-1])
                inp = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
                input_ids = inp.input_ids
                attention_mask = inp.attention_mask
                num_tokens_input = input_ids.shape[1]

                log(f"task={task} sample_idx={sample_idx} input_tokens={num_tokens_input} truncated={truncated}")

                captured = {}
                handles = []
                for l in layers:
                    attn = model.model.layers[l].self_attn

                    def make_hook(layer_idx, key):
                        def hook(module, hook_in, hook_out):
                            captured[(layer_idx, key)] = hook_out.detach().clone()
                        return hook

                    handles.append(attn.k_proj.register_forward_hook(make_hook(l, "k_proj")))
                    handles.append(attn.v_proj.register_forward_hook(make_hook(l, "v_proj")))

                key_calls = []
                value_calls = []

                def spy_chunked_key(data_, group_size_, bit_, chunk_size_):
                    result = real_chunked_key(data_, group_size_, bit_, chunk_size_)
                    key_calls.append(
                        {
                            "tensor": data_.detach().clone(),
                            "code": result[0].detach().clone(),
                            "scale": result[1].detach().clone(),
                            "mn": result[2].detach().clone(),
                            "group_size": group_size_,
                            "bit": bit_,
                        }
                    )
                    return result

                def spy_chunked_value(data_, group_size_, bit_, chunk_size_):
                    result = real_chunked_value(data_, group_size_, bit_, chunk_size_)
                    value_calls.append(
                        {
                            "tensor": data_.detach().clone(),
                            "code": result[0].detach().clone(),
                            "scale": result[1].detach().clone(),
                            "mn": result[2].detach().clone(),
                            "group_size": group_size_,
                            "bit": bit_,
                        }
                    )
                    return result

                forward_start = time.monotonic()
                with mock.patch.object(llama_kivi, "_chunked_key_quantize_and_pack_along_last_dim", spy_chunked_key), \
                     mock.patch.object(llama_kivi, "_chunked_value_quantize_and_pack_along_last_dim", spy_chunked_value):
                    with torch.no_grad():
                        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                forward_elapsed = time.monotonic() - forward_start

                for h in handles:
                    h.remove()

                nonfinite_logits = not bool(torch.isfinite(out.logits).all().item())
                nonfinite_any = nonfinite_any or nonfinite_logits

                # Both axes are always quantized at every probed layer (the
                # policy is per-layer, not per-axis -- see
                # build_probe_policy_obj), so both wrappers fire once per
                # probed layer regardless of which --axes were requested;
                # --axes only filters which records get EMITTED below.
                if len(key_calls) != len(sorted_layers):
                    raise FeatureCollectionConfigError(
                        f"expected {len(sorted_layers)} Key quantizer calls (1 per probed layer), got {len(key_calls)}"
                    )
                if len(value_calls) != len(sorted_layers):
                    raise FeatureCollectionConfigError(
                        f"expected {len(sorted_layers)} Value quantizer calls (1 per probed layer), got {len(value_calls)}"
                    )

                bsz, q_len = input_ids.shape
                calls_by_axis_and_layer = {("key", l): c for l, c in zip(sorted_layers, key_calls)}
                calls_by_axis_and_layer.update({("value", l): c for l, c in zip(sorted_layers, value_calls)})
                axis_layer_pairs = [(layer_idx, call_axis) for layer_idx in sorted_layers for call_axis in ("key", "value")]

                for layer_idx, call_axis in axis_layer_pairs:
                    if call_axis not in axes:
                        continue
                    call = calls_by_axis_and_layer[(call_axis, layer_idx)]
                    attn = model.model.layers[layer_idx].self_attn
                    spy_tensor = call["tensor"]
                    raw = captured[(layer_idx, "k_proj" if call_axis == "key" else "v_proj")]
                    reshaped = raw.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2).contiguous()

                    if call_axis == "key":
                        # distribution_tensor = the full family-agnostic post-RoPE
                        # Key representation, BEFORE KIVI's residual-remainder
                        # exclusion -- this is what the distribution features
                        # characterize, independent of KIVI's own quantized-subset
                        # policy (see utils/feature_schema.py's module docstring).
                        position_ids = torch.arange(q_len, device=device).unsqueeze(0)
                        with torch.no_grad():
                            cos, sin = attn.rotary_emb(reshaped, position_ids)
                            dummy_q = reshaped.clone()
                            distribution_tensor = reshaped.clone()
                            _, distribution_tensor = llama_kivi._apply_rotary_pos_emb_inplace(
                                dummy_q, distribution_tensor, cos, sin, position_ids, attn.rotary_chunk_size
                            )
                        rl = attn.residual_length
                        distribution_tokens = distribution_tensor.shape[-2]
                        residual_tokens = key_residual_tokens(distribution_tokens, rl)
                        if residual_tokens == distribution_tokens:
                            raise FeatureCollectionConfigError(
                                f"prefix too short to trigger Key quantization at layer {layer_idx}"
                            )
                        quant_part = distribution_tensor[:, :, : distribution_tokens - residual_tokens, :]
                        manual = quant_part.transpose(2, 3).contiguous()
                        quantized_tokens = manual.shape[-1]
                    else:
                        # distribution_tensor = the full family-agnostic Value
                        # representation, BEFORE KIVI's fixed-size residual-window
                        # exclusion.
                        distribution_tensor = reshaped
                        rl = attn.residual_length
                        distribution_tokens = distribution_tensor.shape[-2]
                        residual_tokens = value_residual_tokens(distribution_tokens, rl)
                        if residual_tokens == distribution_tokens:
                            raise FeatureCollectionConfigError(
                                f"prefix too short to trigger Value quantization at layer {layer_idx}"
                            )
                        manual = distribution_tensor[:, :, : distribution_tokens - residual_tokens, :].contiguous()
                        quantized_tokens = manual.shape[-2]

                    if manual.shape != spy_tensor.shape or not torch.allclose(
                        manual.float(), spy_tensor.float(), atol=1e-3, rtol=1e-3
                    ):
                        raise FeatureCollectionConfigError(
                            f"call-order/layer attribution parity check FAILED for layer {layer_idx} "
                            f"axis {call_axis} -- refusing to emit an unverified feature record"
                        )

                    dequant = unpack_and_dequant_vcache(
                        call["code"], call["scale"].unsqueeze(-1), call["mn"].unsqueeze(-1), call["group_size"], call["bit"]
                    )
                    # KIVI-specific reconstruction population: exactly the
                    # G2-confirmed production-quantizer input/output pair.
                    recon = reconstruction_stats(spy_tensor.float(), dequant.float())
                    # Family-agnostic distribution population: the full
                    # real FP16 representation, not the quantized subset.
                    dist = distribution_stats(distribution_tensor.float())

                    record = build_feature_record(
                        task=task,
                        sample_idx=sample_idx,
                        layer_idx=layer_idx,
                        tensor_axis=call_axis,
                        input_tokens=num_tokens_input,
                        distribution_tokens=distribution_tokens,
                        quantized_tokens=quantized_tokens,
                        residual_tokens=residual_tokens,
                        recon_stats=recon,
                        dist_stats=dist,
                        aggregation="flattened_full_tensor_no_aggregation",
                    )
                    record["bits"] = call["bit"]
                    record["group_size"] = call["group_size"]

                    feat_f.write(json.dumps(record, sort_keys=False) + "\n")
                    feat_f.flush()
                    os.fsync(feat_f.fileno())

                sample_records.append(
                    {
                        "task": task,
                        "sample_idx": sample_idx,
                        "input_tokens": num_tokens_input,
                        "truncated": truncated,
                        "forward_seconds": forward_elapsed,
                    }
                )

                del captured, key_calls, value_calls, calls_by_axis_and_layer, out
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    wall_elapsed = time.monotonic() - wall_start
    peak_mem_allocated = int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else None
    peak_mem_reserved = int(torch.cuda.max_memory_reserved(device)) if torch.cuda.is_available() else None

    exp_ids = expected_identities(task_sample_pairs, layers, axes)
    validation = validate_feature_jsonl(features_path, exp_ids)

    boot_id_end = get_boot_id()
    exit_status = "OK" if validation["ok"] and not nonfinite_any else "ERROR"

    run_config = {
        "run_label": run_label,
        "args": vars(args),
        "resolved_layer_policy_hash": policy_hash(resolved),
        "resolved_layer_policy": canonical_policy_dict(resolved),
        "num_hidden_layers": num_hidden_layers,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    manifest = {
        "run_label": run_label,
        "boot_id_start": boot_id_start,
        "boot_id_end": boot_id_end,
        "boot_id_stable": boot_id_start == boot_id_end,
        "git_commit": get_git_commit(),
        "git_status_short": get_git_status_short(),
        "model_class": model_class_name,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "samples": sample_records,
        "feature_row_count": validation["row_count"],
        "feature_validation_problems": validation["problems"],
        "nonfinite_logits_seen_any_sample": nonfinite_any,
        "wall_clock_seconds": wall_elapsed,
        "peak_gpu_memory_allocated_bytes": peak_mem_allocated,
        "peak_gpu_memory_reserved_bytes": peak_mem_reserved,
        "features_jsonl_bytes": os.path.getsize(features_path),
        "exit_status": exit_status,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log(f"end exit_status={exit_status} wall_seconds={wall_elapsed:.2f} rows={validation['row_count']}")
    print(json.dumps(manifest, indent=2, default=str))
    print(f"\nArtifacts written to: {out_dir}")

    return 0 if exit_status == "OK" else 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name_or_path", default="lmsys/longchat-7b-v1.5-32k")
    p.add_argument("--cache_dir", default="./cached_models")
    p.add_argument("--tasks", nargs="+", default=None, help="Defaults to the same 4-task Stage-D screening set. Any task in utils.pilot_policy.SUPPORTED_TASK_COUNTS may be requested explicitly.")
    p.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Per-task sample cap (capped further by that task's available example count). Sample indices are always the deterministic first N of the dataset's test split.")
    p.add_argument("--layers", type=int, nargs="+", default=list(range(EXPECTED_NUM_LAYERS)))
    p.add_argument("--axes", nargs="+", default=list(PILOT_AXES), choices=list(PILOT_AXES))
    p.add_argument("--k-bits", type=int, default=DEFAULT_PROBE_BITS, help="Bit width applied to K at every probed layer (both axes are always quantized together; --axes only filters which feature records are emitted).")
    p.add_argument("--v-bits", type=int, default=DEFAULT_PROBE_BITS, help="Bit width applied to V at every probed layer.")
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--residual-length", type=int, default=128)
    p.add_argument("--max-length", type=int, default=31500)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--run_label", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main():
    args = parse_args()

    if args.dry_run:
        task_names = list(args.tasks) if args.tasks else list(DEFAULT_TASKS)
        task_counts = select_task_counts(task_names)
        layers = validate_layers(args.layers)
        axes = validate_axes(args.axes)
        num_samples = validate_num_samples(args.num_samples)

        plan = build_collection_plan(layers, axes, task_counts, num_samples)
        conflicts = check_no_conflicting_process()

        print(f"Selected tasks ({len(task_counts)}): {list(task_counts)}")
        print(f"Layers ({len(layers)}): {layers}")
        print(f"Axes ({len(axes)}): {axes}")
        print(f"Per-task sample cap: {num_samples}")
        print(f"Probe bits: K{args.k_bits}/V{args.v_bits} (both axes quantized together at each probed layer)")
        print(f"\nPlanned feature-record cells (task x layer x axis, per-sample granularity preserved): {len(plan)}")
        total_samples = sum(cell["num_samples"] for cell in plan)
        print(f"Planned total (task, layer, axis, sample) feature records: {total_samples}")
        print(f"\nFeature record schema ({len(FEATURE_RECORD_FIELDS)} fields): {list(FEATURE_RECORD_FIELDS)}")
        print(f"\nOutput root (not written this round): {args.output_root}")
        print(f"\nConflicting-process check ({', '.join(CONFLICTING_PROCESS_PATTERNS)}): "
              f"{'NONE FOUND' if not conflicts else conflicts}")
        print(
            "\nNo model, torch, or transformers import occurred; no activation capture, no "
            "forward pass, no output file was written (--dry-run configuration validation only)."
        )
        return 0

    return run_real_collection(args)


if __name__ == "__main__":
    sys.exit(main())
