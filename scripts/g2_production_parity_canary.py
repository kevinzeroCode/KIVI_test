"""Stage G2: Tiny Production-Parity GPU Canary.

Resolves the two Stage-G1 open questions that require CUDA:
  (1) PRODUCTION_QUANT_DEQUANT_PARITY -- does
      quant.new_pack.unpack_and_dequant_vcache correctly dequantize the
      output of quant.new_pack.triton_quantize_and_pack_along_last_dim
      (Triton kernel, CUDA-only)?
  (2) KEY_CAPTURE_GPU_PARITY / VALUE_CAPTURE_GPU_PARITY -- does manually
      reconstructing production's K/V tensor (via a k_proj/v_proj forward
      hook + the exact reshape/RoPE/slice steps read out of
      models/llama_kivi.py) numerically match the tensor production's own
      Triton quantizer entry point actually received?

This is NOT the feature pilot: no calibration dataset is collected, no
feature-response correlation is computed, no prediction JSONL is written,
no LongBench generation is run (a single deterministic prefill forward
pass only), and nothing is committed.

Two independent single-layer probes are run (layer 0 only, per the
Stage-G0/G1 pilot's own layer00_key_k2 / layer00_value_v2 policies):
  A. Key probe:   layer 0 = K2/V16,  all other layers = K16/V16
  B. Value probe: layer 0 = K16/V2,  all other layers = K16/V16

Writes only under outputs/layer_feature_parity_canary/. Never touches
pred/, outputs/layer_sensitivity_pilot/, outputs/layer_sensitivity_f0/, or
analysis/results/.

Usage:
    CUDA_VISIBLE_DEVICES=0 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \\
    ./.venv/bin/python scripts/g2_production_parity_canary.py
"""
import argparse
import gc
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "outputs", "layer_feature_parity_canary")
KEY_POLICY_PATH = os.path.join(REPO_ROOT, "analysis", "policies", "layer_sensitivity_pilot", "layer00_key_k2.json")
VALUE_POLICY_PATH = os.path.join(REPO_ROOT, "analysis", "policies", "layer_sensitivity_pilot", "layer00_value_v2.json")


class CanaryError(RuntimeError):
    """Any canary precondition/postcondition failure. Fails closed."""


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
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        procs = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return {"gpu": out, "compute_processes": procs}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Part 2: synthetic Triton pack/dequant parity (no model, small deterministic
# CUDA tensors only).
# ---------------------------------------------------------------------------

def run_synthetic_triton_parity(group_size=32):
    import torch
    from quant.new_pack import triton_quantize_and_pack_along_last_dim, unpack_and_dequant_vcache

    from utils.feature_extraction import reconstruction_stats

    device = torch.device("cuda:0")
    results = {}
    for bits in (2, 4):
        torch.manual_seed(1234 + bits)
        # (B, nh, D, T): D is the ungrouped "row" axis, T is the grouped
        # last axis -- matches both K's post-transpose layout (D=head_dim,
        # T=tokens) and V's direct layout (D=tokens, T=head_dim); the
        # kernel itself is agnostic to which semantic axis is which.
        B, nh, D, T = 1, 2, 128, 128
        x = torch.randn(B, nh, D, T, device=device, dtype=torch.float16)

        code, scale, mn = triton_quantize_and_pack_along_last_dim(x.contiguous(), group_size, bits)
        dequant = unpack_and_dequant_vcache(code, scale.unsqueeze(-1), mn.unsqueeze(-1), group_size, bits)

        code2, scale2, mn2 = triton_quantize_and_pack_along_last_dim(x.contiguous(), group_size, bits)
        dequant2 = unpack_and_dequant_vcache(code2, scale2.unsqueeze(-1), mn2.unsqueeze(-1), group_size, bits)
        deterministic = bool(torch.equal(code, code2)) and bool(torch.equal(dequant, dequant2))

        recon = reconstruction_stats(x.float(), dequant.float())

        results[f"bits{bits}"] = {
            "input_shape": list(x.shape),
            "input_dtype": str(x.dtype),
            "code_shape": list(code.shape),
            "code_dtype": str(code.dtype),
            "scale_shape": list(scale.shape),
            "mn_shape": list(mn.shape),
            "dequant_shape": list(dequant.shape),
            "dequant_dtype": str(dequant.dtype),
            "shape_matches_input": list(dequant.shape) == list(x.shape),
            "dtype_matches_input": dequant.dtype == x.dtype,
            "finite": bool(torch.isfinite(dequant).all().item()),
            "deterministic": deterministic,
            "reconstruction": recon,
            # NOT a CPU-reference-parity claim -- this compares the Triton
            # kernel's own round trip (quantize then dequantize) against
            # the original synthetic input, purely to sanity-check the
            # pipeline (finite/deterministic/plausible error), never
            # against utils.feature_extraction.cpu_reference_quantize_dequantize.
        }
    return results


def run_matmul_cross_check(group_size=32, bits=4):
    """Cross-checks unpack_and_dequant_vcache + a manual matmul against
    quant.matmul.cuda_bmm_fA_qB_outer's fused dequant-and-matmul kernel, on
    a small controlled probe. Both paths dequantize from the SAME
    code/scale/mn, so any discrepancy reflects a real layout/kernel
    inconsistency, not quantization error against the original tensor.

    cuda_bmm_fA_qB_outer is only ever called by models/llama_kivi.py inside
    the decode-step (past_key_value is not None) branch, where fA
    (query_states) has q_len == 1 -- an autoregressive single-token GEMV,
    matching the "gemv_forward_cuda_outer_dim" kernel name. This function
    therefore probes M=1 (production-faithful) as the primary check. It
    additionally probes M=2 and M=8 purely as an evidence-gathering step:
    an initial M=8 probe in this same investigation showed large
    disagreement (~40 max abs diff against a ~9-magnitude result) that
    vanished at M=1, indicating the kernel is only valid for the M=1
    decode-step regime it is actually used in -- NOT a bug in this probe's
    construction. That finding is reported, not hidden, but M>1 is never
    used by production so it does not affect PRODUCTION_QUANT_DEQUANT_PARITY.
    """
    import torch
    from quant.new_pack import triton_quantize_and_pack_along_last_dim, unpack_and_dequant_vcache
    from quant.matmul import cuda_bmm_fA_qB_outer

    device = torch.device("cuda:0")
    B, nh, K, N = 1, 2, 128, 128  # K = head_dim (contraction), N = tokens (multiple of group_size)

    probes = {}
    for M in (1, 2, 8):
        torch.manual_seed(4242)
        fA = torch.randn(B, nh, M, K, device=device, dtype=torch.float16)
        raw = torch.randn(B, nh, K, N, device=device, dtype=torch.float16)

        code, scale, mn = triton_quantize_and_pack_along_last_dim(raw.contiguous(), group_size, bits)
        dequant = unpack_and_dequant_vcache(code, scale.unsqueeze(-1), mn.unsqueeze(-1), group_size, bits)

        c_fused = cuda_bmm_fA_qB_outer(group_size, fA, code, scale, mn, bits)
        c_ref = torch.matmul(fA, dequant)

        shapes_match = list(c_fused.shape) == list(c_ref.shape)
        if shapes_match:
            diff = (c_fused.float() - c_ref.float()).abs()
            max_abs_diff = diff.max().item()
            mean_abs_diff = diff.mean().item()
            allclose = bool(torch.allclose(c_fused.float(), c_ref.float(), atol=5e-2, rtol=5e-2))
        else:
            max_abs_diff = mean_abs_diff = None
            allclose = False

        probes[f"M{M}"] = {
            "fA_shape": list(fA.shape),
            "raw_shape": list(raw.shape),
            "qB_code_shape": list(code.shape),
            "c_fused_shape": list(c_fused.shape),
            "c_ref_shape": list(c_ref.shape),
            "shapes_match": shapes_match,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "allclose_atol5e-2_rtol5e-2": allclose,
            "production_faithful": M == 1,
        }

    return {
        "bits": bits,
        "group_size": group_size,
        "probes": probes,
        "note": (
            "cuda_bmm_fA_qB_outer is production-faithful only at M=1 (single-token "
            "decode-step GEMV); M=2/M=8 are included only as evidence that the "
            "kernel diverges outside that regime, not as a production usage claim."
        ),
    }


# ---------------------------------------------------------------------------
# Parts 3-8: real single-layer, single-forward-pass capture.
# ---------------------------------------------------------------------------

class _Args:
    pass


def run_real_capture(axis, model_name_or_path, cache_dir, dataset, prefix_len, group_size, residual_length, seed):
    import torch
    from datasets import load_dataset

    import pred_long_bench as plb
    import models.llama_kivi as llama_kivi
    from utils.layer_policy import load_and_resolve, policy_hash
    from utils.feature_extraction import reconstruction_stats, distribution_stats
    from quant.new_pack import unpack_and_dequant_vcache

    plb.seed_everything(seed)

    policy_path = KEY_POLICY_PATH if axis == "key" else VALUE_POLICY_PATH
    probe_config = plb.LlamaConfig.from_pretrained(model_name_or_path)
    resolved = load_and_resolve(policy_path, probe_config.num_hidden_layers, 16, 16)

    model_args = _Args()
    model_args.model_name_or_path = model_name_or_path
    model_args.k_bits = 16
    model_args.v_bits = 16
    model_args.group_size = group_size
    model_args.residual_length = residual_length
    training_args = _Args()
    training_args.cache_dir = cache_dir
    dtype = torch.float16

    model, tokenizer, model_class_name = plb.build_model_and_tokenizer(
        model_args, training_args, dtype, use_kivi_model=True, resolved_layer_policy=resolved
    )
    model.eval()

    layer0 = model.model.layers[0].self_attn
    expected = {"key": (2, 16), "value": (16, 2)}[axis]
    got0 = (layer0.k_bits, layer0.v_bits)
    if got0 != expected:
        raise CanaryError(f"layer0 policy mismatch for axis={axis}: expected {expected}, got {got0}")
    for i, l in enumerate(model.model.layers):
        if i == 0:
            continue
        got = (l.self_attn.k_bits, l.self_attn.v_bits)
        if got != (16, 16):
            raise CanaryError(f"layer {i} is not K16/V16 control: got {got}")

    captured = {"k_proj_out": None, "v_proj_out": None}

    def make_hook(key):
        def hook(module, inp, output):
            captured[key] = output.detach().clone()
        return hook

    h_k = layer0.k_proj.register_forward_hook(make_hook("k_proj_out"))
    h_v = layer0.v_proj.register_forward_hook(make_hook("v_proj_out"))

    spy_calls = []
    real_quant = llama_kivi.triton_quantize_and_pack_along_last_dim

    def spy_quant(data, group_size_, bit):
        result = real_quant(data, group_size_, bit)
        spy_calls.append(
            {
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "device": str(data.device),
                "tensor": data.detach().clone(),
                "code": result[0].detach().clone(),
                "scale": result[1].detach().clone(),
                "mn": result[2].detach().clone(),
                "group_size": group_size_,
                "bit": bit,
            }
        )
        return result

    data = load_dataset("THUDM/LongBench", dataset, split="test", trust_remote_code=True)
    sample = data[0]
    with open(os.path.join(REPO_ROOT, "config", "dataset2prompt.json"), "r", encoding="utf-8") as f:
        dataset2prompt = json.load(f)
    prompt = dataset2prompt[dataset].format(**sample)

    full_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids
    if full_ids.shape[1] < prefix_len:
        raise CanaryError(f"sample too short: {full_ids.shape[1]} tokens < requested prefix_len {prefix_len}")
    device = torch.device("cuda:0")
    input_ids = full_ids[:, :prefix_len].to(device)

    with mock.patch.object(llama_kivi, "triton_quantize_and_pack_along_last_dim", spy_quant):
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)

    h_k.remove()
    h_v.remove()

    nonfinite_logits = not bool(torch.isfinite(out.logits).all().item())

    if len(spy_calls) != 1:
        raise CanaryError(
            f"expected exactly 1 production quantizer call for axis={axis} (layer 0 only, single chunk), got {len(spy_calls)}"
        )
    call = spy_calls[0]
    spy_tensor = call["tensor"]

    bsz, q_len = input_ids.shape
    num_kv_heads = layer0.num_key_value_heads
    head_dim = layer0.head_dim
    raw = captured["k_proj_out"] if axis == "key" else captured["v_proj_out"]
    if raw is None:
        raise CanaryError(f"{axis} proj hook did not fire")
    reshaped = raw.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2).contiguous()

    if axis == "key":
        position_ids = torch.arange(q_len, device=device).unsqueeze(0)
        with torch.no_grad():
            cos, sin = layer0.rotary_emb(reshaped, position_ids)
            dummy_q = reshaped.clone()
            key_rot = reshaped.clone()
            _, key_rot = llama_kivi._apply_rotary_pos_emb_inplace(
                dummy_q, key_rot, cos, sin, position_ids, layer0.rotary_chunk_size
            )
        rl = layer0.residual_length
        if key_rot.shape[-2] % rl != 0:
            if key_rot.shape[-2] < rl:
                raise CanaryError("prefix too short to trigger Key quantization")
            quant_part = key_rot[:, :, : -(key_rot.shape[-2] % rl), :]
        else:
            quant_part = key_rot
        manual = quant_part.transpose(2, 3).contiguous()
    else:
        rl = layer0.residual_length
        if reshaped.shape[-2] <= rl:
            raise CanaryError("prefix too short to trigger Value quantization")
        manual = reshaped[:, :, :-rl, :].contiguous()

    shape_eq = list(manual.shape) == list(spy_tensor.shape)
    dtype_eq = manual.dtype == spy_tensor.dtype
    if shape_eq:
        diff = (manual.float() - spy_tensor.float()).abs()
        max_abs_diff = diff.max().item()
        mean_abs_diff = diff.mean().item()
        diff_norm = torch.linalg.vector_norm(diff.double())
        orig_norm = torch.clamp(torch.linalg.vector_norm(manual.double()), min=1e-12)
        rel_l2 = (diff_norm / orig_norm).item()
        allclose = bool(torch.allclose(manual.float(), spy_tensor.float(), atol=1e-3, rtol=1e-3))
    else:
        max_abs_diff = mean_abs_diff = rel_l2 = None
        allclose = False

    parity = {
        "axis": axis,
        "manual_shape": list(manual.shape),
        "manual_dtype": str(manual.dtype),
        "spy_shape": list(spy_tensor.shape),
        "spy_dtype": str(spy_tensor.dtype),
        "shape_equal": shape_eq,
        "dtype_equal": dtype_eq,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "relative_l2_diff": rel_l2,
        "allclose_atol1e-3_rtol1e-3": allclose,
    }

    # Part 7: reconstruction-feature parity on the actual production
    # quantizer input, using the ACTUAL production quantize call's own
    # code/scale/mn (already captured by the spy) and the validated
    # compatible dequantizer (unpack_and_dequant_vcache).
    dequant = unpack_and_dequant_vcache(
        call["code"], call["scale"].unsqueeze(-1), call["mn"].unsqueeze(-1), call["group_size"], call["bit"]
    )
    recon = reconstruction_stats(spy_tensor.float(), dequant.float())
    recon_finite = all(torch.isfinite(torch.tensor(v)) for v in recon.values())

    code2, scale2, mn2 = real_quant(spy_tensor.contiguous(), call["group_size"], call["bit"])
    dequant2 = unpack_and_dequant_vcache(code2, scale2.unsqueeze(-1), mn2.unsqueeze(-1), call["group_size"], call["bit"])
    recon_deterministic = bool(torch.equal(dequant, dequant2))

    dist = distribution_stats(spy_tensor.float())
    dist_finite = all(torch.isfinite(torch.tensor(v)) for v in dist.values())
    dist2 = distribution_stats(spy_tensor.float())
    dist_deterministic = dist == dist2

    real_metrics = {
        "axis": axis,
        "model_class": model_class_name,
        "layer0_k_bits": layer0.k_bits,
        "layer0_v_bits": layer0.v_bits,
        "layer_policy_hash": policy_hash(resolved),
        "dataset": dataset,
        "prefix_len": prefix_len,
        "quantizer_call_shape": call["shape"],
        "quantizer_call_dtype": call["dtype"],
        "quantizer_call_device": call["device"],
        "quantizer_bit": call["bit"],
        "quantizer_group_size": call["group_size"],
        "nonfinite_logits": nonfinite_logits,
        "capture_parity": parity,
        "reconstruction_features": recon,
        "reconstruction_features_finite": bool(recon_finite),
        "reconstruction_features_deterministic": recon_deterministic,
        "distribution_features": dist,
        "distribution_features_finite": bool(dist_finite),
        "distribution_features_deterministic": bool(dist_deterministic),
    }

    del model, out, spy_calls, call, captured
    gc.collect()
    torch.cuda.empty_cache()

    return real_metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name_or_path", default="lmsys/longchat-7b-v1.5-32k")
    p.add_argument("--cache_dir", default="./cached_models")
    p.add_argument("--dataset", default="hotpotqa")
    p.add_argument("--prefix_len", type=int, default=256)
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--residual_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run_label", default=None)
    return p.parse_args(argv)


def main():
    args = parse_args()

    import torch  # deferred: keeps this module importable/inspectable without CUDA if ever needed

    if not torch.cuda.is_available():
        raise CanaryError("CUDA is not available; this canary requires a real GPU.")

    run_label = args.run_label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(args.output_root, run_label)
    os.makedirs(out_dir, exist_ok=True)

    boot_id_start = get_boot_id()
    gpu_before = gpu_snapshot()
    start_time = datetime.now(timezone.utc).astimezone().isoformat()

    summary = {
        "run_label": run_label,
        "start_time": start_time,
        "boot_id_start": boot_id_start,
        "git_commit": get_git_commit(),
        "git_status_short": get_git_status_short(),
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "triton_ptxas_path": os.environ.get("TRITON_PTXAS_PATH"),
        "args": vars(args),
        "gpu_before": gpu_before,
    }

    exit_code = 0
    try:
        summary["synthetic_triton_parity"] = run_synthetic_triton_parity(args.group_size)
        summary["matmul_cross_check"] = run_matmul_cross_check(args.group_size, bits=4)

        summary["real_key_probe"] = run_real_capture(
            "key", args.model_name_or_path, args.cache_dir, args.dataset, args.prefix_len, args.group_size, args.residual_length, args.seed
        )
        summary["real_value_probe"] = run_real_capture(
            "value", args.model_name_or_path, args.cache_dir, args.dataset, args.prefix_len, args.group_size, args.residual_length, args.seed
        )
    except Exception as e:  # noqa: BLE001 -- canary must record, not crash uncaught
        summary["error"] = f"{type(e).__name__}: {e}"
        exit_code = 1

    end_time = datetime.now(timezone.utc).astimezone().isoformat()
    boot_id_end = get_boot_id()
    gpu_after = gpu_snapshot()

    summary["end_time"] = end_time
    summary["boot_id_end"] = boot_id_end
    summary["boot_id_stable"] = boot_id_start == boot_id_end
    summary["gpu_after"] = gpu_after
    summary["exit_code"] = exit_code

    summary_path = os.path.join(out_dir, "g2_canary_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary written to: {summary_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
