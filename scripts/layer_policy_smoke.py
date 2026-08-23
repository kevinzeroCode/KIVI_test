"""Stage C smoke test for the layer-wise KV sensitivity project.

Proves end-to-end on GPU that a layer-wise policy survives:
    policy JSON -> config resolution -> model construction
    -> per-layer attention attributes -> actual forward/generate

Reuses the project's real model construction (pred_long_bench.py's
build_model_and_tokenizer) and the same tokenizer/prompt-formatting/
generation call shape as pred_long_bench.get_pred() -- this is NOT a fake
attention-only simulation.

Does NOT invoke pred_long_bench.py's __main__ (see
docs/layer_policy_stage_c_smoke_plan.md for why a bare --e there would be
wrong and unsafe). Writes exclusively under outputs/layer_policy_smoke/;
never touches pred/, never touches the resume system.

Run with (see docs/layer_policy_stage_c_smoke_plan.md for the exact
commands used in this round):
    CUDA_VISIBLE_DEVICES=0 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \\
    ./.venv/bin/python scripts/layer_policy_smoke.py [--layer_policy PATH | none] ...
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

EXPECTED_NUM_LAYERS = 32


class SmokeError(RuntimeError):
    """Any smoke-test precondition/postcondition failure. Fails closed."""


class SmokeModelArgs:
    """Minimal stand-in for utils.process_args.ModelArguments -- only the
    fields build_model_and_tokenizer() actually reads."""

    def __init__(self, model_name_or_path, k_bits, v_bits, group_size, residual_length):
        self.model_name_or_path = model_name_or_path
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.residual_length = residual_length


class SmokeTrainingArgs:
    """Minimal stand-in for utils.process_args.TrainingArguments -- only
    .cache_dir is read by build_model_and_tokenizer()."""

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir


def get_boot_id():
    try:
        with open("/proc/sys/kernel/random/boot_id", "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def get_git_status_short():
    try:
        return subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode()
    except Exception:
        return None


def build_resolved_layer_records(decoder_layers):
    """Pure extraction: read k_bits/v_bits/quantize_key/quantize_value
    directly off each constructed LlamaAttention_KIVI instance (never trust
    the JSON policy file alone -- this inspects the real model)."""
    records = []
    for layer in decoder_layers:
        attn = layer.self_attn
        records.append(
            {
                "layer_idx": attn.layer_idx,
                "k_bits": attn.k_bits,
                "v_bits": attn.v_bits,
                "quantize_key": attn.quantize_key,
                "quantize_value": attn.quantize_value,
            }
        )
    records.sort(key=lambda r: r["layer_idx"])
    return records


def validate_layer_count(records, expected=EXPECTED_NUM_LAYERS):
    if len(records) != expected:
        raise SmokeError(f"Expected exactly {expected} decoder layers, found {len(records)}")


def validate_layer17_k2_v16_policy(records):
    """Fails closed unless layer 17 is exactly K2/V16 and every other layer
    is exactly K16/V16, per the Stage-C required policy."""
    validate_layer_count(records)
    by_idx = {r["layer_idx"]: r for r in records}
    problems = []

    want17 = {"k_bits": 2, "v_bits": 16, "quantize_key": True, "quantize_value": False}
    got17 = {k: by_idx[17][k] for k in want17}
    if got17 != want17:
        problems.append(f"layer 17: expected {want17}, got {got17}")

    want_default = {"k_bits": 16, "v_bits": 16, "quantize_key": False, "quantize_value": False}
    for idx, rec in by_idx.items():
        if idx == 17:
            continue
        got = {k: rec[k] for k in want_default}
        if got != want_default:
            problems.append(f"layer {idx}: expected {want_default}, got {got}")

    if problems:
        raise SmokeError("Layer-17 K2/V16 policy validation failed:\n" + "\n".join(problems))


def validate_records_match_resolved_policy(records, resolved_layer_policy):
    """Generic, policy-agnostic check: the live constructed model's per-layer
    attributes must exactly reproduce what utils.layer_policy resolved from
    the JSON policy file -- for ANY policy (Key-axis, Value-axis, or a
    mixed/multi-layer one), not just the specific layer17-K2/V16 pattern.
    This is what actually proves "policy JSON -> ... -> real attention
    instances" end to end; validate_layer17_k2_v16_policy below is a
    narrower, Key-probe-specific regression check kept for that one case."""
    validate_layer_count(records)
    by_idx = {r["layer_idx"]: r for r in records}
    problems = []
    for i, entry in enumerate(resolved_layer_policy.layers):
        want = {
            "k_bits": entry.k_bits,
            "v_bits": entry.v_bits,
            "quantize_key": entry.k_bits < 16,
            "quantize_value": entry.v_bits < 16,
        }
        got = {k: by_idx[i][k] for k in want}
        if got != want:
            problems.append(f"layer {i}: expected {want}, got {got}")
    if problems:
        raise SmokeError("Resolved-policy validation failed:\n" + "\n".join(problems))


def validate_all_global_k16_v16(records):
    """Baseline-control validation: every layer must be K16/V16 with both
    quantize flags False."""
    validate_layer_count(records)
    want = {"k_bits": 16, "v_bits": 16, "quantize_key": False, "quantize_value": False}
    problems = []
    for rec in records:
        got = {k: rec[k] for k in want}
        if got != want:
            problems.append(f"layer {rec['layer_idx']}: expected {want}, got {got}")
    if problems:
        raise SmokeError("Global K16/V16 control validation failed:\n" + "\n".join(problems))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name_or_path", default="lmsys/longchat-7b-v1.5-32k")
    p.add_argument("--cache_dir", default="./cached_models")
    p.add_argument(
        "--layer_policy",
        default="analysis/policies/key_probe_layer17_k2.json",
        help="Path to a layer policy JSON, or the literal string 'none' for the global K16/V16 control run.",
    )
    p.add_argument("--k_bits", type=int, default=16, help="Global fallback k_bits when --layer_policy is 'none'.")
    p.add_argument("--v_bits", type=int, default=16, help="Global fallback v_bits when --layer_policy is 'none'.")
    p.add_argument("--dataset", default="hotpotqa")
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--max_length", type=int, default=31500)
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--residual_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_root", default="outputs/layer_policy_smoke")
    p.add_argument("--run_label", default=None, help="Defaults to a UTC timestamp.")
    return p.parse_args(argv)


def main():
    args = parse_args()

    # Heavy imports deferred past argparse so --help stays fast/CPU-only.
    import torch
    from datasets import load_dataset

    import pred_long_bench as plb
    from utils.generation_semantics import NO_BUILD_CHAT_DATASETS, resolve_generate_kwargs
    from utils.jsonl_integrity import inspect_jsonl
    from utils.layer_policy import canonical_policy_dict, load_and_resolve, policy_hash

    plb.seed_everything(args.seed)

    run_label = args.run_label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(REPO_ROOT, args.output_root, run_label)
    os.makedirs(out_dir, exist_ok=True)

    boot_id_start = get_boot_id()
    start_time = datetime.now(timezone.utc).astimezone().isoformat()

    use_layer_policy = bool(args.layer_policy) and args.layer_policy.strip().lower() != "none"
    resolved_layer_policy = None
    if use_layer_policy:
        probe_config = plb.LlamaConfig.from_pretrained(args.model_name_or_path)
        resolved_layer_policy = load_and_resolve(
            args.layer_policy, probe_config.num_hidden_layers, args.k_bits, args.v_bits
        )

    model_args = SmokeModelArgs(args.model_name_or_path, args.k_bits, args.v_bits, args.group_size, args.residual_length)
    training_args = SmokeTrainingArgs(args.cache_dir)
    dtype = torch.float16

    # Smoke harness always constructs LlamaForCausalLM_KIVI (use_kivi_model=True),
    # even for the global-K16/V16 control -- unlike production's
    # loader_decision(), which would pick the plain (non-KIVI) HF model for
    # 16/16 and therefore have no k_bits/v_bits/quantize_key/quantize_value
    # attributes to inspect at all. This is a deliberate smoke-harness-only
    # choice so both runs are directly comparable at the per-layer-attribute
    # level; it does not claim production's global-FP16 pred/ directory was
    # generated this way.
    model, tokenizer, model_class_name = plb.build_model_and_tokenizer(
        model_args, training_args, dtype, use_kivi_model=True, resolved_layer_policy=resolved_layer_policy
    )
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0

    decoder_layers = model.model.layers
    records = build_resolved_layer_records(decoder_layers)
    if use_layer_policy:
        # Generic: proves the constructed model matches whatever policy was
        # actually loaded (Key-axis, Value-axis, or otherwise), not just the
        # original Key-probe pattern.
        validate_records_match_resolved_policy(records, resolved_layer_policy)
    else:
        validate_all_global_k16_v16(records)

    resolved_summary = {
        "model_class": model_class_name,
        "num_layers": len(records),
        "layers": records,
        "layer_policy_path": args.layer_policy if use_layer_policy else None,
        "layer_policy_name": resolved_layer_policy.policy_name if resolved_layer_policy else "global",
        "layer_policy_hash": policy_hash(resolved_layer_policy) if resolved_layer_policy else None,
        "global_k_bits": args.k_bits,
        "global_v_bits": args.v_bits,
    }
    with open(os.path.join(out_dir, "resolved_model_policy.json"), "w", encoding="utf-8") as f:
        json.dump(resolved_summary, f, indent=2)

    # Section 6: smoke-only, non-invasive route evidence. Patches the two
    # kernel entry points imported into models.llama_kivi's namespace (the
    # exact pattern already used by tests/test_mixed_kv_kernel_route.py's
    # spy_quant/spy_bmm) for the duration of this process only -- never
    # touches production forward() code. Since only layer 17 has k_bits<16
    # in the layer-17 smoke run (all other layers/axes are 16), any call
    # observed with bit==2 is attributable to layer 17's Key path.
    import models.llama_kivi as llama_kivi

    route_evidence = {"quant_calls_by_bit": {}, "bmm_calls_by_bit": {}}
    real_quant = llama_kivi.triton_quantize_and_pack_along_last_dim
    real_bmm = llama_kivi.cuda_bmm_fA_qB_outer

    def spy_quant(data, group_size, bit):
        route_evidence["quant_calls_by_bit"][bit] = route_evidence["quant_calls_by_bit"].get(bit, 0) + 1
        return real_quant(data, group_size, bit)

    def spy_bmm(group_size, query, key_or_val, scale, mn, bits):
        route_evidence["bmm_calls_by_bit"][bits] = route_evidence["bmm_calls_by_bit"].get(bits, 0) + 1
        return real_bmm(group_size, query, key_or_val, scale, mn, bits)

    # Section 4 (NaN/Inf check): a smoke-only forward hook on the real
    # lm_head module (not a production code change) records whether any
    # logits tensor produced during generation contains NaN/Inf.
    nan_inf_flags = {"seen_nonfinite": False}

    def lm_head_hook(module, inp, output):
        if not torch.isfinite(output).all():
            nan_inf_flags["seen_nonfinite"] = True

    hook_handle = model.lm_head.register_forward_hook(lm_head_hook)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset("THUDM/LongBench", args.dataset, split="test", trust_remote_code=True)
    dataset2prompt = json.load(open(os.path.join(REPO_ROOT, "config/dataset2prompt.json"), "r"))
    dataset2maxlen = json.load(open(os.path.join(REPO_ROOT, "config/dataset2maxlen.json"), "r"))
    prompt_format = dataset2prompt[args.dataset]
    max_gen = dataset2maxlen[args.dataset]
    model_short_name = args.model_name_or_path.split("/")[-1]

    n = min(args.num_samples, len(data))
    out_jsonl_path = os.path.join(out_dir, f"{args.dataset}.smoke.jsonl")
    generation_error = None
    n_completed = 0

    with mock.patch.object(llama_kivi, "triton_quantize_and_pack_along_last_dim", spy_quant), \
         mock.patch.object(llama_kivi, "cuda_bmm_fA_qB_outer", spy_bmm), \
         open(out_jsonl_path, "w", encoding="utf-8") as f:
        for idx in range(n):
            try:
                json_obj = data[idx]
                prompt = prompt_format.format(**json_obj)
                tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
                if len(tokenized_prompt) > args.max_length:
                    half = int(args.max_length / 2)
                    prompt = (
                        tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
                        + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
                    )
                if args.dataset not in NO_BUILD_CHAT_DATASETS:
                    prompt = plb.build_chat(tokenizer, prompt, model_short_name)
                inp = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
                context_length = inp.input_ids.shape[-1]
                generate_kwargs = resolve_generate_kwargs(args.dataset, tokenizer, context_length, max_gen)
                with torch.no_grad():
                    output = model.generate(**inp, **generate_kwargs)[0]
                pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
                pred = plb.post_process(pred, model_short_name)
                row = {"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
                n_completed += 1
                del inp, output, tokenized_prompt
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001 -- smoke test must record, not raise mid-loop
                generation_error = f"sample {idx}: {type(e).__name__}: {e}"
                break

    hook_handle.remove()

    end_time = datetime.now(timezone.utc).astimezone().isoformat()
    boot_id_end = get_boot_id()

    integrity = inspect_jsonl(out_jsonl_path)
    exit_status = "OK" if (generation_error is None and n_completed == n and integrity.invalid_rows == 0) else "ERROR"

    manifest = {
        "run_label": run_label,
        "start_time": start_time,
        "end_time": end_time,
        "boot_id_start": boot_id_start,
        "boot_id_end": boot_id_end,
        "boot_id_stable": boot_id_start == boot_id_end,
        "git_commit": get_git_commit(),
        "git_status_short": get_git_status_short(),
        "model_class": model_class_name,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dataset": args.dataset,
        "num_samples_requested": args.num_samples,
        "num_samples_used": n,
        "num_samples_completed": n_completed,
        "generation_error": generation_error,
        "nonfinite_logits_seen": nan_inf_flags["seen_nonfinite"],
        "route_evidence": route_evidence,
        "output_jsonl": out_jsonl_path,
        "jsonl_integrity": {
            "valid_rows": integrity.valid_rows,
            "invalid_rows": integrity.invalid_rows,
            "ends_with_newline": integrity.ends_with_newline,
        },
        "layer_policy_hash": resolved_summary["layer_policy_hash"],
        "layer_policy_name": resolved_summary["layer_policy_name"],
        "exit_status": exit_status,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(f"\nresolved_model_policy.json and manifest.json written to: {out_dir}")

    if exit_status != "OK":
        sys.exit(1)


if __name__ == "__main__":
    main()
