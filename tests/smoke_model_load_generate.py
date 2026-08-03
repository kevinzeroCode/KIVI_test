"""Phase 11: model-load + short generation smoke test for one (k_bits, v_bits)
config, run against the real lmsys/longchat-7b-v1.5-32k weights.

This intentionally does NOT invoke pred_long_bench.py's __main__ (which would
run the full 15-task / 3550-sample LongBench loop). Instead it calls the same
production functions -- validate_bits, loader_decision, build_model_and_tokenizer,
build_run_config, build_pred_dir_name, prepare_run_directory -- directly, with
a single short synthetic prompt and a small max_new_tokens, and writes only to
an isolated output root (default tmp/smoke_outputs/pred), never the real
pred/ directory used by the committed FP16/KIVI-2/KIVI-4 baselines.

Intended to be invoked once per config, as a fresh subprocess each time, so
GPU memory is fully released between configs (see run_smoke_tests.sh).

Usage:
    ./.venv/bin/python tests/smoke_model_load_generate.py --k_bits 2 --v_bits 16
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("WANDB_DISABLED", "true")

import torch

import pred_long_bench as plb


class _Args:
    pass


def build_prompt(tokenizer, model_name, target_tokens=180):
    sentence = (
        "The quick brown fox jumps over the lazy dog near the river bank every "
        "single morning before the sun rises over the hills. "
    )
    body = sentence
    while len(tokenizer(body, truncation=False).input_ids) < target_tokens:
        body += sentence
    prompt = f"Summarize the following text in one sentence:\n{body}"
    return plb.build_chat(tokenizer, prompt, model_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_bits", type=int, required=True)
    parser.add_argument("--v_bits", type=int, required=True)
    parser.add_argument("--model_name_or_path", default="lmsys/longchat-7b-v1.5-32k")
    parser.add_argument("--cache_dir", default="./cached_models")
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--residual_length", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=31500)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--output_root", default="tmp/smoke_outputs/pred")
    args = parser.parse_args()

    plb.seed_everything(42)
    plb.validate_bits(args.k_bits, args.v_bits)
    quantize_key, quantize_value, use_kivi_model = plb.loader_decision(args.k_bits, args.v_bits)

    model_args = _Args()
    model_args.model_name_or_path = args.model_name_or_path
    model_args.k_bits = args.k_bits
    model_args.v_bits = args.v_bits
    model_args.group_size = args.group_size
    model_args.residual_length = args.residual_length

    training_args = _Args()
    training_args.cache_dir = args.cache_dir

    dtype = torch.float16
    model, tokenizer, model_class_name = plb.build_model_and_tokenizer(
        model_args, training_args, dtype, use_kivi_model
    )
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0

    model_name = args.model_name_or_path.split("/")[-1]

    def route_label(quantize_side):
        if quantize_side:
            return "KIVI quantized"
        return "FP16 pass-through" if use_kivi_model else "FP16"

    run_config = plb.build_run_config(
        model_args, args.max_length, model_class_name, quantize_key, quantize_value, seed=42
    )
    pred_dir_name = plb.build_pred_dir_name(
        model_name, args.max_length, args.k_bits, args.v_bits, args.group_size, args.residual_length
    )
    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)
    pred_dir = os.path.join(args.output_root, pred_dir_name)
    resume_status = plb.prepare_run_directory(pred_dir, run_config)

    print(f"Model class: {model_class_name}")
    print(f"K bits: {args.k_bits}")
    print(f"V bits: {args.v_bits}")
    print(f"quantize_key: {quantize_key}")
    print(f"quantize_value: {quantize_value}")
    print(f"Key route: {route_label(quantize_key)}")
    print(f"Value route: {route_label(quantize_value)}")
    print(f"Output directory: {pred_dir}")
    print(f"Resume metadata status: {resume_status}")

    # Assertions on the loader/log correctness itself, before touching the GPU
    # any further -- these are the "log 的 model class 正確 / quantize_key/
    # quantize_value 正確" checks from the smoke-test plan.
    expected_class = "LlamaForCausalLM_KIVI" if use_kivi_model else "LlamaForCausalLM"
    assert model_class_name == expected_class, f"unexpected model class {model_class_name}"
    assert quantize_key == (args.k_bits < 16)
    assert quantize_value == (args.v_bits < 16)

    prompt = build_prompt(tokenizer, model_name)
    inputs = tokenizer(prompt, truncation=False, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    context_length = inputs["input_ids"].shape[-1]
    print(f"Prompt tokens: {context_length}")

    with torch.no_grad():
        result = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            output_scores=True,
            return_dict_in_generate=True,
        )
    output = result.sequences[0]
    scores = result.scores
    nan_found = any(torch.isnan(s).any().item() for s in scores)
    inf_found = any(torch.isinf(s).any().item() for s in scores)

    pred_text = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    new_tokens = output.shape[-1] - context_length
    print(f"Generated new tokens: {new_tokens}")
    print(f"Generated text repr: {pred_text!r}")
    print(f"NaN in scores: {nan_found}")
    print(f"Inf in scores: {inf_found}")

    assert not nan_found, "NaN detected in generation scores"
    assert not inf_found, "Inf detected in generation scores"
    assert new_tokens > 0, "no new tokens were generated"
    assert len(pred_text.strip()) > 0, "generated text is empty"

    out_path = os.path.join(pred_dir, "smoke_prompt.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        row = {
            "pred": pred_text,
            "context_length": context_length,
            "new_tokens": new_tokens,
            "k_bits": args.k_bits,
            "v_bits": args.v_bits,
        }
        json.dump(row, f, ensure_ascii=False)
        f.write("\n")

    written_config = plb.load_run_config(pred_dir)
    assert written_config is not None, "run_config.json was not written"
    assert written_config["k_bits"] == args.k_bits
    assert written_config["v_bits"] == args.v_bits
    assert written_config["model_class"] == model_class_name
    assert written_config["quantize_key"] == quantize_key
    assert written_config["quantize_value"] == quantize_value

    print("SMOKE_TEST_PASS")


if __name__ == "__main__":
    main()
