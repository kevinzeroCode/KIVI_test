"""Phase 12: five-sample functional test for one (k_bits, v_bits) config.

Runs the real generation pipeline (pred_long_bench.get_pred) over the first
5 rows of the real "qasper" LongBench split -- not a synthetic prompt, but
also nowhere near the full 200-row task, let alone the full 15-task /
3550-sample benchmark. Writes only to an isolated output root (default
tmp/functional_outputs/pred), then runs eval_long_bench.py against that same
isolated root to confirm scoring can read a mixed-bit-named directory.

This only checks that the pipeline is wired correctly end to end:
  - generation completes for all 5 rows
  - the JSONL is parseable with exactly 5 rows
  - run_config.json matches the requested config
  - eval_long_bench.py can read the (possibly mixed-bit) directory and
    produces a result.json with a numeric "qasper" score
  - the mixed-bit configs (K2/V16, K16/V2) did not silently fall back to
    plain FP16 -- checked via the same route labels used elsewhere, not by
    the resulting score, since 5 samples carry no statistical meaning.

The qasper score itself is recorded for visibility only; it is NOT a
pass/fail criterion here (5 samples is far too small to be meaningful, and
mixed-bit scores are not asserted to fall between FP16 and KIVI-2/4).

Usage:
    ./.venv/bin/python tests/functional_five_sample.py --k_bits 2 --v_bits 16
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("WANDB_DISABLED", "true")

import torch
from datasets import load_dataset

import pred_long_bench as plb


class _Args:
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_bits", type=int, required=True)
    parser.add_argument("--v_bits", type=int, required=True)
    parser.add_argument("--model_name_or_path", default="lmsys/longchat-7b-v1.5-32k")
    parser.add_argument("--cache_dir", default="./cached_models")
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--residual_length", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=31500)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--dataset", default="qasper")
    parser.add_argument("--output_root", default="tmp/functional_outputs/pred")
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
    print(f"Key route: {route_label(quantize_key)}")
    print(f"Value route: {route_label(quantize_value)}")
    print(f"Output directory: {pred_dir}")
    print(f"Resume metadata status: {resume_status}")

    expected_class = "LlamaForCausalLM_KIVI" if use_kivi_model else "LlamaForCausalLM"
    assert model_class_name == expected_class
    assert quantize_key == (args.k_bits < 16)
    assert quantize_value == (args.v_bits < 16)

    full_data = load_dataset('THUDM/LongBench', args.dataset, split='test', trust_remote_code=True)
    data = full_data.select(range(args.num_samples))
    print(f"Rows selected: {len(data)} (of {len(full_data)} total in '{args.dataset}')")

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    prompt_format = dataset2prompt[args.dataset]
    max_gen = dataset2maxlen[args.dataset]

    out_path = os.path.join(pred_dir, f"{args.dataset}.jsonl")
    device = next(model.parameters()).device
    plb.get_pred(model, tokenizer, data, args.max_length, max_gen, prompt_format, args.dataset, device, model_name, out_path, start_idx=0)

    # --- structural checks -------------------------------------------------
    rows = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    assert len(rows) == args.num_samples, f"expected {args.num_samples} rows, got {len(rows)}"
    for row in rows:
        assert "pred" in row and "answers" in row and "all_classes" in row and "length" in row
    print(f"JSONL rows written and parsed: {len(rows)}")

    written_config = plb.load_run_config(pred_dir)
    assert written_config is not None
    assert written_config["k_bits"] == args.k_bits
    assert written_config["v_bits"] == args.v_bits
    assert written_config["model_class"] == model_class_name
    print("run_config.json matches requested config: True")

    # --- evaluation can read this (possibly mixed-bit) directory ----------
    env = dict(os.environ)
    env["KIVI_PRED_ROOT"] = args.output_root
    result = subprocess.run(
        [sys.executable, "eval_long_bench.py", "--model", pred_dir_name],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env,
        capture_output=True,
        text=True,
    )
    print("eval_long_bench.py exit code:", result.returncode)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, "eval_long_bench.py failed on the mixed-bit directory"

    result_json_path = os.path.join(pred_dir, "result.json")
    assert os.path.exists(result_json_path), "result.json was not produced"
    with open(result_json_path) as f:
        scores = json.load(f)
    assert args.dataset in scores and isinstance(scores[args.dataset], (int, float)), "qasper score missing/non-numeric"
    print(f"{args.dataset} score (5 samples, informational only, not a pass/fail gate): {scores[args.dataset]}")

    print("FUNCTIONAL_TEST_PASS")


if __name__ == "__main__":
    main()
