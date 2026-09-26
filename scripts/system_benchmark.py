"""Gate C system measurement: KV-cache bytes, peak GPU memory, prefill
latency and decode throughput on this host, per KV configuration.

Model loading goes through pred_long_bench.build_model_and_tokenizer (the
exact production loader); FP16 is the plain HF model with FlashAttention-2,
every other config is LlamaForCausalLM_KIVI. lm_head stays on GPU for every
config (KIVI_OFFLOAD_LM_HEAD=0) so all configs do identical work outside
attention/KV cache.

Greedy decode is a manual loop over model.model + lm_head on the last
position only, so the full-sequence logits tensor never inflates peak
memory. Inputs are fixed-seed random token ids (latency/memory do not
depend on content). Results append to a JSONL, one row per
(config, context_len); finished rows are skipped on rerun.

Usage:
  ./.venv/bin/python scripts/system_benchmark.py [--configs fp16,k2v16,...]
      [--contexts 4096,16384,31500] [--new-tokens 128] [--repeats 3]
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KIVI_OFFLOAD_LM_HEAD", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import pred_long_bench as plb  # noqa: E402
from utils.layer_policy import resolve_layer_policy  # noqa: E402

MODEL = "lmsys/longchat-7b-v1.5-32k"
GROUP_SIZE = 32
RESIDUAL_LENGTH = 128

# name -> (k_bits, v_bits, family)
CONFIGS = {
    "fp16": (16, 16, "kivi"),
    "k2v16": (2, 16, "kivi"),
    "k16v2": (16, 2, "kivi"),
    "k2v2": (2, 2, "kivi"),
    "k4v4": (4, 4, "kivi"),
    "rot_k2v16": (2, 16, "rotation_kivi"),
}


def load(name):
    k_bits, v_bits, family = CONFIGS[name]
    model_args = SimpleNamespace(
        model_name_or_path=MODEL, k_bits=k_bits, v_bits=v_bits,
        group_size=GROUP_SIZE, residual_length=RESIDUAL_LENGTH,
    )
    training_args = SimpleNamespace(cache_dir=str(REPO_ROOT / "cached_models"))
    use_kivi = not (k_bits == 16 and v_bits == 16)
    resolved = None
    if family != "kivi":
        num_layers = plb.LlamaConfig.from_pretrained(MODEL).num_hidden_layers
        resolved = resolve_layer_policy(num_layers, k_bits, v_bits, {
            "policy_name": f"sysbench_{name}",
            "default": {"k_bits": k_bits, "v_bits": v_bits, "family": family},
            "overrides": {},
        })
    model, _tok, cls = plb.build_model_and_tokenizer(
        model_args, training_args, torch.float16, use_kivi, resolved_layer_policy=resolved
    )
    model.eval()
    return model, cls


def cache_bytes(pkv):
    """Sum nbytes of every tensor held by the KV cache (KIVI tuple-of-tuples
    or HF DynamicCache)."""
    if hasattr(pkv, "key_cache"):
        return sum(t.nbytes for t in pkv.key_cache) + sum(t.nbytes for t in pkv.value_cache)
    total = 0
    stack = [pkv]
    while stack:
        x = stack.pop()
        if isinstance(x, torch.Tensor):
            total += x.nbytes
        elif isinstance(x, (tuple, list)):
            stack.extend(x)
    return total


@torch.no_grad()
def run_once(model, input_ids, new_tokens):
    base, head = model.model, model.lm_head
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = base(input_ids=input_ids, use_cache=True)
    nxt = head(out.last_hidden_state[:, -1:, :]).argmax(-1)
    pkv = out.past_key_values
    del out
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(new_tokens - 1):
        out = base(input_ids=nxt, past_key_values=pkv, use_cache=True)
        nxt = head(out.last_hidden_state[:, -1:, :]).argmax(-1)
        pkv = out.past_key_values
        del out
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    kv = cache_bytes(pkv)
    del pkv
    return t1 - t0, t2 - t1, kv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--contexts", default="4096,16384,31500")
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "system_benchmark" / "results.jsonl"))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["config"], r["context_len"]))

    contexts = [int(c) for c in args.contexts.split(",")]
    gen = torch.Generator().manual_seed(42)
    inputs = {c: torch.randint(100, 32000, (1, c), generator=gen) for c in contexts}

    for name in args.configs.split(","):
        todo = [c for c in contexts if (name, c) not in done]
        if not todo:
            print(f"[{name}] all contexts done, skipping")
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model, cls = load(name)
        torch.cuda.synchronize()
        weights_bytes = torch.cuda.memory_allocated()
        print(f"[{name}] loaded {cls}, allocated after load {weights_bytes / 2**30:.2f} GiB")
        run_once(model, inputs[todo[0]][:, :512].cuda(), 8)  # warmup: kernels/autotune

        for ctx in todo:
            ids = inputs[ctx].cuda()
            prefill, decode, kv = [], [], None
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(args.repeats):
                p, d, kv = run_once(model, ids, args.new_tokens)
                prefill.append(p)
                decode.append(d)
            peak = torch.cuda.max_memory_allocated()
            k_bits, v_bits, family = CONFIGS[name]
            row = {
                "config": name, "k_bits": k_bits, "v_bits": v_bits, "family": family,
                "model_class": cls, "context_len": ctx, "new_tokens": args.new_tokens,
                "repeats": args.repeats,
                "weights_gib": weights_bytes / 2**30,
                "kv_cache_mib": kv / 2**20,
                "peak_alloc_gib": peak / 2**30,
                "peak_minus_weights_gib": (peak - weights_bytes) / 2**30,
                "prefill_s_median": sorted(prefill)[len(prefill) // 2],
                "decode_tok_per_s_median": (args.new_tokens - 1) / sorted(decode)[len(decode) // 2],
                "prefill_s_all": prefill, "decode_s_all": decode,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "git_commit": plb.get_git_commit() if hasattr(plb, "get_git_commit") else None,
            }
            with out_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{name}] ctx={ctx} kv={row['kv_cache_mib']:.0f}MiB peak={row['peak_alloc_gib']:.2f}GiB "
                  f"prefill={row['prefill_s_median']:.2f}s decode={row['decode_tok_per_s_median']:.2f}tok/s")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
