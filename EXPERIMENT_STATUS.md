# KIVI Experiment Status

## Repository Base

- Upstream repository: `https://github.com/jy-yuan/KIVI.git`
- Upstream KIVI base commit: `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`
- Current branch: `main`
- Created: `2026-07-26` (`Asia/Taipei`)
- Main modified experiment files:
  - `models/llama_kivi.py`
  - `pred_long_bench.py`
  - `scripts/long_test.sh`
  - `scripts/report_longbench.py`

## Completed Baseline

- Model: `lmsys/longchat-7b-v1.5-32k`
- Attention: MHA (32 query heads and 32 key/value heads)
- Method: KIVI-2
- K bits: `2`
- V bits: `2`
- Group size: `32`
- Residual length: `128`
- Context limit: `31,500`
- LongBench tasks: `15`
- Samples: `3,550`
- Seed: `42`
- Result average: `38.03`
- Reference KIVI paper result: `38.30`

The complete prediction JSONL files, logs, cached datasets, and model weights are
local experimental artifacts and are intentionally not committed to GitHub.

## Implementation Notes

- Keys use per-channel quantization across the token dimension.
- Values use per-token quantization across the head dimension.
- The newest residual tokens remain in FP16 for this experiment.
- Older key/value cache entries use packed low-bit integer storage.
- Decode uses the custom CUDA GEMV extension (`kivi_gemv`).
- A mixed K/V configuration currently falls back to standard Hugging Face
  attention whenever either K or V is configured as 16-bit.
- Prediction directory names include K bits but not V bits, so mixed K/V runs
  with the same K-bit value are not currently isolated from one another.
- Per-layer route maps are not implemented.
- Rotation-KIVI and Polar are not implemented.

## Current Gates

- Gate A - Implementation Correctness: **PARTIAL**. The packed KIVI path and
  custom CUDA GEMV are present and the KIVI-2 parameters are wired into the
  model config, but mixed K/V and per-layer routes are unsupported.
- Gate B - Quality Reproduction: **PARTIAL**. KIVI-2 completed all 15 tasks at
  `38.03`, close to the paper's `38.30`, but a local FP16 comparison has not
  yet been run.
- Gate C - System Reproduction: **PARTIAL**. Packed low-bit storage is
  implemented, but there is no controlled FP16 comparison for peak memory,
  bytes per token, or throughput.

## Next Experiment on DGX Spark

Run the FP16 LongBench baseline with the same model, 15 tasks, context limit,
seed, and evaluation code. The paper reference average is `38.72`; this is an
expectation, not a measured local FP16 result. No memory or throughput result is
claimed here.

Expected prediction path:

`pred/longchat-7b-v1.5-32k_31500_16bits_group32_residual128/`

## Reproduction Command

The following FP16 commands are planned and have not been executed:

```bash
mkdir -p logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/long_test.sh 0 16 16 32 128 lmsys/longchat-7b-v1.5-32k \
  > logs/longbench_fp16.log 2>&1
```

Evaluation:

```bash
./.venv/bin/python eval_long_bench.py \
  --model longchat-7b-v1.5-32k_31500_16bits_group32_residual128
```

Expected evaluation output:

`pred/longchat-7b-v1.5-32k_31500_16bits_group32_residual128/result.json`
