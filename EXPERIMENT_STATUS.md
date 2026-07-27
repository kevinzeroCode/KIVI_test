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

**This baseline has not completed successfully yet.** Two environment
compatibility issues were found and fixed on the DGX Spark host (aarch64,
GB10, CUDA 13.0); see `environment/README.md` for full detail. Neither
required changing KIVI model, quantization, or CUDA kernel code, and neither
changed the LongBench task list or evaluation logic:

- `FAILED_CONFIG_COMPATIBILITY`: `transformers==4.43.1` (the `pyproject.toml`
  pin) raises `KeyError: 'rope_type'` when parsing
  `lmsys/longchat-7b-v1.5-32k`'s config, before any model weights are
  downloaded or any LongBench task starts. Cause: the model's `rope_scaling`
  uses the legacy `{"type": "linear", ...}` key, and 4.43.1's
  `rope_config_validation()` has a bug where the type-specific validator does
  a hard `rope_scaling["rope_type"]` lookup despite the outer function
  resolving `rope_type` via backward-compat. Fixed by pinning
  `transformers==4.43.4` instead (the version already documented above as
  the one actually validated for the completed KIVI-2 run). Not a memory or
  CUDA issue.
- `FAILED_DATASET_COMPATIBILITY`: with the RoPE issue fixed, the FP16 run
  progressed past config parsing, model download, and FP16 model loading,
  then failed loading the LongBench dataset itself:
  `RuntimeError: Dataset scripts are no longer supported, but found
  LongBench.py`. Cause: `datasets==5.0.0` (the default pip resolution)
  dropped support for Hub datasets that ship a legacy loading script, and
  `THUDM/LongBench` still ships `LongBench.py` as one. Fixed by pinning
  `datasets==3.6.0`. Verified with a dataset-only smoke test (no GPU, no
  model weights): `THUDM/LongBench` config `qasper`, split `test`, resolved
  revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`, 200 rows, all required
  columns present.

The FP16 baseline run has not yet been relaunched end-to-end with both fixes
in place.

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
