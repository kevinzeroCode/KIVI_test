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

## Completed Baseline (FP16, DGX Spark)

- Model: `lmsys/longchat-7b-v1.5-32k`
- Method: FP16 KV cache (no KIVI quantization; standard HuggingFace attention
  with flash-attention-2)
- K bits: `16`, V bits: `16`
- Group size: `32`, residual length: `128` (naming/interface parameters only
  under the FP16 path; not used for any quantization math)
- Context limit: `31,500`
- LongBench tasks: `15`
- Samples: `3,550` (validated: 15/15 task files, 0 `.partial`, 0 parse
  errors, 3550/3550 total rows)
- Seed: `42`
- Host: DGX Spark (aarch64, NVIDIA GB10, CUDA 13.0 driver)
- Prediction path:
  `pred/longchat-7b-v1.5-32k_31500_16bits_group32_residual128/`
- Evaluation log: `logs/longbench_fp16_eval_20260729_000100.log`
- Result path:
  `pred/longchat-7b-v1.5-32k_31500_16bits_group32_residual128/result.json`

Per-task scores:

| task | score |
|---|---|
| triviaqa | 83.99 |
| narrativeqa | 20.81 |
| passage_retrieval_en | 30.50 |
| gov_report | 30.81 |
| qasper | 29.37 |
| repobench-p | 56.79 |
| trec | 66.50 |
| multifieldqa_en | 43.42 |
| qmsum | 22.73 |
| musique | 14.71 |
| lcc | 52.99 |
| samsum | 40.80 |
| hotpotqa | 33.05 |
| multi_news | 26.62 |
| 2wikimqa | 24.45 |

- **Result average: `38.50` (38.50266666666666)**
- Reference KIVI paper FP16 result: `38.72` (delta: `-0.22`)
- Local KIVI-2 result (above): `38.03` (delta: `+0.47`)
- No task scored 0, NaN, or was missing; no task's score looked inconsistent
  with typical LongBench per-task distributions (multi-hop QA / summarization
  tasks scoring lower than retrieval/classification tasks, e.g. `trec`,
  `triviaqa`, is expected and matches published LongBench baselines).
- No per-task breakdown exists locally for the completed KIVI-2 run (only its
  overall average, `38.03` above, was retained) to do a task-by-task
  comparison; predictions/results for that run were not committed per the
  policy noted below.

The complete prediction JSONL files, `result.json`, cached datasets, model
weights, and full run logs are local experimental artifacts and are
intentionally not committed to GitHub.

## Current Gates

- Gate A - Implementation Correctness: **PARTIAL**. The packed KIVI path and
  custom CUDA GEMV are present and the KIVI-2 parameters are wired into the
  model config, but mixed K/V and per-layer routes are unsupported.
- Gate B - Quality Reproduction: **PARTIAL, both KIVI-2 and FP16 measured**.
  KIVI-2 completed all 15 tasks at `38.03` (paper: `38.30`). FP16 completed
  all 15 tasks at `38.50` (paper: `38.72`). Both local results are within
  ~0.3-0.5 of their respective paper references.
- Gate C - System Reproduction: **PARTIAL**. Packed low-bit storage is
  implemented, but there is no controlled FP16 vs. KIVI-2 comparison for peak
  memory, bytes per token, or throughput on the same host.

## DGX Spark Reproduction Notes

The FP16 baseline above required two environment compatibility fixes on this
host (aarch64, GB10, CUDA 13.0) plus one evaluation-dependency fix. See
`environment/README.md` for full detail. None of these required changing
KIVI model, quantization, or CUDA kernel code, and none changed the
LongBench task list or evaluation logic:

- `FAILED_CONFIG_COMPATIBILITY` (fixed): `transformers==4.43.1` (the
  `pyproject.toml` pin) raises `KeyError: 'rope_type'` when parsing
  `lmsys/longchat-7b-v1.5-32k`'s config, before any model weights are
  downloaded or any LongBench task starts. Cause: the model's `rope_scaling`
  uses the legacy `{"type": "linear", ...}` key, and 4.43.1's
  `rope_config_validation()` has a bug where the type-specific validator does
  a hard `rope_scaling["rope_type"]` lookup despite the outer function
  resolving `rope_type` via backward-compat. Fixed by pinning
  `transformers==4.43.4` instead (the version already documented above as
  the one actually validated for the completed KIVI-2 run).
- `FAILED_DATASET_COMPATIBILITY` (fixed): with the RoPE issue fixed, the FP16
  run progressed past config parsing, model download, and FP16 model
  loading, then failed loading the LongBench dataset itself:
  `RuntimeError: Dataset scripts are no longer supported, but found
  LongBench.py`. Cause: `datasets==5.0.0` (the default pip resolution)
  dropped support for Hub datasets that ship a legacy loading script, and
  `THUDM/LongBench` still ships `LongBench.py` as one. Fixed by pinning
  `datasets==3.6.0`. Verified with a dataset-only smoke test (no GPU, no
  model weights): `THUDM/LongBench` config `qasper`, split `test`, resolved
  revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`, 200 rows, all required
  columns present.
- Evaluation dependency gap (fixed): `eval_long_bench.py`'s `metrics.py`
  imports `jieba`, `fuzzywuzzy`, and `rouge`, none of which were installed by
  the `pyproject.toml`-based install (they only appear in the older
  `requirements.txt`). Installed at the `requirements.txt`-pinned versions:
  `jieba==0.42.1`, `fuzzywuzzy==0.18.0`, `rouge==1.0.1`. This does not affect
  model output in any way; it only affects scoring the already-generated
  predictions.

All version pins above are recorded in `environment/spark-baseline-requirements.txt`.

## Reproduction Command

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

Evaluation output:

`pred/longchat-7b-v1.5-32k_31500_16bits_group32_residual128/result.json`
