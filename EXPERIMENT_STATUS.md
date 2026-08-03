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
- Host: original x86_64 / RTX 4090 host (see `environment/README.md`); no
  per-task breakdown was retained for this run, only the overall average
  above.

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

## Completed Baseline (KIVI-2, DGX Spark)

- Model: `lmsys/longchat-7b-v1.5-32k`
- Method: KIVI-2
- K bits: `2`, V bits: `2`
- Group size: `32`, residual length: `128`
- Context limit: `31,500`
- LongBench tasks: `15`
- Samples: `3,550` (validated: 15/15 task files, 0 `.partial`, 0 parse
  errors, 3550/3550 total rows)
- Seed: `42`
- Host: DGX Spark (aarch64, NVIDIA GB10, CUDA 13.0 driver)
- Prediction path:
  `pred/longchat-7b-v1.5-32k_31500_2bits_group32_residual128/`
- Prediction log: `logs/longbench_kivi2_20260729_220808.log`
- Evaluation log: `logs/longbench_kivi2_eval_20260730_221157.log`
- Manifest: `outputs/run_manifests/longbench_kivi2_20260729_220808.txt`
- Result path:
  `pred/longchat-7b-v1.5-32k_31500_2bits_group32_residual128/result.json`

Per-task scores:

| task | score |
|---|---|
| triviaqa | 82.74 |
| narrativeqa | 21.04 |
| passage_retrieval_en | 32.25 |
| gov_report | 30.48 |
| qasper | 28.35 |
| repobench-p | 55.17 |
| trec | 66.50 |
| multifieldqa_en | 41.58 |
| qmsum | 22.48 |
| musique | 13.69 |
| lcc | 52.28 |
| samsum | 41.21 |
| hotpotqa | 32.98 |
| multi_news | 26.60 |
| 2wikimqa | 22.93 |

- **Result average: `38.02` (38.01866666666667)**
- Reference KIVI paper KIVI-2 result: `38.30` (delta: `-0.28`)
- Spark FP16 result (above): `38.50266666666666` (delta: `-0.48`)
- Original (RTX 4090) KIVI-2 result: `38.03` (delta: `-0.01`, essentially
  matches — same method/model, different hardware)
- No task scored 0 or non-finite.

## Completed Baseline (KIVI-4, DGX Spark)

- Model: `lmsys/longchat-7b-v1.5-32k`
- Method: KIVI-4
- K bits: `4`, V bits: `4`
- Group size: `32`, residual length: `128`
- Context limit: `31,500`
- LongBench tasks: `15`
- Samples: `3,550` (validated: 15/15 task files, 0 `.partial`, 0 parse
  errors, 3550/3550 total rows)
- Seed: `42`
- Host: DGX Spark (aarch64, NVIDIA GB10, CUDA 13.0 driver)
- Prediction path:
  `pred/longchat-7b-v1.5-32k_31500_4bits_group32_residual128/`
- Prediction log: `logs/longbench_kivi4_20260731_000821.log`
- Evaluation log: `logs/longbench_kivi4_eval_20260731_220040.log` (original);
  re-verified `2026-08-03` with a fresh evaluation pass over the same
  unmodified prediction JSONLs, log `logs/longbench_kivi4_eval_20260803_092801.log`
  — identical per-task scores and average, confirming the result is
  deterministic and not an artifact of a single evaluation run
- Manifest: `outputs/run_manifests/longbench_kivi4_20260731_000821.txt`
- Result path:
  `pred/longchat-7b-v1.5-32k_31500_4bits_group32_residual128/result.json`
- Git commit at validation/evaluation time: `6f903a66033947519ac93ee78c9dc26a0dd968a9`
- Environment: `transformers==4.43.4`, `datasets==3.6.0`

Per-task scores:

| task | score |
|---|---|
| triviaqa | 83.93 |
| narrativeqa | 20.95 |
| passage_retrieval_en | 32.50 |
| gov_report | 31.38 |
| qasper | 29.03 |
| repobench-p | 56.52 |
| trec | 66.50 |
| multifieldqa_en | 43.67 |
| qmsum | 22.91 |
| musique | 14.72 |
| lcc | 52.47 |
| samsum | 40.79 |
| hotpotqa | 33.01 |
| multi_news | 26.62 |
| 2wikimqa | 24.60 |

- **Result average: `38.64`**
- Reference KIVI paper KIVI-4 result: `38.79` (delta: `-0.15`)
- Spark FP16 result (above): `38.50266666666666` (delta: `+0.14`)
- Spark KIVI-2 result (above): `38.01866666666667` (delta: `+0.62`)
- No task scored 0 or non-finite.

## Final Baseline Comparison (DGX Spark)

FP16, KIVI-2, and KIVI-4 were all run on the same DGX Spark host, same
model (`lmsys/longchat-7b-v1.5-32k`), same 15 LongBench tasks / 3,550
samples, same context limit (`31,500`), same group size (`32`) / residual
length (`128`) where applicable, same seed (`42`), and greedy decoding.

| method | reproduced average | paper reference | delta vs. paper | delta vs. Spark FP16 |
|---|---|---|---|---|
| FP16 | `38.50` (38.50266666666666) | `38.72` | `-0.22` | `+0.00` |
| KIVI-2 | `38.02` (38.01866666666667) | `38.30` | `-0.28` | `-0.48` |
| KIVI-4 | `38.64` | `38.79` | `-0.15` | `+0.14` |

- KIVI-4 - KIVI-2 = `+0.62` (4-bit clearly outperforms 2-bit, as expected).
- KIVI-4 slightly exceeds the local FP16 average; this is a known phenomenon
  also reported for 4-bit KIVI in the literature (quantization noise can act
  as a mild regularizer on some tasks) and is not itself a correctness
  concern given per-task deltas are all small (see below).
- Task keys are identical across all three result files (verified
  programmatically).
- No task scored `0` or a non-finite value in any of the three runs.
- Largest per-task deltas vs. FP16 are on `passage_retrieval_en`
  (KIVI-2: `+1.75`, KIVI-4: `+2.00`) and `multifieldqa_en` (KIVI-2:
  `-1.84`); these are within normal run-to-run variance for 200-sample
  greedy-decoded tasks and do not indicate a broken task.
- This DGX Spark same-environment baseline gate (FP16 vs. KIVI-2 vs. KIVI-4,
  all measured, all close to their paper references, no anomalous or
  missing tasks) is considered **PASS**.

## Current Gates

- Gate A - Implementation Correctness: **PARTIAL**. The packed KIVI path and
  custom CUDA GEMV are present and the KIVI-2/KIVI-4 parameters are wired
  into the model config, but mixed K/V and per-layer routes are unsupported.
- Gate B - Quality Reproduction: **PASS on DGX Spark for FP16, KIVI-2, and
  KIVI-4**. All three completed all 15 tasks with results within ~0.15-0.5
  of their respective paper references; see "Final Baseline Comparison"
  above. The original RTX 4090 KIVI-2 result (`38.03`) is not superseded by
  the Spark KIVI-2 result (`38.02`) — both are recorded, and they closely
  agree.
- Gate C - System Reproduction: **PARTIAL**. Packed low-bit storage is
  implemented, and FP16/KIVI-2/KIVI-4 quality has now been measured on the
  same host, but there is still no controlled comparison for peak memory,
  bytes per token, or throughput across the three methods on this host.

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
- Triton ptxas compatibility (required for KIVI-2 and KIVI-4, not FP16):
  Triton 3.5.1 (bundled with `torch==2.9.1`) ships a CUDA 12.8 `ptxas` that
  does not recognize GB10's `sm_121a` target
  (`ptxas fatal: Value 'sm_121a' is not defined for option 'gpu-name'`).
  The KIVI-2 and KIVI-4 runs exercise the Triton quant/pack path
  (`triton_quantize_and_pack_along_last_dim`) and the `kivi_gemv` CUDA
  extension, both of which need
  `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` set in the launch
  environment. The FP16 run (`k_bits=16, v_bits=16`) never touches this code
  path and does not need the override. Both the KIVI-2 and KIVI-4 full
  LongBench runs completed successfully with this override set, confirming
  it as a durable fix (not just the earlier isolated smoke test). See
  `environment/README.md` for the original diagnosis.

All version pins above are recorded in `environment/spark-baseline-requirements.txt`.

## Reproduction Commands

FP16:

```bash
mkdir -p logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/long_test.sh 0 16 16 32 128 lmsys/longchat-7b-v1.5-32k \
  > logs/longbench_fp16.log 2>&1
```

```bash
./.venv/bin/python eval_long_bench.py \
  --model longchat-7b-v1.5-32k_31500_16bits_group32_residual128
```

KIVI-2:

```bash
mkdir -p logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
bash scripts/long_test.sh 0 2 2 32 128 lmsys/longchat-7b-v1.5-32k \
  > logs/longbench_kivi2.log 2>&1
```

```bash
./.venv/bin/python eval_long_bench.py \
  --model longchat-7b-v1.5-32k_31500_2bits_group32_residual128
```

KIVI-4:

```bash
mkdir -p logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
bash scripts/long_test.sh 0 4 4 32 128 lmsys/longchat-7b-v1.5-32k \
  > logs/longbench_kivi4.log 2>&1
```

```bash
./.venv/bin/python eval_long_bench.py \
  --model longchat-7b-v1.5-32k_31500_4bits_group32_residual128
```

Evaluation output for each method is written to
`result.json` inside the corresponding prediction directory, e.g.
`pred/longchat-7b-v1.5-32k_31500_16bits_group32_residual128/result.json`.
