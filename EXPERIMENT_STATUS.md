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
- Mixed K/V quantization is implemented (commit `8721f76e0f67841691c5384aeb3788fc7d9c521d`,
  `feature/mixed-kv-ablation` merged to `main`): a side with `bits==16` is a
  true FP16 pass-through (never quantized, packed, or run through the CUDA
  GEMV kernel), while the other side keeps the original packed low-bit path.
  `K2/V4`, `K4/V2`, `K2/V2`, `K4/V4`, and `K16/V16` all continue to work
  unchanged. See `docs/mixed_kv_quantization.md` for the loader decision
  table and full unit/kernel-route/numerical test results.
- Prediction directory names are asymmetric-safe: symmetric configs
  (`k_bits == v_bits`) keep the legacy `{model}_{len}_{bits}bits_group{g}_residual{r}`
  naming; mixed configs use `{model}_{len}_k{k}_v{v}_group{g}_residual{r}`,
  so `K2/V16` and `K16/V2` no longer collide with the `K2/V2`/`FP16`
  baseline directories.
- Per-layer K/V policies are implemented (`--layer-policy`, `utils/layer_policy.py`);
  families `kivi` and `rotation_kivi` (QuaRot-inspired Key-only Hadamard
  rotation) are executable.
- Polar is not implemented.

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

## Completed Ablation (K2/V16 - Key-only KIVI-2, DGX Spark)

- Model: `lmsys/longchat-7b-v1.5-32k`
- Method: Key-only KIVI-2 (mixed K/V quantization) — Key quantized to 2-bit,
  Value kept as an FP16 pass-through (never quantized/packed/GEMV'd)
- K bits: `2`, V bits: `16`
- Group size: `32`, residual length: `128`
- Context limit: `31,500`
- LongBench tasks: `15`
- Samples: `3,550` (validated: 15/15 task files, 0 `.partial`, 0 parse
  errors, 3550/3550 total rows)
- Seed: `42`
- Host: DGX Spark (aarch64, NVIDIA GB10, CUDA 13.0 driver)
- Git commit at launch: `8721f76e0f67841691c5384aeb3788fc7d9c521d`
- Prediction path:
  `pred/longchat-7b-v1.5-32k_31500_k2_v16_group32_residual128/`
- Prediction log: `logs/longbench_k2_v16_20260803_113403.log`
- Evaluation log: `logs/longbench_k2_v16_eval_20260804_181529.log`
- Manifest: `outputs/run_manifests/longbench_k2_v16_20260803_113403.txt`
- Result path:
  `pred/longchat-7b-v1.5-32k_31500_k2_v16_group32_residual128/result.json`

Per-task scores:

| task | score |
|---|---|
| triviaqa | 83.14 |
| narrativeqa | 20.12 |
| passage_retrieval_en | 36.50 |
| gov_report | 31.44 |
| qasper | 29.67 |
| repobench-p | 55.48 |
| trec | 67.00 |
| multifieldqa_en | 41.86 |
| qmsum | 22.53 |
| musique | 13.87 |
| lcc | 54.30 |
| samsum | 40.79 |
| hotpotqa | 32.74 |
| multi_news | 26.75 |
| 2wikimqa | 22.65 |

- **Result average: `38.5893` (38.589333333333336)**
- Spark FP16 result (above): `38.50266666666666` (delta: `+0.09`)
- No task scored 0 or non-finite. This run completed and was launched/health-
  checked with no interruption.

## Completed Ablation (K16/V2 - Value-only KIVI-2, DGX Spark)

- Model: `lmsys/longchat-7b-v1.5-32k`
- Method: Value-only KIVI-2 (mixed K/V quantization) — Key kept as an FP16
  pass-through, Value quantized to 2-bit
- K bits: `16`, V bits: `2`
- Group size: `32`, residual length: `128`
- Context limit: `31,500`
- LongBench tasks: `15`
- Samples: `3,550` (validated: 15/15 task files, 0 `.partial`, 0 parse
  errors, 3550/3550 total rows)
- Seed: `42`
- Host: DGX Spark (aarch64, NVIDIA GB10, CUDA 13.0 driver)
- Git commit at launch: `8721f76e0f67841691c5384aeb3788fc7d9c521d`
- Prediction path:
  `pred/longchat-7b-v1.5-32k_31500_k16_v2_group32_residual128/`
- Original (interrupted) prediction log:
  `logs/longbench_k16_v2_20260804_181640.log`
- Resume prediction log:
  `logs/longbench_k16_v2_resume_20260806_210613.log`
- Evaluation log: `logs/longbench_k16_v2_eval_20260807_001157.log`
- Manifests:
  `outputs/run_manifests/longbench_k16_v2_20260804_181640.txt`,
  `outputs/run_manifests/longbench_k16_v2_resume_20260806_210613.txt`
- Result path:
  `pred/longchat-7b-v1.5-32k_31500_k16_v2_group32_residual128/result.json`

**Interruption and safe resume.** The original launch (`2026-08-04 18:16`)
progressed cleanly through 11 of 15 tasks and 10/200 rows into `triviaqa`,
then stopped writing to its log at `2026-08-05 06:50` with no further
activity. A read-only root-cause investigation (no prediction re-run, no
file changes) found:

- Root cause classification: **`UNKNOWN_ABRUPT_TERMINATION`** — the process
  and the system's own `journald` logging stopped within the same ~50-minute
  window (`~06:00`-`~06:50` on `2026-08-05`), with no clean shutdown target,
  no OOM message, no CUDA/Python traceback, and no "Killed"/"Terminated"
  text anywhere in the 174 KB log. The actual host reboot did not occur
  until `2026-08-06 05:50`, roughly 23 hours later — consistent with a
  silent whole-system freeze/hang followed by a much later manual/watchdog
  power-cycle, rather than a clean reboot or a single killed process.
  Kernel-level evidence that could confirm or rule out a GPU driver/Xid
  event or an OOM-killer action (`dmesg`, `journalctl -k`) was not
  accessible without `sudo`, which was intentionally not used; this remains
  an open, unresolved gap in the evidence.
- All 11 completed task JSONLs and the `triviaqa.jsonl.partial` (10 valid
  rows) were verified byte-for-byte parseable with 0 errors, and
  `run_config.json` matched the run being resumed.
- Resume used the identical launch command/config (no code, package, or
  config changes). `pred_long_bench.py`'s existing resume logic (`skip` for
  complete `.jsonl`, `resume ... from ... .partial` with `start_idx=done`
  for the partial, atomic `os.replace` to the final name only once a task
  reaches its expected row count) correctly skipped all 11 complete tasks,
  resumed `triviaqa` from row 10 (not row 0), and then ran `samsum`, `trec`,
  and `passage_retrieval_en` from scratch. The resumed run completed all
  remaining work and exited cleanly with no errors.

Final state: **15/15 tasks, 3,550/3,550 rows, 0 `.partial` files, 0 parse
errors.**

Per-task scores:

| task | score |
|---|---|
| triviaqa | 84.15 |
| narrativeqa | 21.40 |
| passage_retrieval_en | 31.50 |
| gov_report | 31.40 |
| qasper | 28.46 |
| repobench-p | 55.99 |
| trec | 66.50 |
| multifieldqa_en | 43.95 |
| qmsum | 22.55 |
| musique | 14.54 |
| lcc | 47.91 |
| samsum | 41.28 |
| hotpotqa | 33.45 |
| multi_news | 26.30 |
| 2wikimqa | 24.43 |

- **Result average: `38.254` (38.254000000000005)**
- Spark FP16 result (above): `38.50266666666666` (delta: `-0.25`)
- No task scored 0 or non-finite.

## Five-Way KV Cache Ablation Comparison (DGX Spark)

FP16, K2/V16 (Key-only), K16/V2 (Value-only), K2/V2 (joint KIVI-2), and
K4/V4 (KIVI-4) were all run on the same host (DGX Spark, aarch64, NVIDIA
GB10, CUDA 13.0), same model (`lmsys/longchat-7b-v1.5-32k`), same 15
LongBench tasks / 3,550 samples, same context limit (`31,500`), same group
size (`32`) / residual length (`128`), same seed (`42`), and greedy
decoding.

| method | average | delta vs. FP16 |
|---|---|---|
| FP16 | `38.5027` | `+0.0000` |
| K2/V16 (Key-only) | `38.5893` | `+0.0867` |
| K16/V2 (Value-only) | `38.2540` | `-0.2487` |
| K2/V2 (joint KIVI-2) | `38.0187` | `-0.4840` |
| K4/V4 (KIVI-4) | `38.6400` | `+0.1373` |

Ablation decomposition (loss = average − FP16 average):

- Key-only loss (K2/V16 − FP16): `+0.0867` (no measurable degradation;
  slightly above FP16, within run-to-run task-level variance)
- Value-only loss (K16/V2 − FP16): `-0.2487`
- Joint K2/V2 loss (K2/V2 − FP16): `-0.4840`
- Interaction (joint − key-only − value-only): `-0.3220`
- K2/V16 − K16/V2: `+0.3353`

Per-task comparison:

| task | FP16 | K2/V16 | K16/V2 | K2/V2 | K4/V4 |
|---|---|---|---|---|---|
| 2wikimqa | 24.45 | 22.65 | 24.43 | 22.93 | 24.60 |
| gov_report | 30.81 | 31.44 | 31.40 | 30.48 | 31.38 |
| hotpotqa | 33.05 | 32.74 | 33.45 | 32.98 | 33.01 |
| lcc | 52.99 | 54.30 | 47.91 | 52.28 | 52.47 |
| multi_news | 26.62 | 26.75 | 26.30 | 26.60 | 26.62 |
| multifieldqa_en | 43.42 | 41.86 | 43.95 | 41.58 | 43.67 |
| musique | 14.71 | 13.87 | 14.54 | 13.69 | 14.72 |
| narrativeqa | 20.81 | 20.12 | 21.40 | 21.04 | 20.95 |
| passage_retrieval_en | 30.50 | 36.50 | 31.50 | 32.25 | 32.50 |
| qasper | 29.37 | 29.67 | 28.46 | 28.35 | 29.03 |
| qmsum | 22.73 | 22.53 | 22.55 | 22.48 | 22.91 |
| repobench-p | 56.79 | 55.48 | 55.99 | 55.17 | 56.52 |
| samsum | 40.80 | 40.79 | 41.28 | 41.21 | 40.79 |
| trec | 66.50 | 67.00 | 66.50 | 66.50 | 66.50 |
| triviaqa | 83.99 | 83.14 | 84.15 | 82.74 | 83.93 |

**Analysis:**

1. On the 15-task average, **Value-only quantization (K16/V2) costs more
   than Key-only (K2/V16)**: Key-only shows no measurable loss versus FP16
   (`+0.09`), while Value-only loses `-0.25`.
2. The joint K2/V2 loss (`-0.48`) is **not** close to the sum of the two
   individual losses (`+0.09 + -0.25 = -0.16`); the actual joint
   degradation is roughly 3x larger than that naive sum.
3. The interaction term is **negative and non-trivial** (`-0.32`), meaning
   quantizing both Key and Value together hurts noticeably more than the
   two individual effects predict on their own (a super-additive/synergistic
   degradation, not an independent/additive one).
4. Tasks most sensitive to **Key** quantization (K2/V16 vs. FP16, biggest
   drops): `2wikimqa` (`-1.80`), `multifieldqa_en` (`-1.56`), `repobench-p`
   (`-1.31`), `musique` (`-0.84`), `triviaqa` (`-0.85`), `narrativeqa`
   (`-0.69`). Multi-hop QA and long-context QA tasks appear more sensitive
   to Key quantization than to Value quantization.
5. Tasks most sensitive to **Value** quantization (K16/V2 vs. FP16, biggest
   drops): `lcc` (`-5.08`, by far the largest single-task delta observed in
   any of the five configurations), `qasper` (`-0.91`), `repobench-p`
   (`-0.80`). Notably, the two code-completion tasks (`lcc`, `repobench-p`,
   both scored with `code_sim_score`) are the most Value-sensitive tasks,
   while `lcc` actually *improves* under Key-only quantization (`+1.31`) —
   a striking asymmetry specific to code-completion-style tasks.
6. The overall K2/V16 vs. K16/V2 average gap (`0.34` points) is **small
   relative to individual per-task swings** seen in this same comparison
   (`lcc`: `-5.08` under Value-only; `passage_retrieval_en`: `+6.00` under
   Key-only). This means the aggregate 15-task average alone is not a
   strong basis for a general "Key quantization is safer than Value
   quantization" claim; the task-level pattern (code-completion tasks being
   Value-sensitive; multi-hop/long-context QA being more Key-sensitive) is
   the more informative and specific signal here.
7. **These results are preliminary.** Each configuration was run once
   (single seed, single pass) with no repeated trials, confidence
   intervals, or significance testing; `passage_retrieval_en` (`+6.00`
   under Key-only) and `lcc` (`-5.08` under Value-only) are large enough
   single-task swings on 200-sample tasks that they should not be
   over-interpreted as precise quantization-sensitivity measurements without
   further repetition. The average-level comparisons (all five methods
   within `~0.6` points of each other) and the interaction estimate should
   be treated as directional evidence, not confirmed effect sizes.

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

## Phase 2 Summary: Layer-Wise KV Quantization (DGX Spark)

Research question (set 2026-08-18, after the Phase-1 3x3 global K/V
ablation): *can layer-wise KV distribution and attention geometry predict
the most suitable quantization route (FP16 / KIVI / Rotation-KIVI / Polar)
under a fixed memory budget?* Answering it needs three things in order:
(1) layer sensitivity is heterogeneous, (2) measurable layer features
predict it, (3) route choice per layer improves quality at a fixed budget.

Phase-1 starting point (`analysis/results/kv_3x3/`): all 9 global K/V
configs validated; only K4/V2 - FP16 had a 95% CI excluding zero
(`-0.31`, `[-0.61, -0.02]`); all explicit KxV interaction CIs cross zero.

All Phase-2 perturbations are one layer at a time (target layer quantized,
all others K16/V16), on layers 0, 4, 9, 13, 18, 22, 27, 31; paired
bootstrap 10,000 / seed 42 throughout. Every stage was pre-registered
(`docs/stage_g3b_pre_registration.md`, `docs/stage_h0_pre_registration.md`,
`docs/stage_i2_pre_registration.md`) and its gate fixed before data.

| stage | question | result | outputs |
|---|---|---|---|
| A-C | per-layer K/V policy support (`--layer-policy`) | implemented, global behavior unchanged | `utils/layer_policy.py` |
| D-E | is sensitivity heterogeneous across layers? (K2/V16 or K16/V2 at one layer; trec, lcc, passage_retrieval_en, 2wikimqa; 17,600 rows) | mostly Layer 0 only: SignedKey `+0.94` `[+0.52, +1.45]`, SignedValue `-1.03` `[-1.43, -0.66]`; other layers within about +-0.3 | `analysis/results/layer_sensitivity_pilot/` |
| F0 | does the Layer-0 K/V asymmetry generalize to new tasks (multifieldqa_en, samsum)? | **`EARLY_LAYER_EXPANSION = NO_GO`**: new-task AxisDifference `+0.0002` `[-0.32, +0.31]`; fails the lcc-removed criterion | `analysis/results/layer_sensitivity_f0/` |
| G3-G4 | do K/V reconstruction features (`relative_l2`) predict sensitivity? | **`FEATURE_PILOT_STAGE = NO_GO`**: median abs Spearman `0.17` (Key) / `0.17` (Value) vs `0.5` threshold | `analysis/results/layer_feature_pilot/` |
| H | do decode-time attention-geometry features (logit / attention distortion) predict sensitivity? | **`STAGE_H_ATTENTION_FEATURE = NO_GO`**: median abs Spearman `0.13` (Key) / `0.14` (Value) | `analysis/results/layer_attention_feature_pilot/` |
| I1-I2 | does Rotation-KIVI beat KIVI at some layers? | **`MIXED_POINT_ESTIMATE_DIRECTION`**: every layer CI crosses zero (see next section) | `analysis/results/i2_layer_family_sensitivity/` |
| - | Polar route | not implemented | - |
| - | per-layer allocation under a fixed memory budget | not run | - |

Conclusion: on `longchat-7b-v1.5-32k`, the preconditions for layer-wise
route selection do not hold. Single-layer sensitivity is small except for a
Layer-0 effect that is driven largely by lcc and does not generalize;
neither feature family predicts it; and swapping the quantizer family
gives no layer-dependent gain. The headroom is also small to begin with:
global KIVI-2 is only `0.48` below FP16 (38.02 vs 38.50) while cutting the
KV cache by 81% (see "System Measurement").

Not established (scope limits): multi-layer joint compression (all probes
are single-layer, i.e. local sensitivity around FP16); an actual
fixed-budget comparison such as "Layer 0 FP16 + all other layers K2/V2" vs
global K2/V2; other models; seeds other than 42.

## Stage I2: Layer x Quantizer-Family Sensitivity (DGX Spark)

Pre-registered in `docs/stage_i2_pre_registration.md`. For each of 8 layers
(0, 4, 9, 13, 18, 22, 27, 31), only that layer is K2/V16 under `kivi` or
`rotation_kivi`; all other layers are K16/V16. 16 conditions x 6 tasks
(trec, lcc, passage_retrieval_en, 2wikimqa, multifieldqa_en, samsum) =
23,200 rows, collected 2026-09-09 to 2026-09-25 at commit `66fcbda`
(8 host hangs, each resumed without data loss). Analysis (I2D):
`analysis/results/i2_layer_family_sensitivity/`; pairing audit PASS.

Delta = Rotation - KIVI, equal-task mean, 95% paired bootstrap CI (10,000, seed 42):

| layer | delta | 95% CI |
|---|---|---|
| 0 | `+0.001` | `[-0.117, +0.112]` |
| 4 | `-0.098` | `[-0.326, +0.114]` |
| 9 | `-0.218` | `[-0.504, +0.041]` |
| 13 | `+0.121` | `[-0.361, +0.616]` |
| 18 | `-0.093` | `[-0.728, +0.530]` |
| 22 | `-0.022` | `[-0.299, +0.274]` |
| 27 | `-0.087` | `[-0.215, +0.050]` |
| 31 | `-0.134` | `[-0.323, +0.025]` |

- Interpretation category: **`MIXED_POINT_ESTIMATE_DIRECTION`** (not
  `STRONG_CROSSOVER`): every layer CI crosses zero.
- 6 of 48 task-level CIs exclude zero (5 favor KIVI, 1 favors Rotation:
  L18 2wikimqa `+1.19`); uncorrected for multiple comparisons (~2-3
  expected by chance).
- Rotation-KIVI has the same per-token KV-cache footprint as KIVI at
  matched bits, so it offers no memory advantage either.

## System Measurement (DGX Spark)

`scripts/system_benchmark.py`, 2026-09-25. Production loader
(`pred_long_bench.build_model_and_tokenizer`), batch size 1, fixed-seed
random prompt, 128 greedy decode tokens, median of 3 repeats (spread < 2%),
`lm_head` kept on GPU for every config. KV cache = bytes of every tensor
held by the cache after decode (includes the FP16 residual). Raw rows:
`analysis/results/system_benchmark/results.jsonl` (also 4K/16K contexts).

Context 31,500 (model weights 12.6 GiB for every config):

| config | KV cache | vs FP16 | peak alloc | prefill | decode |
|---|---|---|---|---|---|
| FP16 | `15.44 GiB` | - | `43.5 GiB` | `12.9 s` | `3.4 tok/s` |
| K2/V16 | `9.17 GiB` | `-41%` | `31.0 GiB` | `13.3 s` | `4.6 tok/s` |
| K16/V2 | `9.19 GiB` | `-40%` | `31.0 GiB` | `13.3 s` | `4.3 tok/s` |
| K4/V4 | `4.85 GiB` | `-69%` | `22.3 GiB` | `13.8 s` | `5.2 tok/s` |
| K2/V2 | `2.92 GiB` | `-81%` | `18.7 GiB` | `13.8 s` | `6.9 tok/s` |
| Rotation K2/V16 | `9.17 GiB` | `-41%` | `31.0 GiB` | `13.4 s` | `4.6 tok/s` |

- KV savings match the expected ~3 bits/element (2-bit codes + FP16
  scale/min per 32-element group).
- KIVI prefill is 3-7% slower (quantize/pack cost); decode is faster, and
  the gap grows with context (K2/V2 vs FP16: +19% at 4K, 2.0x at 31.5K).
  Part of the FP16 decode gap comes from HF `DynamicCache` concatenating
  the full cache every step, so the speedup is not purely a bit-width effect.
- Rotation-KIVI overhead is < 1% in prefill and negligible in decode.

## Current Gates

- Gate A - Implementation Correctness: **PARTIAL**. The packed KIVI path,
  custom CUDA GEMV, and mixed K/V quantization (16-bit pass-through on
  either side) are all present and covered by unit/kernel-route/numerical
  tests; per-layer K/V policies and the `rotation_kivi` family are
  implemented and canary-validated (Stages A-D, I1, I2B).
- Gate B - Quality Reproduction: **PASS on DGX Spark for FP16, KIVI-2,
  KIVI-4, K2/V16, and K16/V2**. All five completed all 15 tasks; FP16/
  KIVI-2/KIVI-4 are within ~0.15-0.5 of their respective paper references
  (see "Final Baseline Comparison"), and K2/V16/K16/V2 are within ~0.09-0.25
  of the Spark FP16 average (see "Five-Way KV Cache Ablation Comparison").
  The original RTX 4090 KIVI-2 result (`38.03`) is not superseded by the
  Spark KIVI-2 result (`38.02`) — both are recorded, and they closely agree.
- Gate C - System Reproduction: **PASS (batch size 1)**. KV-cache bytes,
  peak memory, prefill latency and decode throughput were measured for
  FP16, K2/V16, K16/V2, K2/V2, K4/V4 and Rotation-KIVI K2/V16 on this host
  (see "System Measurement (DGX Spark)"). Larger batch sizes are not yet
  measured.

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
