# K/V Cache 3x3 Ablation: Paired Sample-Level and Bootstrap Analysis

- Generated: 2026-08-18T18:29:16.685872+08:00
- Git commit: `75c37b62f0c02866df23c3b5c69d7c9b8de58349`
- Bootstrap iterations: 10000, base seed: 42
- Pred root: `/home/kevin/Documents/KIVI_test/pred`

## 1. Data validation status

| check | status |
|---|---|
| configuration discovery (9/9) | PASS |
| strict JSONL integrity (row counts, valid JSON, no NUL, trailing newline, no leftover .partial) | PASS |
| task-set completeness (15/15 per config) | PASS |
| sample-level pairing (answers/all_classes/length, all 9 configs) | PASS |
| recomputed scores vs result.json (135 config-task pairs) | PASS (0 mismatches) |

**Pairing method**: predictions carry no explicit sample ID; pairing is validated by same-task row order plus row-by-row equality of `answers`, `all_classes`, and `length` across all 9 configurations. This analysis fails closed (raises, writes nothing) if any of the above checks fail.

**Configuration discovery** (how each of the 9 directories was resolved):

| directory | status | config / reason |
|---|---|---|
| longchat-7b-v1.5-32k_31500_16bits_group32_residual128 | RESOLVED | K16/V16 |
| longchat-7b-v1.5-32k_31500_2bits_group32_residual128 | RESOLVED | K2/V2 |
| longchat-7b-v1.5-32k_31500_4bits_group32_residual128 | RESOLVED | K4/V4 |
| longchat-7b-v1.5-32k_31500_k16_v2_group32_residual128 | RESOLVED | K16/V2 |
| longchat-7b-v1.5-32k_31500_k16_v4_group32_residual128 | RESOLVED | K16/V4 |
| longchat-7b-v1.5-32k_31500_k2_v16_group32_residual128 | RESOLVED | K2/V16 |
| longchat-7b-v1.5-32k_31500_k2_v4_group32_residual128 | RESOLVED | K2/V4 |
| longchat-7b-v1.5-32k_31500_k4_v16_group32_residual128 | RESOLVED | K4/V16 |
| longchat-7b-v1.5-32k_31500_k4_v2_group32_residual128 | RESOLVED | K4/V2 |

## 2. Full 3x3 matrix (official average -- mean of 15 rounded task scores)

| K \ V | V16 | V4 | V2 |
|---|---|---|---|
| K16 | 38.502667 | 38.480667 | 38.254000 |
| K4 | 38.688667 | 38.640000 | 38.188000 |
| K2 | 38.589333 | 38.496000 | 38.018667 |

## 3. Deltas vs FP16 (raw/unrounded, percentage points)

Sign convention: delta = score(destination) - score(source); negative = degradation.

| config | delta vs FP16 | 95% bootstrap CI | |
|---|---|---|---|
| K16/V4 | -0.0215 | [-0.1584, +0.1107] | CI crosses 0 (unstable) |
| K16/V2 | -0.2499 | [-0.5203, +0.0230] | CI crosses 0 (unstable) |
| K4/V16 | +0.1855 | [-0.0185, +0.4015] | CI crosses 0 (unstable) |
| K4/V4 | +0.1368 | [-0.1163, +0.3912] | CI crosses 0 (unstable) |
| K4/V2 | -0.3148 | [-0.6055, -0.0198] | CI excludes 0 |
| K2/V16 | +0.0842 | [-0.4169, +0.6046] | CI crosses 0 (unstable) |
| K2/V4 | -0.0084 | [-0.4993, +0.5075] | CI crosses 0 (unstable) |
| K2/V2 | -0.4833 | [-0.9532, +0.0010] | CI crosses 0 (unstable) |

## 4. Key trajectories (fixed Value precision)

| contrast | observed | 95% CI | |
|---|---|---|---|
| K16->K4 @ V16 | +0.1855 | [-0.0185, +0.4015] | CI crosses 0 (unstable) |
| K4->K2 @ V16 | -0.1013 | [-0.6067, +0.4048] | CI crosses 0 (unstable) |
| K16->K2 @ V16 | +0.0842 | [-0.4169, +0.6046] | CI crosses 0 (unstable) |
| K16->K4 @ V4 | +0.1583 | [-0.0737, +0.3987] | CI crosses 0 (unstable) |
| K4->K2 @ V4 | -0.1452 | [-0.6267, +0.3367] | CI crosses 0 (unstable) |
| K16->K2 @ V4 | +0.0131 | [-0.4773, +0.5222] | CI crosses 0 (unstable) |
| K16->K4 @ V2 | -0.0649 | [-0.2654, +0.1357] | CI crosses 0 (unstable) |
| K4->K2 @ V2 | -0.1685 | [-0.6499, +0.3099] | CI crosses 0 (unstable) |
| K16->K2 @ V2 | -0.2334 | [-0.7218, +0.2570] | CI crosses 0 (unstable) |

## 5. Value trajectories (fixed Key precision)

| contrast | observed | 95% CI | |
|---|---|---|---|
| V16->V4 @ K16 | -0.0215 | [-0.1584, +0.1107] | CI crosses 0 (unstable) |
| V4->V2 @ K16 | -0.2284 | [-0.5096, +0.0587] | CI crosses 0 (unstable) |
| V16->V2 @ K16 | -0.2499 | [-0.5203, +0.0230] | CI crosses 0 (unstable) |
| V16->V4 @ K4 | -0.0487 | [-0.2077, +0.1070] | CI crosses 0 (unstable) |
| V4->V2 @ K4 | -0.4516 | [-0.7405, -0.1689] | CI excludes 0 |
| V16->V2 @ K4 | -0.5003 | [-0.8002, -0.2011] | CI excludes 0 |
| V16->V4 @ K2 | -0.0925 | [-0.2142, +0.0193] | CI crosses 0 (unstable) |
| V4->V2 @ K2 | -0.4749 | [-0.8000, -0.1663] | CI excludes 0 |
| V16->V2 @ K2 | -0.5675 | [-0.8980, -0.2543] | CI excludes 0 |

## 6. Bootstrap design

Paired, task-stratified bootstrap: for each of the 15 tasks, the same with-replacement sample of that task's row indices is applied to all 9 configurations in a given replicate; each configuration's replicate task score is the mean of the resampled per-sample scores; each configuration's replicate overall score is the equal (1/15) weighted mean of its 15 replicate task scores -- reproducing eval_long_bench.py's aggregation exactly (never a raw mean over all 3550 samples). All contrasts below are linear combinations of these paired per-replicate overall scores, so every reported CI is fully paired.

## 7. K x V interactions

Interaction = [S(K_dest,V_dest) - S(K_source,V_dest)] - [S(K_dest,V_source) - S(K_source,V_source)]

| interaction | observed | 95% CI | |
|---|---|---|---|
| K16->4 x V16->4 | -0.0272 | [-0.2337, +0.1764] | CI crosses 0 (unstable) |
| K16->4 x V4->2 | -0.2232 | [-0.5003, +0.0555] | CI crosses 0 (unstable) |
| K4->2 x V16->4 | -0.0438 | [-0.2379, +0.1541] | CI crosses 0 (unstable) |
| K4->2 x V4->2 | -0.0233 | [-0.4313, +0.3818] | CI crosses 0 (unstable) |
| K16->2 x V16->2 (broad) | -0.3176 | [-0.7356, +0.0856] | CI crosses 0 (unstable) |

## 8. Task-level sensitivity

Formulas (raw/unrounded per-task percentage points): `k_sensitivity` = mean over V in {16,4,2} of |score(K2,V) - score(K16,V)|; `v_sensitivity` = mean over K in {16,4,2} of |score(K,V2) - score(K,V16)|; `tolerance_4bit` = mean of {K4/V16-FP16, K16/V4-FP16} (single-axis 4-bit drop); `sensitivity_2bit` = mean of {K2/V16-FP16, K16/V2-FP16} (single-axis 2-bit drop); `joint_mixed_sensitivity` = mean of {K2/V2, K4/V4, K2/V4, K4/V2} deltas vs FP16 (both axes quantized).

| task | n | worst delta vs FP16 | worst config | k_sensitivity | v_sensitivity | tol_4bit | sens_2bit | joint_mixed |
|---|---|---|---|---|---|---|---|---|
| narrativeqa | 200 | -0.6902 | K2/V16 | +0.5757 | +0.5255 | +0.1696 | -0.0502 | +0.0243 |
| qasper | 200 | -1.0194 | K2/V2 | +0.3357 | +0.7497 | -0.6344 | -0.3085 | -0.5002 |
| multifieldqa_en | 150 | -1.8342 | K2/V2 | +2.0574 | +0.2747 | +0.3815 | -0.5144 | -0.7729 |
| hotpotqa | 200 | -0.3170 | K2/V16 | +0.3495 | +0.2850 | -0.0479 | +0.0391 | -0.0658 |
| musique | 200 | -1.0177 | K2/V2 | +0.7902 | +0.1215 | -0.0717 | -0.5078 | -0.4633 |
| 2wikimqa | 200 | -1.8039 | K2/V16 | +1.7129 | +0.1592 | +0.1865 | -0.9111 | -0.7507 |
| gov_report | 200 | -0.3365 | K2/V2 | +0.5680 | +0.6653 | +0.2915 | +0.6078 | +0.3516 |
| qmsum | 200 | -0.4585 | K2/V4 | +0.2550 | +0.1342 | +0.1349 | -0.1923 | -0.1178 |
| multi_news | 200 | -0.3470 | K4/V2 | +0.2669 | +0.3115 | -0.0379 | -0.0969 | -0.0455 |
| lcc | 500 | -6.2140 | K4/V2 | +2.2533 | +4.4793 | +0.2240 | -1.8890 | -1.5110 |
| repobench-p | 500 | -1.6240 | K2/V2 | +1.1313 | +0.6247 | -0.0400 | -1.0550 | -0.9995 |
| triviaqa | 200 | -1.2441 | K2/V2 | +0.9835 | +0.1926 | -0.1292 | -0.3415 | -0.5586 |
| samsum | 200 | -0.2801 | K2/V4 | +0.1445 | +0.5300 | +0.0533 | +0.2273 | +0.2108 |
| trec | 200 | +0.0000 | K16/V4 | +0.3333 | +0.1667 | +0.0000 | +0.2500 | +0.1250 |
| passage_retrieval_en | 200 | -0.5000 | K16/V4 | +4.2500 | +2.0833 | +0.7500 | +3.5000 | +2.5625 |

Tasks most responsible for large changes in specific highlighted contrasts (most negative 5):

- **K16/V16 -> K16/V2**: lcc, qasper, repobench-p, multi_news, qmsum
- **K16/V16 -> K2/V16**: 2wikimqa, multifieldqa_en, repobench-p, triviaqa, musique
- **K16/V4 -> K16/V2**: lcc, repobench-p, qmsum, 2wikimqa, qasper
- **K2/V4 -> K2/V2**: passage_retrieval_en, lcc, qasper, gov_report, trec
- **K4/V4 -> K4/V2**: lcc, passage_retrieval_en, repobench-p, multi_news, qasper
- **K16/V2 -> K2/V2**: multifieldqa_en, 2wikimqa, triviaqa, gov_report, musique

## 9. Leave-one-task-out robustness

| contrast | full 15-task | min (task removed) | max (task removed) | sign ever flips |
|---|---|---|---|---|
| K16/V4 - FP16 | -0.0215 | -0.0559 (multifieldqa_en) | +0.0293 (qasper) | qasper, triviaqa, passage_retrieval_en |
| K16/V2 - FP16 | -0.2499 | -0.3391 (passage_retrieval_en) | +0.0954 (lcc) | lcc |
| K4/V16 - FP16 | +0.1855 | +0.0559 (passage_retrieval_en) | +0.2371 (qasper) | no |
| K4/V4 - FP16 | +0.1368 | +0.0037 (passage_retrieval_en) | +0.1837 (lcc) | no |
| K4/V2 - FP16 | -0.3148 | -0.4087 (passage_retrieval_en) | +0.1066 (lcc) | lcc |
| K2/V16 - FP16 | +0.0842 | -0.3384 (passage_retrieval_en) | +0.2191 (2wikimqa) | lcc, passage_retrieval_en |
| K2/V4 - FP16 | -0.0084 | -0.4018 (passage_retrieval_en) | +0.1189 (multifieldqa_en) | narrativeqa, qasper, multifieldqa_en, hotpotqa, musique, 2wikimqa, qmsum, repobench-p, triviaqa, samsum |
| K2/V2 - FP16 | -0.4833 | -0.6428 (passage_retrieval_en) | -0.3868 (multifieldqa_en) | no |
| K16->4 x V16->4 | -0.0272 | -0.0957 (qasper) | +0.0400 (lcc) | multifieldqa_en, lcc |
| K16->4 x V4->2 | -0.2232 | -0.2632 (samsum) | -0.0606 (passage_retrieval_en) | no |
| K4->2 x V16->4 | -0.0438 | -0.0997 (lcc) | -0.0024 (qasper) | no |
| K4->2 x V4->2 | -0.0233 | -0.2816 (lcc) | +0.1714 (passage_retrieval_en) | qasper, gov_report, triviaqa, trec, passage_retrieval_en |
| K16->2 x V16->2 (broad) | -0.3176 | -0.5597 (lcc) | +0.0347 (passage_retrieval_en) | passage_retrieval_en |

## 10. Safe interpretation

This section distinguishes point estimates, bootstrap-supported conclusions (CI excludes 0), and unstable/task-dependent observations (CI crosses 0, or leave-one-task-out changes sign).

**Point estimates only** (not claims of significance):
- The full 3x3 official-average matrix is reported in Section 2; all 9 cells are within a narrow band of FP16 (38.5027).

**Bootstrap-supported observations** (95% CI excludes 0):
- K4/V2 - FP16: -0.3148, 95% CI [-0.6055, -0.0198].
- V4->V2 @ K4: -0.4516, 95% CI [-0.7405, -0.1689].
- V16->V2 @ K4: -0.5003, 95% CI [-0.8002, -0.2011].
- V4->V2 @ K2: -0.4749, 95% CI [-0.8000, -0.1663].
- V16->V2 @ K2: -0.5675, 95% CI [-0.8980, -0.2543].

**Unstable / task-dependent observations** (CI crosses 0, or sign flips under leave-one-task-out):
- K16/V4 - FP16: -0.0215, 95% CI [-0.1584, +0.1107] crosses 0.
- K16/V2 - FP16: -0.2499, 95% CI [-0.5203, +0.0230] crosses 0.
- K4/V16 - FP16: +0.1855, 95% CI [-0.0185, +0.4015] crosses 0.
- K4/V4 - FP16: +0.1368, 95% CI [-0.1163, +0.3912] crosses 0.
- K2/V16 - FP16: +0.0842, 95% CI [-0.4169, +0.6046] crosses 0.
- K2/V4 - FP16: -0.0084, 95% CI [-0.4993, +0.5075] crosses 0.
- K2/V2 - FP16: -0.4833, 95% CI [-0.9532, +0.0010] crosses 0.
- K16->K4 @ V16: +0.1855, 95% CI [-0.0185, +0.4015] crosses 0.
- K4->K2 @ V16: -0.1013, 95% CI [-0.6067, +0.4048] crosses 0.
- K16->K2 @ V16: +0.0842, 95% CI [-0.4169, +0.6046] crosses 0.
- K16->K4 @ V4: +0.1583, 95% CI [-0.0737, +0.3987] crosses 0.
- K4->K2 @ V4: -0.1452, 95% CI [-0.6267, +0.3367] crosses 0.
- K16->K2 @ V4: +0.0131, 95% CI [-0.4773, +0.5222] crosses 0.
- K16->K4 @ V2: -0.0649, 95% CI [-0.2654, +0.1357] crosses 0.
- K4->K2 @ V2: -0.1685, 95% CI [-0.6499, +0.3099] crosses 0.
- K16->K2 @ V2: -0.2334, 95% CI [-0.7218, +0.2570] crosses 0.
- V16->V4 @ K16: -0.0215, 95% CI [-0.1584, +0.1107] crosses 0.
- V4->V2 @ K16: -0.2284, 95% CI [-0.5096, +0.0587] crosses 0.
- V16->V2 @ K16: -0.2499, 95% CI [-0.5203, +0.0230] crosses 0.
- V16->V4 @ K4: -0.0487, 95% CI [-0.2077, +0.1070] crosses 0.
- V16->V4 @ K2: -0.0925, 95% CI [-0.2142, +0.0193] crosses 0.
- K16/V4 - FP16: sign flips under leave-one-task-out when removing qasper, triviaqa, passage_retrieval_en.
- K16/V2 - FP16: sign flips under leave-one-task-out when removing lcc.
- K4/V2 - FP16: sign flips under leave-one-task-out when removing lcc.
- K2/V16 - FP16: sign flips under leave-one-task-out when removing lcc, passage_retrieval_en.
- K2/V4 - FP16: sign flips under leave-one-task-out when removing narrativeqa, qasper, multifieldqa_en, hotpotqa, musique, 2wikimqa, qmsum, repobench-p, triviaqa, samsum.
- K16->4 x V16->4: sign flips under leave-one-task-out when removing multifieldqa_en, lcc.
- K4->2 x V4->2: sign flips under leave-one-task-out when removing qasper, gov_report, triviaqa, trec, passage_retrieval_en.
- K16->2 x V16->2 (broad): sign flips under leave-one-task-out when removing passage_retrieval_en.

**On the working hypothesis** ("Value-cache quantization may remain relatively stable at 4-bit precision, while a larger degradation emerges when Value precision is reduced from 4 to 2 bits; Key sensitivity may also depend on Value precision"): see Sections 3-7 above for the exact point estimates and CIs this hypothesis should be checked against (V16->V4 vs V4->V2 trajectories at each fixed K, and the K x V interaction terms). This script reports the paired estimates; it does not itself assert the hypothesis is confirmed.

**What this analysis does NOT claim:**
- It does not claim any quantized configuration is better than FP16 merely because a point estimate is higher.
- It does not claim statistical equivalence between any two configurations (no equivalence test was implemented).
- It does not claim significance from point-estimate differences alone; only CI-based statements above are bootstrap-supported.
- It does not compute or report p-values.

## Limitations

1. Each configuration has exactly one deterministic greedy-decoded prediction pass; no repeated seeds.
2. Bootstrap estimates benchmark sample/task uncertainty, not run-to-run or cross-hardware variance.
3. Predictions carry no sample ID; pairing relies on validated row order plus answers/all_classes/length equality.
4. The 3 earliest configurations (FP16, K2/V2, K4/V4) predate run_config.json; their k_bits/v_bits are inferred from the established `_<N>bits_` legacy directory-naming convention, not independently confirmed metadata.
5. Only one model (LongChat-7B), one context length, and one host/bit-configuration set is covered.
6. Interaction terms are operational ablation interactions, not evidence of a causal mechanism.
