# Stage F0: Layer-0 K/V Asymmetry Task-Generalization Analysis

- Generated: 2026-08-24T13:25:02.477879+08:00
- Bootstrap: 10000 iterations, seed=42

One-layer-at-a-time perturbation measures LOCAL sensitivity around an otherwise-FP16 operating point, on a fixed screening task set. This analysis tests whether the Layer-0 Key/Value asymmetry found in Stage E's 4-task pilot is more task-general than that pilot alone could show -- it does not test multi-layer joint compression, and it does not establish that Layer 0 is universally special.

## A. Original Stage-E observation (4 tasks, restated here for comparison)

- SignedKeySensitivity = +0.9432, SignedValueSensitivity = -1.0274, AxisDifference = +1.9705

## B. New-task-only replication (2 tasks)

- SignedKeySensitivity = -0.0213, SignedValueSensitivity = -0.0215, AxisDifference = +0.0002
  - multifieldqa_en: KeyDelta=-0.1072, ValueDelta=+0.0249, AxisDifference=-0.1322
  - samsum: KeyDelta=+0.0646, ValueDelta=-0.0680, AxisDifference=+0.1326

## C. Combined six-task result

- SignedKeySensitivity = +0.6217, SignedValueSensitivity = -0.6921, AxisDifference = +1.3138

## D. lcc-removed robustness

- Original 4 minus lcc (3 tasks): SignedKey=+0.3775, SignedValue=+0.0208, AxisDifference=+0.3567
- Combined 6 minus lcc (5 tasks): SignedKey=+0.2180, SignedValue=+0.0039, AxisDifference=+0.2141

## E. Bootstrap uncertainty (95% paired CI, not a significance gate)

**original-4**
- SignedKeySensitivity: +0.9432, 95% CI [+0.5154, +1.4531]
- SignedValueSensitivity: -1.0274, 95% CI [-1.4299, -0.6555]
- AxisDifference: +1.9705, 95% CI [+1.4124, +2.5822]

**new-2**
- SignedKeySensitivity: -0.0213, 95% CI [-0.3228, +0.2647] (crosses zero)
- SignedValueSensitivity: -0.0215, 95% CI [-0.1666, +0.1128] (crosses zero)
- AxisDifference: +0.0002, 95% CI [-0.3191, +0.3098] (crosses zero)

**combined-6**
- SignedKeySensitivity: +0.6217, 95% CI [+0.3132, +0.9672]
- SignedValueSensitivity: -0.6921, 95% CI [-0.9686, -0.4398]
- AxisDifference: +1.3138, 95% CI [+0.9223, +1.7367]

## Pre-registered gate

- criterion_1_combined6_signed_key_gt_0: PASS
- criterion_2_combined6_signed_value_lt_0: PASS
- criterion_3_combined6_axis_difference_gt_0: PASS
- criterion_4_lcc_removed_all_three_hold: FAIL
- criterion_5_at_least_one_new_task_axis_difference_gt_0: PASS

**EARLY_LAYER_EXPANSION = NO_GO**

## What this analysis does NOT claim

- Does not claim Layer 0 is universally special.
- Does not claim Layer-0 quantization improves model quality in general.
- Does not claim statistical significance proves mechanism.
- Does not claim single-layer perturbation predicts multi-layer compression.
- Does not treat a bootstrap CI crossing zero as proof of equivalence.

## Master-log limitation

The F0 master nohup log is unusable due to a shell/job-control launch artifact (it contains only the line '[1]+: command not found', not actual generation output). This is NOT claimed to be a clean log scan. Independent evidence supporting generation integrity instead: exact 700/700 rows, pairing PASS, exit_code=0 for both conditions, nonfinite_logits_seen=false for both, stable boot IDs, and continuous host-monitor telemetry (~110 samples/condition at 30s cadence). Fixing the launcher's logging is a separate engineering task, not attempted here.
