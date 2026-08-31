# Stage G4: Pre-Registered Scientific Feature-Response Analysis

- Generated: 2026-08-31T07:45:06.656753+00:00
- Primary features: `/home/kevin/Documents/KIVI_test/outputs/layer_feature_pilot/primary/primary_g3b/features.jsonl` (256 rows)
- Diagnostic features: `/home/kevin/Documents/KIVI_test/outputs/layer_feature_pilot/diagnostic/diagnostic_g3b/features.jsonl` (16 rows)
- Aggregation: equal-sample arithmetic mean across exactly 4 pre-registered calibration samples per (task, layer, axis) -- never weighted by token counts. Median reported as the pre-registered secondary robustness summary only.
- 32 primary task-layer observations per axis (4 tasks x 8 layers).

## A. Preregistered primary reconstruction result (relative_l2)

**Key axis, task-specific Spearman(mean relative_l2, SignedKeySensitivity), n=8 layers each:**

| task | raw_rho | gate_rho |
|---|---|---|
| trec | -0.0825 | -0.0825 |
| lcc | +0.1190 | +0.1190 |
| passage_retrieval_en | +0.2829 | +0.2829 |
| 2wikimqa | +0.2196 | +0.2196 |

- median gate_rho = +0.1693, median(abs(gate_rho)) = 0.1693, negative_count = 1/4, range [-0.0825, +0.2829]

**Value axis, task-specific Spearman(mean relative_l2, SignedValueSensitivity), n=8 layers each:**

| task | raw_rho | gate_rho |
|---|---|---|
| trec | undefined | +0.0000 |
| lcc | -0.0952 | -0.0952 |
| passage_retrieval_en | -0.2520 | -0.2520 |
| 2wikimqa | +0.4048 | +0.4048 |

- median gate_rho = -0.0476, median(abs(gate_rho)) = 0.1736, negative_count = 2/4, range [-0.2520, +0.4048]

## B. Exact A/B/C/D gate (analysis/feature_pilot_gate.py, unmodified)

**Key axis:**
- A (>=3/4 gate_rho<0): FAIL (negative_count=1/4)
- B (median(abs(gate_rho))>=0.5): FAIL (median_abs_gate_rho=0.1693)
- C (lcc-removed, >=2/3 negative AND median<0): FAIL (negative_count=1/3, median=+0.2196)
- D (Layer-0 n=3 gate_rho<0): FAIL (gate_rho=+0.5000)
- **axis_go = False**

**Value axis:**
- A (>=3/4 gate_rho<0): FAIL (negative_count=2/4)
- B (median(abs(gate_rho))>=0.5): FAIL (median_abs_gate_rho=0.1736)
- C (lcc-removed, >=2/3 negative AND median<0): FAIL (negative_count=1/3, median=+0.0000)
- D (Layer-0 n=3 gate_rho<0): PASS (gate_rho=-1.0000)
- **axis_go = False**

**FEATURE_PILOT_STAGE = NO_GO**

## C. Layer-0 diagnostic (lcc / multifieldqa_en / samsum)

**Key axis:** Criterion-D raw_rho = +0.5000

**Value axis:** Criterion-D raw_rho = -1.0000

(Full 3-row-per-axis table: `diagnostic_layer0.csv`.)

## D. Family-agnostic exploratory distribution results (EXPLORATORY / NON-GATE)

These never affect FEATURE_PILOT_STAGE; reported for exploratory purposes only.

- key/p99_abs: median gate_rho = -0.0900, negative_count = 4/4, range [-0.1029, -0.0476]
- key/outlier_fraction: median gate_rho = +0.0293, negative_count = 2/4, range [-0.2857, +0.4124]
- value/p99_abs: median gate_rho = -0.0476, negative_count = 2/4, range [-0.1429, +0.0000]
- value/outlier_fraction: median gate_rho = +0.0476, negative_count = 1/4, range [-0.1260, +0.1429]

## E. Secondary/descriptive analyses

Secondary features (mean, std, variance, max_abs, p50_abs, p95_abs, mse, max_abs_error) are tabulated as aggregated values only in `secondary_features.csv` -- no correlation leaderboard was computed for them, and none can override the preregistered gate.

**LOTO (descriptive):** the preregistered Criterion C already tests lcc-removal specifically. Additional single-task-removal summaries for all 4 tasks are in `loto.csv`; none of them constitutes a new decision criterion.

## F. Limitations

- Correlation does not prove mechanism. Reconstruction error is not claimed to universally predict sensitivity. Layer index is not claimed irrelevant. Task-conditioned features are not claimed to solve quantizer routing. n=8 layers per task is not claimed sufficient for a learned predictor. An undefined correlation was never treated as rho=0 scientifically -- only gate_rho=0.0 for gate evaluation. A failed gate is not claimed to prove no relationship exists at all -- only that this preregistered relative_l2 hypothesis did not clear this small pilot's preregistered bar.
