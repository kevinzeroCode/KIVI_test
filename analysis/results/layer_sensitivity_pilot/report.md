# Stage E: 8-Layer K/V Sensitivity Pilot -- Analysis

- Generated: 2026-08-23T00:50:56.833302+08:00
- Bootstrap: 10000 iterations, seed=42

One-layer-at-a-time perturbation measures LOCAL sensitivity around an otherwise-FP16 operating point. It does not directly predict multi-layer joint compression behavior.

## A. Point-estimate observations

FP16 baseline scores: {'trec': 66.5, 'lcc': 52.99, 'passage_retrieval_en': 30.5, '2wikimqa': 24.45}

| layer | SignedKey | SignedValue | AbsKey | AbsValue |
|---|---|---|---|---|
| 0 | +0.9432 | -1.0274 | +0.9432 | +1.0586 |
| 4 | +0.2630 | +0.0872 | +0.2630 | +0.0872 |
| 9 | +0.1118 | +0.0792 | +0.1382 | +0.0792 |
| 13 | -0.2093 | +0.2580 | +0.4593 | +0.2700 |
| 18 | -0.0108 | +0.1941 | +0.7778 | +0.1941 |
| 22 | +0.0338 | -0.0467 | +0.1562 | +0.0467 |
| 27 | +0.0501 | +0.0891 | +0.1189 | +0.0891 |
| 31 | +0.0011 | +0.0235 | +0.0699 | +0.0235 |

- Signed Key: min=L13 (-0.2093), max=L00 (+0.9432), range=+1.1524, std=+0.3472
- Signed Value: min=L00 (-1.0274), max=L13 (+0.2580), range=+1.2854, std=+0.4088
- Abs Key max/min ratio: {'ratio': 13.497677039716628, 'stable': True, 'note': None}
- Abs Value max/min ratio: {'ratio': 45.047872340425, 'stable': False, 'note': 'min absolute sensitivity (0.0235) is below 0.05 pp -- ratio is not a meaningful magnitude comparison'}

## B. Bootstrap-supported observations (95% CI excludes 0)

- [Key] KeySensitivity(L00): +0.9432, 95% CI [+0.5154, +1.4531]
- [Key] KeySensitivity(L04): +0.2630, 95% CI [+0.0155, +0.5200]
- [Value] ValueSensitivity(L00): -1.0274, 95% CI [-1.4299, -0.6555]
- [Value] ValueSensitivity(L18): +0.1941, 95% CI [+0.0041, +0.5070]
- [AxisDifference] AxisDifference(L00): +1.9705, 95% CI [+1.4124, +2.5822]
- [Signed range contrast] SignedKeyRangeContrast_L13_minus_L00: -1.1524, 95% CI [-1.9499, -0.3584]
- [Signed range contrast] SignedValueRangeContrast_L00_minus_L13: -1.2854, 95% CI [-1.8839, -0.7182]

Naming note: "MostHarmed"/"MostImproved" below describe SIGNED direction (most negative / most positive delta vs FP16) of the fixed point-estimate-selected layers, never magnitude. Magnitude ranking uses AbsKeySensitivity/AbsValueSensitivity (Section A) and must not be conflated with these signed-range contrasts.

## C. Task-dependent / LOTO-unstable observations

- SignedKeyRangeContrast (MostHarmedKey L13 minus MostImprovedKey L00) LOTO: full=-1.1524, min=-1.5365 (trec), max=-0.2579 (lcc), sign_changed=no
- SignedValueRangeContrast (MostHarmedValue L00 minus MostImprovedValue L13) LOTO: full=-1.2854, min=-1.7427 (2wikimqa), max=-0.1379 (lcc), sign_changed=no

Pairwise Spearman correlations between tasks' 8-layer sensitivity vectors (descriptive, n=8 per vector):

- Key: {'trec_vs_lcc': -0.24743582965269675, 'trec_vs_passage_retrieval_en': -0.5345224838248487, 'trec_vs_2wikimqa': 0.5808178695515156, 'lcc_vs_passage_retrieval_en': 0.18002057495577392, 'lcc_vs_2wikimqa': 0.3233590907165799, 'passage_retrieval_en_vs_2wikimqa': 0.0}
- Value: {'trec_vs_lcc': None, 'trec_vs_passage_retrieval_en': None, 'trec_vs_2wikimqa': None, 'lcc_vs_passage_retrieval_en': 0.629940788348712, 'lcc_vs_2wikimqa': -0.09523809523809526, 'passage_retrieval_en_vs_2wikimqa': -0.25197631533948484}

## D. Exploratory depth / Key-vs-Value patterns

- Spearman(layer_idx, SignedKey) = -0.5952 (descriptive, 8 sampled layers only -- not a smooth-depth-trend claim)
- Spearman(layer_idx, SignedValue) = +0.1429
- early/mid/late bucket means (SignedKey): {'early_0_4_9': 0.43931979789216563, 'mid_13_18_22': -0.062106408140077875, 'late_27_31': 0.025624999999999787}
- early/mid/late bucket means (SignedValue): {'early_0_4_9': -0.28695833333333426, 'mid_13_18_22': 0.1351392390289433, 'late_27_31': 0.056291666666666185}
- Spearman(SignedKey, SignedValue) across 8 layers = -0.6190
- Median-split quadrants (exploratory): {0: 'high_key_high_value', 4: 'high_key_low_value', 9: 'low_key_low_value', 13: 'high_key_high_value', 18: 'high_key_high_value', 22: 'low_key_low_value', 27: 'low_key_high_value', 31: 'low_key_low_value'}

## What this analysis does NOT claim

- Does not claim a layer is universally sensitive based on only 4 screening tasks.
- Does not claim correlation implies mechanism (Spearman correlations here are descriptive, n=4 or n=8).
- Does not claim a CI crossing zero proves equivalence.
- Does not claim one-layer-at-a-time perturbation predicts multi-layer joint compression.
- Does not claim Phase-1 (the global 3x3 K/V matrix) found a real K x V interaction. The correct Phase-1 result is: K4/V2 - FP16 was the only configuration-level delta whose paired-bootstrap 95% CI excluded zero; all five explicit K x V difference-in-differences interaction CIs crossed zero.
