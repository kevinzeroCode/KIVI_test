# Stage G3B-PRE: Scientific Feature Pilot Pre-Registration

Status: DESIGN ONLY. No GPU forward pass, no scientific feature record, no
correlation has been computed as of this document's creation. This document
is the pre-registration artifact itself -- it exists specifically so this
gate is never again unrecoverable the way the original Stage-G0 gate turned
out to be (see Section 0).

## 0. Recovery of the Stage-G0 decision gate

Searched: every `*.md`/`*.py`/`*.json` file in the repository (excluding
`.venv`, `pred/`, `outputs/`), `EXPERIMENT_STATUS.md`, `docs/`,
`analysis/results/*/report.md` and `summary.json`, and `git log --all` for
any commit message mentioning "feature", "G0", or "gate".

**Result: NOT RECOVERED.** The only persisted Stage-G0 artifact anywhere is
the verdict `FEATURE_PILOT_INFRASTRUCTURE = GO` (referenced only in prose
docstrings in `utils/feature_extraction.py` / `scripts/collect_layer_features.py`,
never as a standalone file) -- and that gate answered a different, narrower
question ("should CPU feature-extraction infrastructure be built at all?"),
not "what evidence would justify treating a feature/sensitivity association
as real?". No numeric threshold, no correlation-direction hypothesis, and no
GO/NO-GO structure for the scientific question was ever written down.

Per instruction, GPU collection does not proceed on the basis of a
recovered gate. Instead, Section 13 below defines a new gate now, before any
feature data exists, explicitly marked as newly pre-registered (not a
recovered Stage-G0 artifact).

## 1. Primary labeled observations (Stage-E, already finalized -- not re-collected)

Source: `analysis/results/layer_sensitivity_pilot/task_layer_sensitivity.csv`
and `summary.json` (Stage E, generated 2026-08-23). Sensitivity labels are
`delta = perturbed_score - baseline_score` per (layer, axis, task); positive
= quantizing that layer/axis IMPROVED the score vs FP16 (less harm),
negative = HARMED it. `SignedKeySensitivity(layer, task)` = the `key`-axis
delta; `SignedValueSensitivity(layer, task)` = the `value`-axis delta;
`AxisDifference(layer, task) = SignedKeySensitivity - SignedValueSensitivity`.

8 layers: 0, 4, 9, 13, 18, 22, 27, 31.
4 tasks: trec, lcc, passage_retrieval_en, 2wikimqa.
-> 32 Key labels + 32 Value labels, reused verbatim -- no relabeling, no
re-running Stage E.

## 2. F0 diagnostic tasks (Layer-0-only labels)

multifieldqa_en, samsum -- from `analysis/results/layer_sensitivity_f0/report.md`
(Stage F0, generated 2026-08-24). Only Layer 0 was probed for these two
tasks, so they are NOT part of the primary 8-layer correlation dataset.
Known Layer-0 labels (already finalized):

| task | KeyDelta(L0) | ValueDelta(L0) | AxisDifference(L0) |
|---|---|---|---|
| lcc (primary task, restated) | +2.64 | -4.17 | +6.81 |
| multifieldqa_en | -0.1072 | +0.0249 | -0.1322 |
| samsum | +0.0646 | -0.0680 | +0.1326 |

lcc shows large-magnitude, same-signed-as-hypothesis separation;
multifieldqa_en is small and OPPOSITE in sign on AxisDifference; samsum is
small and same-signed but ~50x smaller in magnitude than lcc. These are the
target response values Section 12's diagnostic will compare feature
summaries against -- purely descriptively, no significance claim from n=3.

## 3. Calibration sample selection (Part 4) -- executed as a real CPU dry-run

Method: deterministic, length-stratified, metadata-only. For each task,
every example's LongBench-provided `length` field (a precomputed metadata
field, NOT a tokenizer count under this model -- exact token count is only
known at real collection time and will be recorded then) is sorted by
`(length, original_index)` ascending; the 20th/40th/60th/80th percentile
positions (nearest-rank, `round(p/100 * (n-1))`, clamped) are selected, with
collision-avoidance search if two percentiles would land on the same
position. No generated prediction, sensitivity value, or feature value is
used. Implemented as `select_calibration_indices()` /
`percentile_rank_index()` / `load_task_lengths()` in
`scripts/collect_layer_features.py` (CPU-only aside from the `datasets`
library metadata read; no torch/transformers/model/GPU).

Executed for real via `--preview-calibration` (see Section 15 for the exact
command). Results below are the actual pre-registered sample set:

| task | p20 idx / len | p40 idx / len | p60 idx / len | p80 idx / len |
|---|---|---|---|---|
| trec | 153 / 2892 | 154 / 4544 | 82 / 5916 | 151 / 7466 |
| lcc | 122 / 650 | 174 / 793 | 214 / 1027 | 357 / 1528 |
| passage_retrieval_en | 141 / 8559 | 185 / 9032 | 105 / 9446 | 41 / 10004 |
| 2wikimqa | 164 / 3131 | 44 / 3859 | 138 / 4580 | 38 / 6416 |
| multifieldqa_en | 50 / 2088 | 103 / 3590 | 81 / 5567 | 92 / 6706 |
| samsum | 58 / 3135 | 118 / 5133 | 151 / 7157 | 52 / 9449 |

No within-task duplicate-sample collisions. `batch_size=1` throughout (no
padding, so `distribution_tokens`/`quantized_tokens`/`residual_tokens`
semantics stay unambiguous per Stage-G3A-hardening's schema).

## 4. Collection scope

Primary (Stage-E labeled): 4 tasks x 4 samples x 8 layers x 2 axes = **256 records**.
F0 diagnostic (Layer-0 only): 2 tasks x 4 samples x 1 layer x 2 axes = **16 records**.
**Total: 272 records.** Confirmed by real dry-run execution (Section 15).

No other layers collected. No full tensors written to disk (matches the
existing collector design: only scalar feature records are persisted).

## 5. Sample -> task/layer/axis aggregation (pre-registered)

Primary: **equal-sample arithmetic mean** across the 4 fixed calibration
samples for a given (task, layer, axis) -- NOT token-length-weighted (a
short sample and a long sample contribute equally to the summary).
Secondary robustness check: **median** across the same 4 samples. Both will
be reported when used; the gate in Section 13 is evaluated on the mean only,
pre-registered now, not switched to median post hoc.

## 6. Primary vs secondary features (pre-registered now; no prior Stage-G0 fixing found)

**Primary, KIVI-specific (reconstruction):**
- Key `relative_l2` vs `SignedKeySensitivity`
- Value `relative_l2` vs `SignedValueSensitivity`

**Primary, family-agnostic (distribution):**
- `p99_abs` (per axis) vs the corresponding axis's `SignedSensitivity`
- `outlier_fraction` (per axis) vs the corresponding axis's `SignedSensitivity`

**Secondary/descriptive only** (not used in the Section 13 gate unless the
primary features fail to resolve anything and a follow-up round explicitly
re-opens this): `mean`, `std`, `variance`, `max_abs`, `p50_abs`, `p95_abs`,
`mse`, `max_abs_error`. No feature may be promoted to primary after its
correlation is observed.

## 7. Response targets

Primary: `SignedKeySensitivity(layer, task)`, `SignedValueSensitivity(layer, task)`.
Secondary: `AbsKeySensitivity = |SignedKeySensitivity|`, `AbsValueSensitivity = |SignedValueSensitivity|`.
`AxisDifference` is analyzed only against the matching feature-side contrast
`KeyRelativeL2 - ValueRelativeL2` (Section 9), never against a single-axis
feature alone.

## 8. Directional hypotheses (pre-registered NOW, before any feature data exists)

- **H1 (Key reconstruction):** higher Key `relative_l2` (worse 2-bit
  reconstruction) is associated with MORE HARM, i.e. a MORE NEGATIVE
  `SignedKeySensitivity` -> expected Spearman **rho < 0**.
- **H2 (Value reconstruction):** same logic -> expected Spearman **rho < 0**
  between Value `relative_l2` and `SignedValueSensitivity`.
- **H3 (family-agnostic tail mass):** higher `p99_abs` / `outlier_fraction`
  in the real FP16 representation (heavier tails / more outliers) gives a
  2-bit min-max quantizer more range to compress, plausibly producing WORSE
  reconstruction and more harm -> expected Spearman **rho < 0** against the
  corresponding axis's `SignedSensitivity`.
- **H4 (axis contrast, secondary):** if Key has relatively more
  reconstruction error than Value at a layer (`KeyRelativeL2 - ValueRelativeL2`
  large positive), Key should be relatively more harmed than Value, pushing
  `AxisDifference = SignedKey - SignedValue` MORE NEGATIVE -> expected
  Spearman **rho < 0**. (Note this is a direction hypothesis derived from
  H1/H2's logic, not yet checked against any data.)

These directions are fixed before data collection. A result in the opposite
direction is a valid, reportable outcome -- not evidence of a design error.

## 9. Per-task analysis plan (primary; n=8 layers per task, 4 tasks)

For each of trec, lcc, passage_retrieval_en, 2wikimqa **separately**:
Spearman(Key relative_l2 across 8 layers, SignedKeySensitivity across 8
layers); Spearman(Value relative_l2, SignedValueSensitivity); Spearman(Key
p99_abs, SignedKeySensitivity); Spearman(Key outlier_fraction,
SignedKeySensitivity); and the same two for Value. Report each task's rho
separately (8 numbers per task, never averaged blindly across layers within
a task in place of the correlation). Then, across the 4 task-level rho
values for each feature/target pair: sign consistency (how many of 4 agree
with the pre-registered direction), median rho, and range (min, max). 32
pooled observations across tasks/layers are explicitly NOT treated as
independent (they share layers and tasks) -- this is why the primary
analysis is per-task, not one pooled n=32 correlation.

## 10. Optional pooled descriptive analysis (secondary only)

If reported: within-task normalize (z-score) both the feature values and
the sensitivity values before pooling to n=32, to avoid one task's absolute
scale dominating. Label explicitly as "pooled, descriptive, non-independent
observations" in any output. Pooled results may never override an
inconsistent per-task Section 9 result in the Section 13 gate.

## 11. Leave-one-task-out (LOTO) plan

For every aggregate statistic used in the Section 13 gate (sign-consistency
count, median rho), recompute it 4 times, each time excluding one of
{trec, lcc, passage_retrieval_en, 2wikimqa}. Report: does the direction
conclusion flip; does removing lcc specifically (Stage F0 already showed
lcc-specific fragility for the Layer-0 signed contrasts) change the
conclusion; does any single task dominate. This directly feeds gate
criterion C.

## 12. Layer-0 task-conditioning diagnostic (secondary, descriptive)

At Layer 0 only, compare feature summaries (Key/Value relative_l2, p99_abs,
outlier_fraction) across lcc vs multifieldqa_en vs samsum. Ask
descriptively: do reconstruction/distribution features separate lcc from
the two near-null tasks in the direction consistent with Section 2's known
response magnitudes (lcc: large separation; multifieldqa_en: small/opposite;
samsum: small/same-sign)? No significance test from n=3; qualitative
consistency only. This diagnostic is secondary to Section 9 and feeds gate
criterion D.

## 13. Pre-registered decision gate (NEWLY DEFINED -- Stage-G0's original gate was not recoverable, see Section 0; HARDENED in the Stage-G3B-IMPL round; undefined-Spearman semantics added in the Stage-G3B-IMPL-2 round -- superseding all earlier drafts verbatim)

### 13.0 Undefined-Spearman convention (defined BEFORE feature collection)

Known pre-existing Stage-E fact: **trec's Value SignedSensitivity is
constant (zero variance) across the 8 sampled layers**, so its
task-specific Spearman correlation is mathematically undefined -- Spearman
rho is not defined when either input vector has zero variance (a
ranking-based correlation coefficient requires both vectors to actually
vary). This must be handled by a fixed, pre-registered rule, not decided
ad hoc after seeing which task hits it.

**RAW REPORTING.** If either vector in a task-specific (or Layer-0
3-task) Spearman calculation has zero variance, the result is reported as:

```
raw_rho = undefined
```

`raw_rho=undefined` (represented as `None` in
`analysis/feature_pilot_gate.py`) is NEVER reported or treated as an
observed `rho=0` -- "the correlation could not be computed" and "the
correlation was computed and found to be exactly zero" are different,
non-interchangeable claims, and conflating them would misrepresent the
evidence.

**GATE EVALUATION.** For deterministic, conservative gate evaluation
ONLY:

```
gate_rho = 0.0   whenever raw_rho is undefined
gate_rho = raw_rho   otherwise
```

Interpretation: an undefined correlation contributes **no evidence** for
the hypothesized negative association. `gate_rho=0.0` can never itself
satisfy a `rho < 0` direction check, and it pulls a magnitude median
toward zero rather than being silently dropped from that median (dropping
it would shrink n and could inflate the median -- substituting 0 is the
conservative choice). This rule is applied identically to Criteria A, B,
C, and D -- never selectively.

### 13.1 The gate itself

**FEATURE_PILOT_STAGE = GO** only if, for at least ONE tensor axis (Key or
Value), the SAME axis satisfies ALL of:

- **A (direction):** `gate_rho < 0` for relative_l2-vs-SignedSensitivity
  (H1/H2 direction) in at least 3 of the 4 Stage-E tasks (Section 9). An
  undefined `raw_rho` (e.g. trec's Value case) maps to `gate_rho=0.0` and
  therefore does NOT count as negative.

- **B (magnitude):** `median(abs(gate_rho))` computed over exactly the 4
  Stage-E tasks is >= **0.5**. This 0.5 is **a pre-registered pragmatic
  magnitude threshold**, fixed now, before any feature data exists -- not
  a universal statistical convention and not a Spearman-specific
  convention; it is simply a specific number chosen in advance so it
  cannot be adjusted after seeing results. An undefined task contributes
  `gate_rho=0.0` to this median and is NEVER silently dropped from the
  4-element set.

- **C (LOTO robustness, hardened -- exact wording):** after removing lcc,
  the remaining tasks are trec, passage_retrieval_en, 2wikimqa. Using the
  same `gate_rho` convention: (i) at least 2 of these 3 `gate_rho` values
  are < 0, AND (ii) the median of `gate_rho` across exactly those 3
  remaining tasks is < 0. An undefined task among the remaining 3 (e.g.
  trec) is never dropped -- it contributes `gate_rho=0.0` to both the
  count and the median. This criterion never requires 3-of-3.

- **D (Layer-0 diagnostic, hardened -- exact wording):** using Layer 0
  only and the three tasks lcc, multifieldqa_en, samsum, the Spearman
  correlation across these 3 task-conditioned observations between
  relative_l2(task) and SignedSensitivity(task) for that SAME axis has
  `gate_rho < 0`. No significance threshold is required or applicable
  (n=3). If either n=3 vector has zero variance, `raw_rho` is undefined ->
  `gate_rho=0.0` -> **this criterion FAILS** (0 is not < 0) -- undefined
  can never pass Criterion D by default. Distribution features
  (p99_abs/outlier_fraction) must never be substituted for relative_l2 in
  this criterion after seeing data.

**FEATURE_PILOT_STAGE = NO_GO** otherwise. No `CONDITIONAL_GO` option --
Stage G0 never pre-registered one, so none is available here.

The primary family-agnostic distribution features (`p99_abs`,
`outlier_fraction`, Section 6) remain separately reported primary
exploratory features in all analysis output, but they do NOT rescue a
reconstruction-error (relative_l2) gate that fails criteria A-D above --
there is no fallback path from a failed Key/Value relative_l2 gate to a
distribution-feature-based GO.

Implemented exactly as `analysis/feature_pilot_gate.py`
(`to_gate_rho` / `evaluate_feature_pilot_gate` / `evaluate_axis_gate` /
`evaluate_criterion_a..d`), covered by CPU-only unit tests in
`tests/test_feature_pilot_gate.py`, including a dedicated regression class
(`KnownTrecValueUndefinedCaseTest`) for the known trec-Value constant-
sensitivity case, using synthetic correlation inputs (no real feature data
exists yet). Reuses the project's one Spearman implementation
(`analysis.analyze_layer_sensitivity_pilot.spearman_corr`) rather than
reimplementing correlation math -- `spearman_corr` already returns `None`
for a zero-variance input, which is exactly `raw_rho=undefined`.

## 14. Cost estimate (approximate, not exact)

Measured G3A timings: model load ~96s; one 18,062-token forward pass ~7.7s;
peak GPU memory ~28.8GB for 2 probed layers.

- Model is loaded **once** per collector invocation; all 24 distinct
  samples (16 primary + 8 F0-diagnostic) are processed sequentially under
  that one loaded model -- no repeated ~96s reload cost.
- Calibration sample lengths (Section 3) range ~650-10,004 tokens, all
  shorter than G3A's 18,062-token canary, so per-sample forward time should
  be lower than 7.7s each; rough estimate 2-8s per forward pass depending on
  length (assume ~5s average).
- 8-layer capture (vs G3A's 2-layer canary) multiplies hook/clone overhead
  ~4x, but calibration samples are also shorter, so per-tensor memory
  roughly offsets; expect peak memory in the same tens-of-GB range as G3A
  (roughly 29-35GB), not a qualitative jump.
- CPU percentile aggregation: `torch.quantile`'s ~2^24-element ceiling
  (Stage-G3A-hardening fix) is hit only when `32 heads x head_dim x tokens`
  exceeds ~16.7M elements, i.e. roughly tokens > ~4,000 for this model's
  shape -- true for passage_retrieval_en/multifieldqa_en/samsum samples but
  not for most trec/lcc/2wikimqa samples. Each triggered fallback
  (`numpy.percentile` on a CPU copy) adds on the order of a few seconds;
  expect this to add tens of seconds of extra CPU-bound time in total, not
  a dominant cost.
- **Overall estimate: model load (~96s) + 24 forward passes (~2-8 min
  total) + modest CPU stat overhead (tens of seconds) -> roughly 5-10
  minutes wall-clock for the full 272-record collection.** This is a rough
  estimate, not a committed number; the actual run will be timed and
  reported exactly.

## 15. Dry-run (executed for real this round)

```
CUDA_VISIBLE_DEVICES="" ./.venv/bin/python scripts/collect_layer_features.py \
    --preview-calibration --output-root outputs/layer_feature_pilot
```

No torch/transformers/model/GPU import occurs (confirmed); only the
`datasets` library is used, to read each example's precomputed `length`
metadata field. Output: the table in Section 3, 256+16=272 expected
records, zero within-task collisions, output root
`outputs/layer_feature_pilot/` (not yet created), and the existing
conflicting-process check (one observed cosmetic false positive this run --
the check matched its own `timeout <N> python ...` wrapper process, a
variant of the already-documented shell-wrapper false-positive class in
`check_no_conflicting_process`'s docstring; not a real conflict, confirmed
no other genuine invocation was running).

## 16. Future output / provenance plan

`run_config.json` (git commit, model, layers, axes, group_size,
residual_length, seed), `sample_selection.json` (one entry per selected
sample: `task`, `dataset_index`, `selection_percentile`, `selection_rank`,
`selection_length` -- the LongBench metadata field used for pre-registration
selection, Section 3's table in machine-readable form), `manifest.json`
(per-sample **`input_tokens`** -- the actual tokenized model input length
after prompt formatting/truncation, deliberately never conflated with
`sample_selection.json`'s `selection_length` -- plus boot IDs, exit status),
`features.jsonl` (272 rows total across both scopes, schema per
`utils/feature_schema.py`), `host_monitor.log`. No full activation tensors
stored. All under `outputs/layer_feature_pilot/` -- never `pred/`,
`outputs/layer_sensitivity_pilot/`, `outputs/layer_sensitivity_f0/`, or
`analysis/results/`.

**Stage-G3B-IMPL wiring status (this round): DONE.**
`scripts/collect_layer_features.py::run_real_collection` now selects samples
via the exact same `select_calibration_indices(extract_length_pairs(data))`
call previewed by `--preview-calibration` (verified identical by
`tests/test_collect_layer_features.py::PreviewVersusRealSelectionParityTest`,
run against the real cached LongBench data) -- first-N indexing
(`range(n)`) has been removed from the real-collection path entirely and is
guarded against regressing via a source-inspection test. A new
`validate_task_layer_scope()` guard raises before model load if a
diagnostic task is paired with anything other than exactly `--layers 0`, if
a primary task is paired with anything other than the exact 8-layer set,
or if primary and diagnostic tasks are mixed in one invocation.

## Future GPU collection commands (NOT executed this round)

Two separate invocations under one experiment root, per Section 7 (the
current CLI expresses one `--layers` value per invocation, and primary vs.
diagnostic tasks require different layer sets). Every argument name below
was verified against `scripts/collect_layer_features.py --help` (Stage
G3B-IMPL-2 Part 7) -- none invented. `--num-samples 4` and `--seed 42` are
pinned explicitly (never relying on `--num-samples`'s generic default of
20, which is NOT the pre-registered design and, since Stage G3B-IMPL,
controls the number of length-percentile points selected, not a first-N
cap); `--model_name_or_path`, `--max-length`, `--group-size`, and
`--residual-length` are pinned explicitly even though their defaults
already match, for full reproducibility from the command alone:

```
# Primary scope: 4 tasks x 4 samples x 8 layers x 2 axes = 256 records
CUDA_VISIBLE_DEVICES=0 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    ./.venv/bin/python scripts/collect_layer_features.py \
        --tasks trec lcc passage_retrieval_en 2wikimqa \
        --layers 0 4 9 13 18 22 27 31 --axes key value \
        --num-samples 4 --seed 42 \
        --model_name_or_path lmsys/longchat-7b-v1.5-32k \
        --max-length 31500 --group-size 32 --residual-length 128 \
        --output-root outputs/layer_feature_pilot/primary --run_label <label>

# Diagnostic scope: 2 tasks x 4 samples x 1 layer x 2 axes = 16 records
CUDA_VISIBLE_DEVICES=0 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    ./.venv/bin/python scripts/collect_layer_features.py \
        --tasks multifieldqa_en samsum \
        --layers 0 --axes key value \
        --num-samples 4 --seed 42 \
        --model_name_or_path lmsys/longchat-7b-v1.5-32k \
        --max-length 31500 --group-size 32 --residual-length 128 \
        --output-root outputs/layer_feature_pilot/diagnostic --run_label <label>
```

`validate_task_layer_scope()` will refuse to run either command with the
wrong layer set for its task group, or a mix of both groups in one
invocation. `--num-samples 4` was verified (Stage G3B-IMPL-2 Part 8, CPU
preview only) to reproduce the exact persisted sample-selection table in
Section 3 byte-for-byte via `calibration_percentiles_for_sample_count(4) ==
DEFAULT_CALIBRATION_PERCENTILES == (20, 40, 60, 80)`.
