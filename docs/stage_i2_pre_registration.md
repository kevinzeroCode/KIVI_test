# Stage I2A-PRE: Layer x Quantizer-Family Sensitivity Pre-Registration

Status: DESIGN + INFRASTRUCTURE ONLY. No GPU has been run for Stage I2, no
downstream LongBench score has been collected under this experiment, no
bootstrap CI has been computed from real data. This document locks the
exact scientific question, condition/task matrix, aggregation rule,
bootstrap method, crossover/interpretation vocabulary, and staged
authorization plan BEFORE any Stage-I2 GPU execution, mirroring the
discipline of `docs/stage_h0_pre_registration.md` and
`docs/stage_g3b_pre_registration.md`.

## 0. Recap: production family semantics (unchanged, cited not re-derived)

`models/llama_kivi.py::_resolve_layer_kv_bits(config, layer_idx)` resolves
`(k_bits, v_bits, family)` per layer from `config.layer_kv_policy` (a list
produced by `utils.layer_policy.resolve_layer_policy`), falling back to
`(config.k_bits, config.v_bits, "kivi")` when no per-layer policy is set --
this fallback keeps every existing global run byte-for-byte unchanged.
`LlamaAttention_KIVI.__init__` sets `self.rotate_kv = self.family ==
"rotation_kivi"`; Value is never rotated regardless of family (rotation
touches only the Key path, and only in `LlamaFlashAttention_KIVI`, which is
what `pred_long_bench.py::build_model_and_tokenizer` always selects when
`use_kivi_model` and `config.use_flash = True`, which it always sets). A bit
width of 16 under either family is the existing FP16/pass-through route,
not a distinct quantizer.

## 1. Accepted upstream state (frozen, not re-litigated here)

```
ORIGINAL_I1_C0_RESULT = FAIL
C0_NUMERICAL_CAUSE = FP16_REPRESENTATION_ROUNDING
C0_V2_ENGINEERING_GATE = PASS

I1_C1_PATH_VALIDATION = PASS
I1_C1_DISTORTION_DIRECTION = HIGHER_FOR_ROTATION
I1_C1_GPU_CANARY = PASS

I2_LAYER_FAMILY_SENSITIVITY = AUTHORIZED (Stage I2A only, per this document's
staging plan; I2B/I2C/I2D each require their own separate authorization)
```

## 2. Scientific question -- LOCKED

> At a fixed K2/V16 KV-cache precision, does the relative downstream effect
> of Rotation-KIVI versus standard KIVI depend on decoder layer and
> workload?

```
Delta_family(layer, task) = score(rotation_kivi K2/V16, layer, task)
                           - score(kivi K2/V16, layer, task)
```

Positive: Rotation-KIVI has the higher LongBench score.
Negative: standard KIVI has the higher LongBench score.
**Positive is never defined as "significant improvement."** A positive
point estimate with a bootstrap CI that crosses zero is still positive --
"significant" language is reserved for the explicit CI-based statements in
Section 9/11.

## 3. Layers -- LOCKED

```
I2_LAYERS = (0, 4, 9, 13, 18, 22, 27, 31)
```

Identical set to the prior Stage D0 layer-sensitivity pilot's
`utils.pilot_policy.PILOT_LAYERS` (`utils/i2_layer_family_conditions.py`
asserts this is a deliberate reuse, not a coincidence). **No layer is
added, removed, or reselected after looking at I2 results.**

## 4. Quantizer conditions -- LOCKED

For every target layer, exactly two conditions:

| | target layer | all other layers |
|---|---|---|
| REFERENCE | `family=kivi`, K2/V16 | `family=kivi`, K16/V16 |
| CANDIDATE | `family=rotation_kivi`, K2/V16 | `family=kivi`, K16/V16 |

Global: `group_size=32`, `residual_length=128`. No Value quantization
anywhere. No K4. No Polar. No mixed-family multi-layer policy. One target
layer per condition only.

```
8 layers x 2 families = 16 conditions
```

Implemented in `utils/i2_layer_family_conditions.py` (mirrors
`utils/pilot_policy.py`'s design exactly, varying `family` instead of
`axis`): `build_policy_obj`, `all_i2_specs`, `write_i2_policies`,
`discover_and_validate_i2_policies` (which additionally asserts, per
condition, that the target layer resolves to exactly the requested
`(k_bits, v_bits, family)` and every other layer resolves to exactly
`(16, 16, "kivi")` -- the same assertion shape as
`scripts/h2_attention_decode_parity_canary.py::load_canary_model`'s
post-construction check, applied here at CPU-only policy-discovery time).
Each condition's stable identity is `policy_hash(resolved)`
(`utils/layer_policy.py::policy_hash`, a 12-hex-char sha256 of the fully
resolved per-layer bits -- hashes *behavior*, not the label), and
`output_dir_name(condition) = f"{condition_id}_{hash12}"`.

## 5. Task set -- LOCKED

The six already-established LongBench workloads, at their exact normal
full test-split counts (verified against
`analysis/analyze_kv_ablation.py::EXPECTED_TASK_COUNTS`, never re-derived
or hand-typed):

```
trec                  200
lcc                   500
passage_retrieval_en  200
2wikimqa              200
multifieldqa_en       150
samsum                200
                     ----
                     1450  samples/condition
```

```
16 conditions x 1450 samples/condition = 23,200 planned generation rows
```

No sub-sampling. lcc is not the only task used. Task-specific generation
semantics (prompt construction, `build_chat` skip-list, greedy decoding,
samsum's `min_length`/`eos_token_id` guard) are reused unchanged from
`utils/generation_semantics.py::resolve_generate_kwargs` and
`scripts/h2_attention_decode_parity_canary.py::build_prompt` -- never
reimplemented inside the I2 runner.

## 6. Model / dataset / generation identity -- LOCKED

```
model:              lmsys/longchat-7b-v1.5-32k
context limit:       31500
seed:                42
generation:          greedy, num_beams=1, do_sample=False, batch size=1
group_size:          32
residual_length:     128
```

**LongBench revision note**: no existing code path in this repository pins
a `revision=` kwarg to `load_dataset("THUDM/LongBench", ...)` (verified
across every call site: `pred_long_bench.py`,
`scripts/run_layer_sensitivity_pilot.py`,
`scripts/h2_attention_decode_parity_canary.py`,
`scripts/i1_rotation_kivi_parity_canary.py`, and others). The revision hash
`5e628be450b7e67fb7ae6e201bd6d8f7056f7672` referenced in this stage's
authorization is an *observed* fact recorded from a prior one-off smoke
test (`environment/README.md`, `EXPERIMENT_STATUS.md`), not something any
current runner pins. Stage I2C's formal launch command must pin this
revision explicitly (`load_dataset(..., revision=...)`) for full
reproducibility, since the installed `datasets` package version in this
environment (`3.6.0`) differs from `requirements.txt`'s stale pin
(`2.16.1`) -- this is flagged here as a Stage I2C launch-command
requirement, not silently assumed.

## 7. Primary outcome -- LOCKED

Primary outcome is **downstream LongBench task score**, computed with the
project's existing evaluator machinery
(`analysis/analyze_kv_ablation.py::sample_score` /
`compute_task_sample_scores`, which reproduce `eval_long_bench.py`'s
`scorer()` exactly -- imported, never reimplemented). Key decode distortion
(C1's measurement) is explicitly **not** the I2 primary outcome; C1 was a
path-validation measurement, not a scientific endpoint (Section 12).

```
Delta_family(layer, task) = Rotation score(layer, task) - KIVI score(layer, task)
```

## 8. Task aggregation -- LOCKED

```
Delta_family(layer) = UNWEIGHTED MEAN across the six task-level Delta_family(layer, task)
```

Each task gets equal weight regardless of sample count -- lcc's 500 samples
must not receive 2.5x the aggregate weight of a 200-sample task. Both
task-specific deltas and the equal-task aggregate are reported (never only
the aggregate). Implemented in
`analysis/analyze_i2_layer_family_sensitivity.py::compute_layer_aggregate_deltas`.

## 9. Paired statistics -- LOCKED

Reuses the project's existing paired-bootstrap primitives
(`analysis/analyze_kv_ablation.py::derive_seed`, `summarize_bootstrap`) and
the task-stratified shared-index resampling algorithm first established
there and generalized in
`analysis/analyze_layer_sensitivity_pilot.py::bootstrap_condition_scores` --
reimplemented once more in
`analysis/analyze_i2_layer_family_sensitivity.py` only because the config
universe differs (16 `(layer, family)` cells, not a fixed-FP16-baseline
axis matrix); the resampling algorithm itself (one shared per-task index
array, applied identically to every condition in that replicate) is
unchanged.

```
bootstrap repetitions = 10,000
seed = 42
```

**Pairing**: samples are paired within the SAME task, dataset index, and
layer between `kivi` and `rotation_kivi`. Pairing integrity is verified
BEFORE any CI calculation
(`analyze_i2_layer_family_sensitivity.py::validate_pairing`, which requires
the two families' `dataset_index` sets to match exactly at every
layer/task and their `answers`/`all_classes`/`length` fields to agree at
every shared index) -- fails closed, never silently reorders or drops rows.

For task-level family deltas: the two families' scores are resampled with
the SAME per-task index array per replicate (this is what makes the
resulting delta distribution paired). For the layer aggregate delta: the
six task-level bootstrap delta arrays are combined with equal task
weighting per replicate (never combined via a single pooled per-sample
resample).

Reported: point estimate + 95% bootstrap CI (2.5th/97.5th percentile of the
bootstrap distribution, `summarize_bootstrap`). **No p-value** is reported
(no existing project utility requires one for this analysis). A nonzero
point estimate is never described as "significant" by itself -- only a CI
that excludes zero is described as bootstrap-supported.

## 10. Layer-dependence / crossover analysis -- LOCKED, pre-registered before generation

**A. Layer delta vector**: `Delta_family(0), Delta_family(4), ...,
Delta_family(31)` -- the 8 equal-task-aggregate point estimates, each with
its own 95% bootstrap CI.

**B. STRONG_CROSSOVER** (exact definition, no other criterion may
substitute): call it `STRONG_CROSSOVER` if and only if at least one
pre-registered layer has a 95% CI entirely `> 0` **and** at least one
pre-registered layer has a 95% CI entirely `< 0`.

**C. MIXED_POINT_ESTIMATE_DIRECTION**: layer point estimates contain both
positive and negative values, even if their CIs cross zero -- reported
whenever this holds, but this is explicitly **not** `STRONG_CROSSOVER`.

**D. Family-effect range**: `max_layer Delta_family(layer) - min_layer
Delta_family(layer)`, reported descriptively. Its bootstrap CI is also
reported (`bootstrap_family_effect_range`, which lets the argmax/argmin
layer vary per bootstrap replicate -- the only valid way to bootstrap an
extremum statistic; a fixed pair of layers chosen from the observed point
estimate would reintroduce winner-selection bias). **No range threshold is
invented after seeing the results** -- the CI is reported unconditionally,
with no pass/fail gate attached to it.

**E. Task-conditioned behavior**: an 8 (layers) x 6 (tasks) matrix of
`Rotation - KIVI` score, since the prior Stage D0/E layer-sensitivity study
was strongly workload-dependent and I2 must check whether family behavior
is too. Implemented as `task_conditioned_matrix` and written to
`i2_family_delta_matrix.csv` (long form) / embedded in
`i2_interpretation.json` (nested form).

## 11. Interpretation categories -- LOCKED vocabulary, exact priority order

```
STRONG_CROSSOVER                 -- Section 10B's exact condition holds.
MIXED_POINT_ESTIMATE_DIRECTION   -- some layer point estimates > 0 and some < 0,
                                     but STRONG_CROSSOVER's condition is not met.
UNIFORM_ROTATION_ADVANTAGE       -- all 8 aggregate point estimates > 0.
                                     Does NOT by itself mean statistically
                                     established superiority.
UNIFORM_KIVI_ADVANTAGE           -- all 8 aggregate point estimates < 0.
                                     Does NOT by itself mean statistically
                                     established superiority.
NO_CLEAR_FAMILY_DIFFERENCE       -- results do not cleanly support any of the above.
```

Checked in exactly this priority order
(`analyze_i2_layer_family_sensitivity.py::classify_interpretation`):
`STRONG_CROSSOVER`'s defining condition always implies mixed point-estimate
direction too, so `STRONG_CROSSOVER` (the more specific claim) is reported
instead of the weaker `MIXED_POINT_ESTIMATE_DIRECTION` whenever it holds.
**No category is redefined after results are seen.**

## 12. C1 must not bias I2 -- LOCKED

C1 (Stage I1) observed, on `lcc`/dataset_index `122`/layer `0`/K2-V16 only,
that Rotation-KIVI had higher Key decode distortion than standard KIVI.
Explicitly:

- C1 distortion is **not** an I2 endpoint (Section 7).
- C1's sample 122 does **not** determine I2's task selection (Section 5).
- C1's direction does **not** determine any expected I2 downstream
  direction (Section 2/11).
- **No layer is removed** from `I2_LAYERS` (Section 3) because of C1.

`utils/i2_layer_family_conditions.py` and
`analysis/analyze_i2_layer_family_sensitivity.py` each carry this statement
verbatim (`C1_NON_SELECTION_NOTE`) for provenance, but no computation in
either module reads or conditions on it -- it is recorded, never consumed.

## 13. Memory-budget fairness -- LOCKED

Both families use the same `K2`, `V16`, `group_size=32`,
`residual_length=128` at the target layer. Rotation-KIVI adds no per-token
persistent KV-cache metadata relative to standard KIVI: the deterministic
Hadamard matrix (`utils.hadamard.get_normalized_hadamard`) is
context-independent static runtime/model state, not per-token KV cache, and
is tracked separately if systems cost is ever reported. **This is not a
claim of zero total overhead** -- only that the two families' per-token
cache footprint at matched K/V bits is identical; the static Hadamard
matrix and the rotation compute itself are real, separately-trackable
costs.

## 14. Formal runner architecture

`scripts/run_i2_layer_family_sensitivity.py` mirrors
`scripts/run_layer_sensitivity_pilot.py`'s already-validated architecture,
generalized from `(layer, axis)` to `(layer, family)`, reusing rather than
reimplementing:

- model loading: `pred_long_bench.py::build_model_and_tokenizer` with a
  per-condition `resolved_layer_policy`.
- task prompt construction + generation semantics:
  `utils.generation_semantics.resolve_generate_kwargs` /
  `NO_BUILD_CHAT_DATASETS`, `pred_long_bench.py::build_chat`,
  `pred_long_bench.py`'s truncation logic.
- exact-count resume logic: a per-script copy of
  `resolve_pilot_task_resume_plan`'s exact-count-only contract
  (`resolve_i2_task_resume_plan` -- a final file must have *exactly* the
  expected row count, never `>=`; any invalid row, or an over-count
  partial, fails closed).
- file locking: `flock(LOCK_EX | LOCK_NB)`, copied per this repo's
  established per-script convention (no shared lock util exists).
- host/GPU monitoring: `get_boot_id`/`get_git_commit`/
  `get_git_status_short`/`gpu_snapshot`/`gpu_preflight`/
  `check_no_conflicting_process`, copied per the same convention.
- manifest/provenance: per-condition `run_config.json` (resume-identity
  keys include `policy_hash`) + per-condition `manifest.json`
  (boot_id/exit_code/git_commit) + a top-level `manifest.json`-style
  pilot-wide manifest with per-condition `status`, written atomically
  (temp file + `fsync` + `os.replace`).
- dataset revision handling: not yet pinned anywhere in this repo (Section
  6) -- an explicit gap flagged for the I2C launch command, not silently
  assumed resolved.

Requires `--run` to execute anything beyond policy generation; without it
(the default), both `--mode generation` and `--mode canary` only print a
CPU-only dry-run report and exit 0 -- torch/transformers/datasets are never
imported. Even with `--run`, this module refuses (`I2ConfigError` /
`I2CanaryError`) to actually execute GPU generation or the I2B canary,
because Stage I2A's task scope is infrastructure/preregistration only; a
future stage removes that refusal once I2B/I2C are separately authorized.

## 15. Condition identities

Each condition's stable identity: `target_layer`, `family`, `k_bits=2`,
`v_bits=16`, `group_size=32`, `residual_length=128`, plus the resolved
`policy_hash`. The manifest records `HEAD`, `policy_hash`, `model`,
`task`, `expected count`, `actual count` (via `run_config.json` +
per-task `.jsonl` row counts), `seed`, generation settings, start/end
timestamp, `exit_code`, and `boot_id` (start/end + stability flag).

## 16. Output root / isolation

```
outputs/i2_layer_family_sensitivity/
    <condition_id>_<policy_hash12>/
        run_config.json
        manifest.json
        host_monitor.log
        <task>.jsonl
```

Mirrors the flat per-condition layout `outputs/layer_sensitivity_pilot/`
already uses in practice (no nested generation/manifests/logs/analysis
split at the output-root level; analysis output is a fully separate tree,
`analysis/results/i2_layer_family_sensitivity/`). Never writes into
`outputs/layer_sensitivity_pilot/`, `outputs/layer_attention_feature_pilot/`,
`outputs/i1_rotation_kivi_canary/`, or `pred/`.

## 17. Resume / completeness

A condition/task unit is complete only if: `actual row count == expected
row count` (exact match, never `>=`), `manifest exit_code == 0`, no
malformed JSONL row (`utils.jsonl_integrity.inspect_jsonl`), and every
expected `dataset_index` is present exactly once (no duplicate, no
missing -- `analyze_i2_layer_family_sensitivity.py::validate_index_integrity`,
which additionally rejects a short/partial row list even if somehow not
already caught by the JSONL-row-count check). Partial output is never
classified complete. Single-instance safety via `flock`.

## 18. Runtime K2 cache-proof design (I2B) -- closes the C1 reporting gap

Before formal generation, the eventual I2 execution stage must verify at
runtime, for the target layer: requested family, requested K2/V16; for all
non-target layers: `family=kivi`, K16/V16. **For K2 routes, a real
quantized Key prefix must actually be exercised** for an appropriately
long-enough canary -- C1's own canary never established this (its
6-token canary never ran long enough to trigger a Key-cache rollover, which
fires in bulk at a ~114-step cadence per `residual_length=128` --
`docs/stage_h0_pre_registration.md` Section 0). I2B is designed precisely
to close this gap; heavy cache capture is never added to the 23,200 formal
generations themselves.

Recorded per canary observation: `key_quant_trans` presence and shape,
`key_full` residual shape, `key_scale` shape, `key_mn` shape (via
`utils.attention_decode_features.parse_kivi_cache_tuple`, never
reimplemented).

## 19. I2 execution staging -- LOCKED

```
I2A (this document + this task): infrastructure, preregistration, tests. NO GPU.
I2B (separate authorization required): small GPU execution canary.
I2C (separate authorization required, only if I2B passes):
     formal 16-condition x 6-task generation (23,200 rows).
I2D (separate authorization required): paired evaluation + bootstrap analysis.
```

I2A does not jump directly to I2C. Neither I2B nor I2C is executed by this
task.

## 20. I2B canary design -- IMPLEMENTED, NOT RUN

`scripts/run_i2_layer_family_sensitivity.py::run_i2b_canary_condition` /
`i2b_canary_conditions` / `compute_i2b_canary_gate`. Exercises, at minimum:

```
layers:   0, 18, 31      (a strict subset of I2_LAYERS -- first, middle, last)
families: kivi, rotation_kivi
bits:     K2 / V16
```

using one already-fixed sample from a long-context task (`lcc`,
`dataset_index=122` -- the same already-vetted calibration sample the
H2/I1 canaries use; reused specifically *because* it is long-context, not
because of any connection to C1's use of the same index for a different
purpose -- see Section 12) solely for runtime path validation, **not** for
efficacy conclusions. `CANARY_MAX_NEW_TOKENS = 140` is deliberately longer
than `residual_length` (128) so a real Key-cache rollover has a chance to
fire, unlike the H2/C1 canaries' 6-token generations.

Per-condition gate (`compute_i2b_canary_gate`, a pure CPU-testable function
mirroring `scripts/i1_rotation_kivi_parity_canary.py::compute_c0_structural_gate`'s
discipline), all required True:

```
policy_resolved_correctly   -- target layer + all-others assertion (reused from
                                scripts/h2_attention_decode_parity_canary.py::load_canary_model)
quantized_prefix_observed   -- key_quant_trans is not None at >=1 captured decode step
                                (closes the C1 reporting gap -- Section 18)
cache_shapes_valid          -- key_quant_trans/key_scale/key_mn leading-dim agreement
generation_finite           -- generated token ids are well-formed
hook_neutral                -- hooks-on generation == hooks-off generation
ran_without_error           -- no uncaught exception during model load/generate
```

`quantized_prefix_observed = False` in a real run must be reported as an
**inconclusive/blocked** canary for that condition, never silently treated
as a pass. Rotation is **not** required to score better or produce the
same generated tokens as standard KIVI -- this canary answers a structural
question only.

## 21. Evaluation / analysis outputs

`analysis/analyze_i2_layer_family_sensitivity.py` consumes the I2
generation root and produces:

```
i2_task_score_matrix.csv     layer, family, task, score
i2_family_delta_matrix.csv   layer, task, kivi_score, rotation_score, delta
i2_layer_aggregate.csv       layer, delta_family_layer, ci_low, ci_high
i2_bootstrap_ci.csv          contrast, layer, task, observed, ci_low, ci_high, ci_excludes_zero
i2_pairing_audit.json        pairing validation result (fails closed before analysis if broken)
i2_interpretation.json       full summary: question, point estimates, CIs, task-conditioned
                              matrix, family-effect range, interpretation category, C1 note
```

All derived results are reproducible from the raw per-condition JSONL
generation outputs alone (no manual score copying) -- scores are
re-derived via `analysis.analyze_kv_ablation.sample_score` /
`compute_task_sample_scores`, the same functions `eval_long_bench.py`
itself is built on.

## 22. Stop conditions

- If runtime policy validation (I2B) finds any condition's resolved policy
  does not match the requested target-layer/all-others contract -> BLOCK,
  do not proceed to I2C.
- If a K2 route's canary never observes a real quantized Key prefix
  (`quantized_prefix_observed = False`) -> that condition's I2B result is
  inconclusive; formal generation for the corresponding real I2C
  conditions must not proceed on the assumption the K2 route was proven.
- If measurement hooks (when used) alter generated tokens/logits -> BLOCK.
- If pairing integrity fails at analysis time (Section 9) -> BLOCK, fix the
  underlying data before any CI is computed or reported.
- The metric formulas (Section 2/7/8), bootstrap method (Section 9),
  crossover definition (Section 10B), and interpretation categories
  (Section 11) are not to be weakened, redefined, or have a new threshold
  invented after seeing I2C results.
- I2A itself must not touch `models/llama_kivi.py`, `utils/hadamard.py`,
  `quant/new_pack.py`, or `quant/matmul.py` unless an actual correctness
  defect is discovered during infrastructure construction, in which case
  work stops and the defect is reported before any production-code edit.
