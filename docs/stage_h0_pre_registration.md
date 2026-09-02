# Stage H0-PRE: Decode-Aware Attention Feature Pre-Registration

Status: DESIGN ONLY. No GPU has been run for Stage H, no attention-aware
feature has been collected, no correlation computed. This document locks
the exact algorithm, gate, and cost accounting BEFORE any Stage-H
implementation or collection, mirroring the discipline of
`docs/stage_g3b_pre_registration.md`.

## 0. Corrected production semantics (persisted verbatim from Stage H0-DECODE-AUDIT)

**Prefill** (`models/llama_kivi.py`, `past_key_value is None` branch):
Q/K/V projected -> RoPE (full precision) -> FlashAttention runs on the
full-precision, UNQUANTIZED Q/K/V -> `attn_output` computed -> **only
after** `attn_output` exists does the code quantize K/V for storage into
`past_key_value`. No prefill query ever consumes quantized K/V. Confirmed
by direct source read, not assumed.

**Decode** (`past_key_value is not None` branch, `q_len=1`): current Q/K/V
projected + RoPE -> `att_qkquant = cuda_bmm_fA_qB_outer(...)` (fused
dequant+matmul against the static quantized Key prefix, unscaled) ->
`key_states_full = cat([key_states_full, key_states_current])` (FP16
residual grows) -> `att_qkfull = Q_t @ key_states_full^T` (unscaled) ->
`attn_weights = cat([att_qkquant, att_qkfull], dim=-1) / sqrt(head_dim)`
(quantized columns first, full-precision columns last) -> Key rollover
only when the FP16 residual buffer hits exactly `residual_length` (bulk,
~114-step cadence after a typical prefill) -> softmax -> analogous Value
computation, but Value rollover is per-token/FIFO and fires on **every**
decode step once `value_full_length > residual_length` (which is true from
decode step 1 onward for any T > residual_length).

**Chronology**: prefill forward -> generated token #1 (0% KIVI-cache
exposure) -> cached decode step 1 (`past_key_value != None`) -> generated
token #2 (**first** token whose prediction the KIVI cache actually
influenced).

## 1. Scientific question

Does the actual production KIVI cache, on the actual generated-token
trajectory it itself produced, measurably perturb the local attention
computation of real cached decode queries -- and does that perturbation,
summarized per (task, layer, axis), associate with the existing Stage-E
`SignedKeySensitivity`/`SignedValueSensitivity` labels in the
pre-registered direction?

## 2. Trajectory -- LOCKED

**Production-KIVI trajectory.** For each Stage-E policy condition, run the
actual KIVI model, use its own actual generated tokens, capture the actual
decode `Q_t`, and maintain a measurement-only FP16 shadow cache built from
the SAME token history (never fed back into the model). A fixed
FP16-reference trajectory is explicitly reserved for a later, separate
cross-family (KIVI / Rotation-KIVI / Polar) comparison -- not used here.
Not chosen based on any observed correlation.

## 3. Policy matching -- LOCKED

Every feature observation uses the SAME single-layer policy that produced
its Stage-E label:
- Key feature collection: probed layer = K2/V16, all other layers = FP16
  baseline (exactly Stage E's Key condition).
- Value feature collection: probed layer = K16/V2, all other layers = FP16
  baseline (exactly Stage E's Value condition).
No K2/V2 joint probing for Stage H.

## 4. Stage-G policy note (clarification, not invalidation)

Stage G collected prefill-only tensor reconstruction features under a
joint K2/V2 per-layer probe. Because KIVI cache quantization happens
strictly AFTER prefill attention (Section 0), Stage G's joint probe did
not create any KIVI-decode-trajectory confound: the Key reconstruction
computed under K2/V2 is numerically the same K2 reconstruction relevant to
K2/V16 (Value's bit-width does not affect Key's quantizer at all, and vice
versa -- the two axes are quantized independently). **Stage G4's frozen
NO_GO analysis is not invalidated and is not rerun.** This is a
configuration-design note for Stage H's stricter trajectory-fidelity
requirement (which needs the actual axis-isolated generation trajectory,
not just axis-isolated tensor reconstruction), not evidence of a Stage-G
defect.

## 5. Primary Key metric -- LOCKED

```
L_fp16_t = Q_t @ K_shadow_t^T / sqrt(d)
L_kivi_t = cat([cuda_bmm_fA_qB_outer(Q_t, key_states_quant_trans_real, scale_real, mn_real, k_bits),
                Q_t @ key_states_full_real^T], dim=-1) / sqrt(d)
           # real captured production tensors, real production kernel -- reused, not reimplemented
KeyDecodeDistortion_t = ||L_kivi_t - L_fp16_t||_F / max(||L_fp16_t||_F, epsilon)
epsilon = 1e-12
```
Expected: Spearman(KeyDecodeDistortion, SignedKeySensitivity) `rho < 0`.

## 6. Primary Value metric -- LOCKED

Collected under K16/V2, where production's real `attn_weights` already
equals the FP16-Key attention geometry (Key was never quantized in this
policy), so `A_fp16_t` can be reconstructed directly from `Q_t` and
`K_shadow_t` without needing any monkeypatch on the real forward pass for
the production collector (a direct softmax spy is reserved for the GPU
canary's parity check only, Section 17):
```
A_fp16_t = softmax(Q_t @ K_shadow_t^T / sqrt(d))
O_fp16_t = A_fp16_t @ V_shadow_t
O_value_kivi_t = same production kernel split (cuda_bmm_fA_qB_outer + matmul), fed A_fp16_t
                 with the REAL captured value_states_quant / value_states_full
ValueDecodeDistortion_t = ||O_value_kivi_t - O_fp16_t||_F / max(||O_fp16_t||_F, epsilon)
```
Expected: Spearman(ValueDecodeDistortion, SignedValueSensitivity) `rho < 0`.

## 7. Requested decode steps -- LOCKED

```
requested_steps = {1, 2, 4, 8, 16}
```
where cached decode step 1 is the first `past_key_value != None` forward
(Section 0). Fixed before collection. No relative-percentile sampling
(unknowable before generation completes), no feature-dependent sampling.
Verified against real `config/dataset2maxlen.json`: all 6 pre-registered
tasks have `max_new_tokens` in [32, 128], so step 16 sits within every
task's ceiling.

## 8. Early-EOS / no-decode-exposure rule -- LOCKED

```
valid_steps = {1,2,4,8,16} ∩ actually_executed_cached_decode_steps
```
If `valid_steps` is non-empty: `SampleAttentionDistortion` = equal-step
arithmetic mean of the per-step distortion over `valid_steps` -- never
weighted by sequence length, cache length, decode step number, or key
count. Record `valid_step_count`, `requested_step_count` (=5),
`actual_generated_token_count`, `sampled_decode_steps` explicitly per
sample.

**Zero-exposure case** (model generated only token #1 and stopped before
any cached-decode forward ever ran): this is a real, valid observation --
genuinely zero KIVI-cache exposure for that sample, not a missing value.
Pre-register:
```
no_decode_exposure = true
SampleAttentionDistortion = 0.0
valid_step_count = 0
```
The sample is never dropped or replaced.

## 9. Sample -> task/layer/axis aggregation

Reuse the exact same 4 pre-registered calibration prompts per task from
Stage G (Section 13). For each (task, layer, axis): **equal-sample
arithmetic mean** of the 4 samples' `SampleAttentionDistortion` values --
never token-weighted. All 4 selected samples are always represented,
including a zero-exposure sample if one occurs. Median may be reported
only as a secondary robustness summary.

## 10. Primary labeled matrix (unchanged from Stage G)

Primary: trec, lcc, passage_retrieval_en, 2wikimqa x layers {0,4,9,13,18,
22,27,31} x {Key, Value} -> 32 task-layer observations per axis, using the
same finalized Stage-E `SignedKeySensitivity`/`SignedValueSensitivity`.
F0 diagnostic: multifieldqa_en, samsum, Layer 0 only.

## 11. True collection trajectory count -- corrected accounting

Because each (layer, axis) condition is its own single-layer policy, it is
its own real model configuration with its own real generation trajectory
-- layers cannot share one generation run the way Stage G's single joint
forward pass could.

- **Primary**: 4 tasks x 4 calibration prompts x 8 single-layer policies x
  2 axes = **256 generation trajectories**.
- **Diagnostic**: 2 tasks x 4 prompts x 1 layer x 2 axes = **16 generation
  trajectories**.
- **Total: 272 generation trajectories.** This is not "twice Stage G's
  collection" -- Stage G ran 2 total forward passes (one per scope); Stage
  H requires 272 independent generation runs.

Distinct (layer, axis) policy configurations: **16** (8 layers x 2 axes).
Layer 0's two configs (Key, Value) each additionally carry the 2
diagnostic tasks' 4 prompts (8 extra prompts), since Layer 0 is shared
between primary and diagnostic scope: `2 x (16 primary-task-samples + 8
diagnostic-samples) + 14 x 16 primary-task-samples = 48 + 224 = 272`
(cross-checks against the total above).

## 12. Generation horizon -- LOCKED

The Stage-H measurement run is explicitly **not** the same generation as
Stage-E's scored run. Since decoding is deterministic/greedy
(`do_sample=False`) and strictly causal, decode step t's computation
depends only on tokens 1..t, never on how much further generation
eventually continues. Therefore truncating the Stage-H measurement
generation at `max_new_tokens ~= 17-20` (enough to reach cached decode
step 16: token #1 from prefill + steps 1-16) produces **provably
identical** early-decode-step computations to what the full Stage-E-length
generation would have produced for those same steps -- not merely a cheap
approximation. This makes stopping after step 16 both valid and
substantially cheaper than running each task's full `max_new_tokens`
budget (32-128).

**Explicit interpretation lock**: Stage H's hypothesis is "early cached-
decode attention distortion (steps 1-16) associates with the SignedSensitivity
label measured over the full task"; the feature is an early-trajectory
descriptor and is never claimed to summarize or predict later decode
behavior beyond what is empirically tested via the Stage-E label
correlation itself.

## 13. Estimated runtime -- conservative range, NOT claimed to be 5-10 minutes

Two cost components:
- **Model load**: 16 distinct (layer,axis) policy configurations exist.
  Source-confirmed (`models/llama_kivi.py` lines ~181-187): `k_bits`,
  `v_bits`, `quantize_key`, `quantize_value` are plain instance attributes
  read fresh every `forward()` call -- in principle a single loaded model
  could have these attributes mutated between policy groups instead of
  reloading weights each time, which would cut this cost to ~one load
  total. **This is not yet validated** (unclear whether any other
  construction-time-only state implicitly depends on these values) and is
  flagged as the first correctness check for Stage H1-IMPL, not assumed
  here. Conservative bound: 16 x ~96s = ~1536s. Optimistic bound (if
  mutate-in-place is validated safe): ~96s (one load).
- **Per-trajectory generation**: prefill cost comparable to Stage G3B's
  measured per-sample forward times (~2-6s depending on length) + up to 16
  decode-step forwards (each far cheaper than prefill; no real profiling
  data exists yet for this exact decode+shadow-hook workload, so this is a
  rough estimate of roughly 2-8s for 16 steps) -> ~5-15s per trajectory,
  times 272 trajectories -> ~1360s-4080s.

**Conservative total range: roughly 25 minutes to just over 1.5 hours**,
dominated by whichever end of the model-reload uncertainty applies. This
range will be replaced with a measured number once a real timing check
runs; it is not claimed to be exact, and is explicitly not claimed to be
5-10 minutes.

Shadow-cache and KIVI-fused-kernel side-calculation overhead is negligible
against this (Section 18) and does not materially change the range.

## 14. Task-specific primary analysis (structure only, not executed)

For each of the 4 Stage-E tasks separately, across the 8 layers: Spearman
between `KeyDecodeDistortion` (mean per task/layer) and
`SignedKeySensitivity`; separately, `ValueDecodeDistortion` vs
`SignedValueSensitivity`. Reported per task, never pooled as 32
independent observations (identical discipline to Stage G4).

## 15. Undefined-Spearman convention -- reused unchanged

`raw_rho = undefined` whenever either vector (feature or target) has zero
variance; NEVER reported as an observed `rho=0`. For gate evaluation only:
`gate_rho = 0.0`. Undefined values never count as satisfying `<0`, and are
never dropped from a median (contribute `abs(gate_rho)=0`). The known
constant-target example (trec's Value `SignedSensitivity`) remains the
expected concrete case this rule must handle correctly. Implemented via
the SAME `analysis/feature_pilot_gate.py::to_gate_rho`, unmodified.

## 16. Stage-H gate -- LOCKED (structure reused, magnitude threshold re-justified, not auto-inherited)

Reusing the Stage-G A/B/C/D structure is justified because the labeled
matrix (32 observations/axis, 4 tasks, 8 layers) and the task-specific
Spearman setting are identical -- only the feature definition changed.

**FEATURE_PILOT_STAGE = GO** only if, for the SAME axis (Key or Value), ALL
of:
- **A**: `gate_rho < 0` in at least 3 of 4 Stage-E tasks.
- **B**: `median(abs(gate_rho))` across all 4 tasks >= **0.5** -- described
  strictly as **"a pre-registered pragmatic magnitude threshold"**, never
  as Cohen's convention or a universal large-effect standard.
- **C**: after removing lcc, at least 2 of the remaining 3 `gate_rho` < 0
  AND the median of exactly those 3 `gate_rho` < 0.
- **D**: Layer-0 across lcc/multifieldqa_en/samsum, Spearman(mean
  attention-aware distortion, SignedSensitivity) has `gate_rho < 0`.

**FEATURE_PILOT_STAGE = NO_GO** otherwise. No `CONDITIONAL_GO`. Implemented
by reusing `analysis/feature_pilot_gate.py`'s existing, already-generic
`evaluate_feature_pilot_gate`/`evaluate_axis_gate`/`evaluate_criterion_a..d`
unmodified -- these functions operate on any `raw_rho_by_task` dict
regardless of the underlying feature, so no new gate code is needed, only
new feature inputs.

## 17. GPU canary plan (designed, NOT run)

One short real prompt, one layer, one axis at a time, enough generation to
reach cached decode steps 1 and 2 only. Must prove, with exact numerical
targets:
- (A) captured decode `Q_t` matches production `Q_t` (allclose, atol=1e-3,
  same protocol as Stage G2).
- (B) shadow K/V token counts and positions align exactly with production
  history (exact count match every step; bit-exact content match against
  independently hooked production tensors).
- (C) Key: `L_kivi_t` reconstructed from captured real production tensors
  via the real `cuda_bmm_fA_qB_outer` kernel matches a directly-hooked
  real production logits tensor -- target **bit-exact** (`torch.equal`),
  since it is the same kernel called twice on identical inputs.
- (D) Value: `O_value_kivi_t` similarly matches a directly-hooked real
  production attention output -- target bit-exact.
- (E) measurement hooks (read-only) do not change generated tokens or
  logits -- verified via exact text/logit match with hooks on vs off.
- (F) Key/Value cache rollover timing (bulk ~114-step for Key, per-step
  FIFO for Value) matches observed real `past_key_value` tuple shape
  changes at the predicted steps.
A **direct** `nn.functional.softmax` production-attention spy may be used
in this canary ONLY, to independently verify (C)/(D) without relying on
the collector's own reconstruction -- never used in the production
collector itself (Section 6). Production parity (A-D, F) and scientific
correlation are explicitly separate concerns; this canary answers only the
former.

## 18. Shadow-cache memory (one layer, K+V FP16 upper bound)

`2 x 32 heads x 128 head_dim x 2 bytes = 16,384 bytes/token`.
- 10,000 tokens: 163,840,000 bytes = 163.84 MB (decimal) = 156.25 MiB
- 18,000 tokens: 294,912,000 bytes = 294.91 MB (decimal) = 281.25 MiB
- 31,500 tokens: 516,096,000 bytes = 516.10 MB (decimal) = 492.19 MiB

Axis-specific implementations may need less (e.g. Key-only collection
needs only a K shadow, Value-only needs only a V shadow -- half the above),
but budget conservatively using the combined K+V figure. Negligible next
to Stage G3B's measured ~29GB model+cache footprint.

## 19. Output / provenance plan

Future output root: `outputs/layer_attention_feature_pilot/` -- kept
strictly separate from the frozen `outputs/layer_feature_pilot/`. Planned
per-record fields (minimum): `task`, `dataset_index`, `layer`, `axis`,
`policy`, `generated_token_count`, `requested_decode_steps`,
`sampled_decode_steps`, `valid_step_count`, `no_decode_exposure`,
`sample_attention_distortion`, per-step distortion values, `git_commit`,
model/config provenance (mirroring `run_config.json`/`manifest.json`'s
existing fields). No full shadow caches or attention tensors written to
disk -- scalar summaries only, same discipline as Stage G.

## 20. Stop conditions (unchanged, reaffirmed)

Before any scientific collection: if production parity (Section 17 A-D,F)
fails -> BLOCK. If measurement hooks alter generated tokens/logits ->
BLOCK. If cache alignment cannot be proven -> BLOCK. If the measured
collection cost (once profiled) is unreasonable, report it before
launching rather than launching anyway. The metric formulas and gate
(Sections 5-8, 16) are not to be weakened after seeing feature values.
