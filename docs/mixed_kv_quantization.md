# Mixed K/V Cache Quantization

Status: **infrastructure implemented and tested on this branch
(`feature/mixed-kv-ablation`), not merged to `main`.** No full 15-task
LongBench ablation has been run. This document describes what was built,
how it was validated, and what is still required before a real K2/V16 or
K16/V2 ablation run.

Base commit: `67591f41341a394474a0590f83aca46a42e15038` (Spark FP16 / KIVI-2 /
KIVI-4 baselines, all completed and committed on `main`).

## What changed

- `pred_long_bench.py`
  - `validate_bits(k_bits, v_bits)`: fails closed before any model load if
    either value isn't in `{2, 4, 16}`. Prevents e.g. `k_bits=8` from ever
    reaching the CUDA kernel's `else` branch (see "Known limitations" in the
    prior read-only review -- that branch silently assumes 2-bit packing for
    anything that isn't 4-bit).
  - `loader_decision(k_bits, v_bits)`: pure function returning
    `(quantize_key, quantize_value, use_kivi_model)`. Only `k_bits == 16 and
    v_bits == 16` gives `use_kivi_model = False` (plain
    `transformers.LlamaForCausalLM`, standard flash-attention-2, nothing
    quantized). Every other allowed combination -- both sides quantized, or
    exactly one side -- uses `LlamaForCausalLM_KIVI`.
  - `build_model_and_tokenizer(...)`: the loader body factored into a
    function so smoke/functional tests call the exact same production code
    path instead of a re-implementation.
  - `build_pred_dir_name(...)`: pure naming function (see "Output
    directory naming" below).
  - `build_run_config`, `load_run_config`, `write_run_config`,
    `prepare_run_directory`: `run_config.json` metadata and the resume/skip
    guard (see "Resume metadata" below).
  - `KIVI_PRED_ROOT` / `KIVI_PRED_E_ROOT` env vars override the `pred/` /
    `pred_e/` root, defaulting to the original hardcoded values. Lets smoke
    and functional tests write to `tmp/...` instead of the real baseline
    directories.
  - A one-time runtime log block after model load prints model class,
    k_bits/v_bits, quantize_key/quantize_value, Key/Value route, output
    directory, and resume metadata status.
  - The Mistral loader branch is **unchanged** (still the old
    `k_bits < 16 and v_bits < 16` AND condition) -- out of scope, since no
    Mistral baseline exists and `models/mistral_kivi.py` was not given a
    16-bit pass-through implementation this round. Mixed K/V for Mistral is
    a known gap, not a regression.

- `models/llama_kivi.py` (`LlamaAttention_KIVI` and `LlamaFlashAttention_KIVI`)
  - `self.quantize_key = self.k_bits < 16`, `self.quantize_value = self.v_bits
    < 16`, set in `__init__`, plus `assert self.k_bits in (2, 4, 16)` /
    same for `v_bits`.
  - Prefill: when a side's `quantize_*` is `False`, that side's cache is
    stored as the raw, un-split, un-quantized tensor (`key_states_full =
    key_states` / `value_states_full = value_states`); the existing
    residual-split-and-quantize logic is now inside an `if
    self.quantize_key: ... else: ...` (same for value), so the quantized
    path is byte-for-byte unchanged when both sides are quantized.
  - Decode: the residual-flush triggers (`if key_states_full.shape[-2] ==
    residual_length`, `if value_full_length > residual_length`) are gated
    with `self.quantize_key and` / `self.quantize_value and`. When a side
    isn't quantized, that trigger never fires, so `*_quant_trans` /
    `*_states_quant` stay `None` forever and the side's cache just keeps
    growing in full precision -- the code's existing "no quantized part yet"
    branch (`if key_states_quant_trans is not None: ... else: plain
    torch.matmul`) already handles this correctly; it was never touched.
  - `quant.new_pack.triton_quantize_and_pack_along_last_dim` and
    `quant.matmul.cuda_bmm_fA_qB_outer` are never called for a
    `quantize_*=False` side -- confirmed by the kernel-route tests below,
    not just by code inspection.
  - `quant/matmul.py`'s `assert bits in [2, 4]` and the CUDA kernel
    (`quant/csrc/gemv_cuda.cu`) were **not modified**.

- `eval_long_bench.py`
  - Same `KIVI_PRED_ROOT` / `KIVI_PRED_E_ROOT` override as
    `pred_long_bench.py`, default unchanged (`pred/` / `pred_e/`). Needed so
    the functional test could run real evaluation against an isolated mixed
    -bit directory without touching the real baselines.

## Supported bit combinations

`{2, 4, 16} x {2, 4, 16}` = 9 combinations, all validated by
`validate_bits`. Anything else (3, 8, 12, ...) fails before model load with
a message naming the actual `k_bits`/`v_bits` given.

| k_bits | v_bits | model class | Key route | Value route |
|---|---|---|---|---|
| 16 | 16 | `LlamaForCausalLM` | FP16 | FP16 |
| 2 | 2 | `LlamaForCausalLM_KIVI` | KIVI quantized | KIVI quantized |
| 4 | 4 | `LlamaForCausalLM_KIVI` | KIVI quantized | KIVI quantized |
| 2 | 4 | `LlamaForCausalLM_KIVI` | KIVI quantized | KIVI quantized |
| 4 | 2 | `LlamaForCausalLM_KIVI` | KIVI quantized | KIVI quantized |
| 2 | 16 | `LlamaForCausalLM_KIVI` | KIVI quantized | FP16 pass-through |
| 16 | 2 | `LlamaForCausalLM_KIVI` | FP16 pass-through | KIVI quantized |
| 4 | 16 | `LlamaForCausalLM_KIVI` | KIVI quantized | FP16 pass-through |
| 16 | 4 | `LlamaForCausalLM_KIVI` | FP16 pass-through | KIVI quantized |

## Output directory naming

`build_pred_dir_name(model_name, max_length, k_bits, v_bits, group_size,
residual_length)`:

- `k_bits == v_bits`: legacy format, unchanged --
  `{model}_{max_length}_{k_bits}bits_group{g}_residual{r}`. The three real
  baseline directory names (`_16bits_`, `_2bits_`, `_4bits_`) are asserted
  unchanged by `tests/test_mixed_kv_unit.py`.
- `k_bits != v_bits`: `{model}_{max_length}_k{k}_v{v}_group{g}_residual{r}`,
  e.g. `longchat-7b-v1.5-32k_31500_k2_v16_group32_residual128`.

This was the most severe gap found in the prior read-only review: the old
naming used only `k_bits`, so a hypothetical K2/V16 run would have landed in
the exact same directory as the real K2/V2 baseline (and K16/V2 in the real
FP16 baseline directory), and the old row-count resume logic would have
silently skipped all 15 tasks against the wrong baseline's predictions.
`tests/test_mixed_kv_unit.py::TestOutputNaming` checks the 5 primary
configs and all 9 supported configs are pairwise distinct, and that mixed
names never collide with symmetric ones.

## Resume metadata (`run_config.json`)

Written once per prediction directory, containing `model_name_or_path`,
`k_bits`, `v_bits`, `group_size`, `residual_length`, `max_length`, `seed`,
`model_class`, `quantize_key`, `quantize_value`, `git_commit`,
`transformers_version`. No tokens or credentials.

`prepare_run_directory(pred_dir, run_config)`:

- Directory doesn't exist -> create it, write `run_config.json`, status
  `"created"`.
- Directory exists, `run_config.json` matches on the core keys
  (`model_name_or_path`, `k_bits`, `v_bits`, `group_size`,
  `residual_length`, `max_length`) -> status `"validated"`, resume/skip
  proceeds as before.
- Directory exists, `run_config.json` exists but a core key differs ->
  **raises `RuntimeError`** naming the mismatched keys. Nothing is
  written or skipped.
- Directory exists, no `run_config.json`, `k_bits == v_bits` (a real
  pre-existing symmetric baseline from before this feature existed) ->
  status `"legacy_no_metadata"`, left completely alone, no metadata written
  in this pass either. Verified: the three real baseline directories still
  have no `run_config.json` after every test run in this branch.
- Directory exists, no `run_config.json`, `k_bits != v_bits` -> **raises
  `RuntimeError`**. A mixed-bit-named directory can never legitimately
  predate this feature, so unverified state there is always refused rather
  than silently resumed/skipped.

## 16-bit pass-through: what "full precision" means here

When a side is `quantize_*=False`, its cache is exactly the raw
rotary-embedded (Key) or raw (Value) tensor, concatenated across steps, with
no packing, no scale/zero-point, and no residual-length-triggered flush --
i.e. a standard unbounded FP16 KV cache for that side only. Confirmed
numerically (see below): the K16 side's attention scores are bit-identical
(max abs diff `0.0`) to an independent, unquantized reference
implementation, and the V16 side's output is bit-identical (max abs diff
`0.0`) to a plain `torch.matmul` against independently-computed
full-precision V.

## Known limitations

- Only `models/llama_kivi.py` was changed. `models/mistral_kivi.py` and its
  loader branch are untouched -- mixed K/V for Mistral is unimplemented
  (same AND-based fallback as before this branch).
- `LlamaAttention_KIVI` (the non-flash class) was updated for consistency,
  but production always sets `config.use_flash = True`, so
  `LlamaFlashAttention_KIVI` is the only class actually exercised by
  `pred_long_bench.py` / the real baselines.
- Per-layer mixed routing is still not implemented (same limitation as
  before this branch): `k_bits`/`v_bits` apply uniformly to every layer.
- No throughput/memory benchmarking was done this round (out of scope --
  this round was infrastructure + correctness only, no full LongBench run).

## Test results (this branch, `feature/mixed-kv-ablation`)

All commands run from the repo root with `.venv` active:

```bash
./.venv/bin/python -m unittest tests.test_mixed_kv_unit -v            # 20 tests, CPU only
TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
  ./.venv/bin/python -m unittest tests.test_mixed_kv_kernel_route -v  # 7 tests, GPU
TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
  ./.venv/bin/python -m unittest tests.test_mixed_kv_numerical -v     # 5 tests, GPU
```

32/32 pass. Kernel-route tests spy on `triton_quantize_and_pack_along_last_dim`
and `cuda_bmm_fA_qB_outer` inside a real `LlamaFlashAttention_KIVI` forward
(prefill + one decode step) with real weights-shaped random tensors on GPU,
and assert which side's kernel was (not) called, and with which `bits`
value. Numerical tests compare against an independent
`transformers.models.llama.modeling_llama.LlamaAttention` reference with
copied weights:

| comparison | max abs diff | interpretation |
|---|---|---|
| K16/V2 attn_weights vs. FP16 reference | `0.0` | Key side untouched by V's quantization, bit-identical |
| K2/V16 output vs. manual matmul against reference V | `0.0` | Value side never quantized despite K being 2-bit |
| K16/V16 attn_weights and output vs. reference | `0.0` | full pass-through == plain attention |
| K2/V2 attn_weights vs. FP16 reference | `0.0121` | expected 2-bit quantization error, present |
| K4/V4 attn_weights vs. FP16 reference | `0.0017` | expected 4-bit error, smaller than 2-bit as expected |

`CLOSE_RTOL = CLOSE_ATOL = 1e-2` in `tests/test_mixed_kv_numerical.py`; the
measured diffs above show this tolerance is not vacuous (0.0 for the
pass-through claims, ~0.01-0.001 for the deliberately-not-asserted
quantized-side claims).

Model-load + short generation smoke tests (`tests/smoke_model_load_generate.py`,
real `lmsys/longchat-7b-v1.5-32k` weights, ~240-token prompt,
`max_new_tokens=8`, output under `tmp/smoke_outputs/pred/`, never the real
`pred/`) and 5-sample functional tests
(`tests/functional_five_sample.py`, first 5 rows of the real `qasper`
LongBench split, output under `tmp/functional_outputs/pred/`, including a
real `eval_long_bench.py` run against the isolated directory) both passed
for all 5 primary configs (16/16, 2/2, 4/4, 2/16, 16/2): correct model
class and route labels in the log, no NaN/Inf in generation scores, no
CUDA assertion errors, non-empty output, correct `run_config.json`, no
directory collisions, and (functional test only) `eval_long_bench.py` exit
code 0 with a numeric `qasper` score in the resulting `result.json`. GPU
memory returned to 0 MiB between every subprocess. The real baseline
directories under `pred/` were confirmed untouched (file counts and
modification times unchanged, no `run_config.json` leaked in, `result.json`
checksums match the committed baseline) throughout.

5-sample qasper scores (informational only -- not a pass/fail gate, 5
samples has no statistical meaning): FP16 36.65, KIVI-2 23.12, KIVI-4
35.87, K2/V16 37.37, K16/V2 35.33.

## Before a real K2/V16 or K16/V2 LongBench ablation

1. Merge (or otherwise land) this branch, or run directly from
   `feature/mixed-kv-ablation`.
2. Launch exactly like the existing baselines
   (`scripts/long_test.sh <gpu> <k_bits> <v_bits> <group> <residual>
   <model>`), e.g. `bash scripts/long_test.sh 0 2 16 32 128
   lmsys/longchat-7b-v1.5-32k`. No script changes are needed for this --
   `k_bits`/`v_bits` were already passed through separately (see the prior
   read-only review, item 12).
3. Expect a new prediction directory
   `pred/longchat-7b-v1.5-32k_31500_k2_v16_group32_residual128/` (or
   `..._k16_v2_...`) with its own `run_config.json` -- distinct from all
   three existing baseline directories.
4. Evaluate with `python eval_long_bench.py --model
   longchat-7b-v1.5-32k_31500_k2_v16_group32_residual128` as usual.
