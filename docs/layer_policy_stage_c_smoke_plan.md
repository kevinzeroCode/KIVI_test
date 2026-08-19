# Stage C smoke test

## Correction (this revision)

The previous revision of this document showed a smoke command built on top
of `pred_long_bench.py` ending in a bare `--e`. **That was wrong and must
not be run.**

- `DataArguments.e` (`utils/process_args.py`) means **"Evaluate on
  LongBench-E"** — a *different, separate* task/dataset variant from
  standard LongBench, not a sample-count or smoke-test flag.
- A bare `--e` (no value) sets it to truthy and switches the run to the
  LongBench-E task list (`config/dataset2prompt.json`'s `_e`-suffixed
  entries) and `pred_e/` output root — a completely different experiment
  from what Stage C needs.
- The project's own established convention (`scripts/long_test.sh`) always
  passes this flag **explicitly with a value**: `--e ${e}` with `e=0`, i.e.
  a normal formal LongBench run is `--e 0`, never a bare `--e`.
- Separately, and just as important: `pred_long_bench.py` has **no
  sample-count limiter**. Even `--e 0` runs the full 15-task, ~3550-example
  LongBench loop and writes into the formal `pred/<config_dir>/` resume
  system (`prepare_run_directory`, `resolve_dataset_resume_plan`) — running
  it directly for a "smoke test" would not be a smoke test at all, and would
  risk writing into or colliding with real experiment output.

**Stage C therefore does not invoke `pred_long_bench.py` at all.** It uses a
dedicated smoke harness, `scripts/layer_policy_smoke.py`, that reuses the
same model-construction function (`pred_long_bench.build_model_and_tokenizer`)
and the same tokenizer/prompt-formatting/generation call shape, but runs a
hard-capped number of examples and writes exclusively under
`outputs/layer_policy_smoke/<run_label>/` — never `pred/`, never touching
the resume system.

## Example policy

`analysis/policies/key_probe_layer17_k2.json` — default K16/V16 on every
layer, layer 17 overridden to K2/V16 (LongChat-7b-v1.5-32k has 32 decoder
layers, confirmed from the model's `config.json`).

## Smoke harness

`scripts/layer_policy_smoke.py` — see its `--help` for the full flag list.
Defaults: model `lmsys/longchat-7b-v1.5-32k`, policy
`analysis/policies/key_probe_layer17_k2.json`, dataset `hotpotqa`,
`num_samples=5`, `group_size=32`, `residual_length=128`, `seed=42`, output
root `outputs/layer_policy_smoke/`. Pass `--layer_policy none` to run the
global-K16/V16 control instead of the layer-17 policy.

Actual commands executed and their results are in the response for this
round, not duplicated here (this file documents the harness design and the
`--e` correction, not a specific run's output).
