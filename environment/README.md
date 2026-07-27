# Reproducible Environment

This file records the environment observed on `2026-07-26`. It describes the
completed KIVI-2 run host; it is not a claim that the DGX Spark environment is
identical.

## Observed Versions

- Operating system: Ubuntu 24.04.2 LTS
- Kernel: Linux 6.17.0-20-generic x86_64
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB
- NVIDIA driver: 570.211.01
- Python: 3.12.3
- PyTorch: 2.4.1+cu121
- PyTorch CUDA runtime: 12.1
- `nvcc`: CUDA 12.0, V12.0.140
- Transformers: 4.43.4
- Virtual environment: `/home/m11415015/KIVI/.venv`

The virtual environment itself, compiled extensions, model caches, datasets,
predictions, and logs are excluded from version control.

## Project Installation

The project uses `pyproject.toml` for the KIVI Python package and
`quant/setup.py` for the custom CUDA extension. The installed package metadata
confirmed both packages were installed in editable mode from this checkout.

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
cd quant
python -m pip install -e .
cd ..
```

The custom extension command is the repository's documented command:

```bash
cd quant && pip install -e .
```

After installation, verify imports without starting model inference:

```bash
python -c "import torch, transformers, kivi_gemv; print(torch.__version__, transformers.__version__)"
```

## Dependency Notes

- `pyproject.toml` requests PyTorch 2.4.1 and Transformers 4.43.1.
- The completed run environment contained PyTorch 2.4.1+cu121 and Transformers
  4.43.4.
- The older `requirements.txt` pins PyTorch 2.1.2 and Transformers 4.36.2, so
  it does not exactly describe the completed run environment.
- `environment/pip-freeze.txt` is intentionally omitted because the current
  environment freeze contains local editable/file references and is not
  directly portable. The observed critical versions above and
  `pyproject.toml` are the safer reproduction inputs.

## DGX Spark Environment (aarch64, GB10 Blackwell)

Observed on `2026-07-27`. This is a different host from the RTX 4090 section
above: aarch64 (Grace CPU), CUDA 13.0 driver, GB10 GPU (compute capability
12.1). None of this section overrides the RTX 4090 observations; it records
what was required to reproduce on this specific hardware.

- PyTorch: `2.9.1+cu130` (aarch64 CUDA 13 build; `torch==2.4.1` from
  `pyproject.toml` has no aarch64 CUDA wheel and cannot be used on this host)
- flash-attn: `2.8.3` (prebuilt aarch64 wheel matching cu13/torch2.9/cp312,
  fetched from the flash-attention GitHub releases; no PyPI aarch64 wheel
  exists, and a from-source build was avoided)
- Transformers: `4.43.4` (`transformers==4.43.1` from `pyproject.toml` fails
  to load `lmsys/longchat-7b-v1.5-32k`'s config: its `rope_scaling` uses the
  legacy `{"type": "linear", "factor": 8.0}` format, and 4.43.1's
  `rope_config_validation()` resolves `rope_type` via backward-compat for
  dispatch but the type-specific validator
  (`_validate_linear_scaling_rope_parameters`) still does a hard
  `rope_scaling["rope_type"]` lookup, raising `KeyError: 'rope_type'`. 4.43.4
  does not have this bug. No KIVI model/quantization code was changed to work
  around this; only the transformers patch version was pinned differently
  from `pyproject.toml`.)
- datasets: `3.6.0` (`datasets==5.0.0`, the version pip resolves by default,
  removed support for Hub dataset repos that use a legacy loading script.
  `THUDM/LongBench` still ships `LongBench.py` as a loading script, so
  `load_dataset('THUDM/LongBench', ..., trust_remote_code=True)` fails with
  `RuntimeError: Dataset scripts are no longer supported, but found
  LongBench.py` under 5.0.0. `datasets==3.6.0` still supports loading
  scripts. Verified via a dataset-only smoke test (no GPU, no model weights):
  `THUDM/LongBench` config `qasper`, split `test`, resolved revision
  `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`, 200 rows, all required columns
  present.)
- kivi_gemv CUDA extension: builds and runs correctly on GB10
  (`TORCH_CUDA_ARCH_LIST=12.1`, built with `pip install -e . --no-build-isolation`
  since the package's `setup.py` imports `torch` at build time and pip's
  default build isolation does not see the already-installed torch).
  Verified numerically against the reference dequant+matmul path
  (`cuda_bmm_fA_qB_outer` vs. Triton dequant, ~0.02% relative error).
- Triton (bundled with `torch==2.9.1`, version `3.5.1`) ships a CUDA 12.8
  `ptxas` that does not recognize GB10's `sm_121a` target
  (`ptxas fatal: Value 'sm_121a' is not defined for option 'gpu-name'`).
  Any code path that JIT-compiles Triton kernels (the KIVI quant/pack path
  used by `k_bits < 16` / `v_bits < 16` runs) needs
  `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` (or another CUDA 13+ `ptxas`)
  set in the environment. This does not affect the FP16 (`k_bits=16,
  v_bits=16`) baseline path, which uses standard HuggingFace attention with
  flash-attention-2 and never touches Triton or `kivi_gemv`.
- These are environment/dependency-version compatibility fixes only. No
  changes were made to `models/llama_kivi.py`, `models/mistral_kivi.py`,
  `quant/` CUDA kernels or quantization math, or the LongBench task list in
  `pred_long_bench.py`.
