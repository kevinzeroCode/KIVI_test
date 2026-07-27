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
