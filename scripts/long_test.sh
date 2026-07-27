# model e.g.: meta-llama/Llama-2-7b-hf

set -euo pipefail

gpuid=$1
k_bits=$2
v_bits=$3
group_size=$4
residual_length=$5
model=$6
e=0

if [ -z "${PYTHON:-}" ]; then
    if [ -x ./.venv/bin/python ]; then
        PYTHON=./.venv/bin/python
    else
        PYTHON=python3
    fi
fi

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

CUDA_VISIBLE_DEVICES=$gpuid "$PYTHON" pred_long_bench.py --model_name_or_path $model \
    --cache_dir ./cached_models \
    --k_bits $k_bits \
    --v_bits $v_bits \
    --group_size $group_size \
    --residual_length $residual_length \
    --e ${e}