#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

CONFIG=${1:-configs/qwen25_3b_base.yaml}

torchrun --standalone --nproc_per_node=8 train.py --config "${CONFIG}"