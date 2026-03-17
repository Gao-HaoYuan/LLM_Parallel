# Qwen2.5-3B DDP Demo

Single-node multi-GPU PyTorch DDP training demo for `Qwen/Qwen2.5-3B`.

## Features

- Pure DDP only
- Config-driven
- Train / eval split
- Gradient accumulation
- Checkpoint save
- Resume support
- Metrics jsonl logging

## Install

```bash
conda create -n llm-ddp python=3.10 -y
conda activate llm-ddp

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Environment check
```
python scripts/env_check.py
```

## Run debug
```
bash run.sh configs/qwen25_3b_debug.yaml
```

## Run base
```
bash run.sh configs/qwen25_3b_base.yaml
```

## Override config from CLI
```
torchrun --standalone --nproc_per_node=8 train.py \
  --config configs/qwen25_3b_base.yaml \
  --max_length 256 \
  --save_every 50
```

## Resume
```
torchrun --standalone --nproc_per_node=8 train.py \
  --config configs/qwen25_3b_base.yaml \
  --resume_from ./outputs/qwen25_3b_ddp_base/step-200
```

## Infer && Chat
```
python scripts/infer.py \
  --model_path ./outputs/qwen25_3b_ddp_base/final \
  --prompt "Large language model training uses data parallelism because"


python scripts/chat.py --model_path ./outputs/qwen25_3b_ddp_base/final
```