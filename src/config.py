from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class TrainConfig:
    model_name: str
    output_dir: str

    dataset_name: str
    train_split: str
    eval_split: str

    max_length: int
    epochs: int

    micro_batch_size: int
    grad_accum_steps: int

    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    grad_clip: float

    num_workers: int
    seed: int

    dtype: str

    log_every: int
    save_every: int
    eval_every: int

    resume_from: Optional[str]
    save_optimizer: bool


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return TrainConfig(**data)


def apply_overrides(cfg: TrainConfig, args):
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg