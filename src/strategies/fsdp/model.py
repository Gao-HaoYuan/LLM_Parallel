import os
from functools import partial

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM

from ...download import ensure_model_assets
from ...dist_utils import log


def build_model(cfg, device, rank):
    log(rank, f"Loading model {cfg.model_name} with FSDP ...")
    model_source = _resolve_model_source(cfg, rank)

    if cfg.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif cfg.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=dtype,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    auto_wrap_policy = _build_auto_wrap_policy(model)
    mixed_precision = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,

        # Sharding strategy trade-off:
        # - FULL_SHARD: shards params, grads, and optimizer state; lowest
        #   memory usage, but usually the highest communication cost.
        # - SHARD_GRAD_OP: uses more memory than FULL_SHARD, but usually
        #   reduces communication overhead and improves training speed.
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        mixed_precision=mixed_precision,

        # Prefetch helps overlap communication with compute, but the real gain
        # depends on model size, sequence length, and interconnect bandwidth.
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,

        device_id=device,
        limit_all_gathers=True,
        use_orig_params=True,
    )


def _resolve_model_source(cfg, rank):
    if cfg.resume_from and os.path.exists(os.path.join(cfg.resume_from, "config.json")):
        log(rank, f"Loading model weights from resume checkpoint {cfg.resume_from}")
        return cfg.resume_from
    return ensure_model_assets(cfg.model_name, rank)


def _build_auto_wrap_policy(model):
    transformer_layers = getattr(getattr(model, "model", None), "layers", None)
    if transformer_layers is None or len(transformer_layers) == 0:
        return None

    transformer_layer_cls = transformer_layers[0].__class__
    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_cls},
    )
