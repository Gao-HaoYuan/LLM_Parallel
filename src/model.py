import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM

from .download import ensure_model_assets
from .dist_utils import log


def build_model(cfg, device, rank):
    log(rank, f"Loading model {cfg.model_name} ...")
    model_source = ensure_model_assets(cfg.model_name, rank)

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
    model.to(device)

    ddp_model = DDP(
        model,
        device_ids=[device.index],
        output_device=device.index,
        find_unused_parameters=False,
    )
    return ddp_model
