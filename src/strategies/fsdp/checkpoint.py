import os

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    StateDictType,
)


FULL_STATE_DICT_CONFIG = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
FULL_OPTIM_STATE_DICT_CONFIG = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)


def save_checkpoint(cfg, model, tokenizer, optimizer, scheduler, global_step, rank):
    ckpt_dir = os.path.join(cfg.output_dir, f"step-{global_step}")
    save_model_and_tokenizer(model, tokenizer, ckpt_dir, rank)

    state = {"global_step": global_step}

    if cfg.save_optimizer:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FULL_STATE_DICT_CONFIG,
            FULL_OPTIM_STATE_DICT_CONFIG,
        ):
            state["optimizer"] = FSDP.optim_state_dict(model, optimizer)
            state["scheduler"] = scheduler.state_dict()

    if rank == 0:
        torch.save(state, os.path.join(ckpt_dir, "trainer_state.pt"))
        print(f"Saved checkpoint to {ckpt_dir}", flush=True)


def load_checkpoint_if_needed(cfg, model, optimizer, scheduler, rank):
    if not cfg.resume_from:
        return 0

    trainer_state_path = os.path.join(cfg.resume_from, "trainer_state.pt")
    if rank == 0:
        print(f"Resuming from {cfg.resume_from}", flush=True)

    state = torch.load(trainer_state_path, map_location="cpu")

    if cfg.save_optimizer and "optimizer" in state:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FULL_STATE_DICT_CONFIG,
            FULL_OPTIM_STATE_DICT_CONFIG,
        ):
            optim_state = FSDP.optim_state_dict_to_load(model, optimizer, state["optimizer"])
        optimizer.load_state_dict(optim_state)
    if cfg.save_optimizer and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])

    return state.get("global_step", 0)


def save_model_and_tokenizer(model, tokenizer, output_dir, rank):
    os.makedirs(output_dir, exist_ok=True)

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FULL_STATE_DICT_CONFIG,
    ):
        state_dict = model.state_dict()

    if rank == 0:
        model.module.save_pretrained(output_dir, state_dict=state_dict)
        tokenizer.save_pretrained(output_dir)
