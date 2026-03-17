import os
import torch

def save_checkpoint(cfg, model, tokenizer, optimizer, scheduler, global_step, rank):
    if rank != 0:
        return

    ckpt_dir = os.path.join(cfg.output_dir, f"step-{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    model.module.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    state = {"global_step": global_step}

    if cfg.save_optimizer:
        state["optimizer"] = optimizer.state_dict()
        state["scheduler"] = scheduler.state_dict()

    torch.save(state, os.path.join(ckpt_dir, "trainer_state.pt"))
    print(f"Saved checkpoint to {ckpt_dir}", flush=True)


def save_final(cfg, model, tokenizer, global_step, rank):
    if rank != 0:
        return

    final_dir = os.path.join(cfg.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.module.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    torch.save({"global_step": global_step}, os.path.join(final_dir, "trainer_state.pt"))
    print(f"Training finished. Final checkpoint saved to {final_dir}", flush=True)


def should_save_on_step(rank):
    return rank == 0


def load_checkpoint_if_needed(cfg, model, optimizer, scheduler, rank):
    if not cfg.resume_from:
        return 0

    trainer_state_path = os.path.join(cfg.resume_from, "trainer_state.pt")
    if rank == 0:
        print(f"Resuming from {cfg.resume_from}", flush=True)

    state = torch.load(trainer_state_path, map_location="cpu")

    if cfg.save_optimizer and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if cfg.save_optimizer and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])

    return state.get("global_step", 0)
