import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from transformers import get_cosine_schedule_with_warmup

from ...dist_utils import log
from ...evaluator import evaluate
from ...logger import JsonlLogger
from .checkpoint import load_checkpoint_if_needed, save_checkpoint


def train(cfg, model, tokenizer, train_loader, train_sampler, eval_loader, device, rank):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    num_update_steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum_steps)
    max_train_steps = cfg.epochs * num_update_steps_per_epoch
    warmup_steps = int(max_train_steps * cfg.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    global_step = load_checkpoint_if_needed(cfg, model, optimizer, scheduler, rank)
    last_log_time = time.perf_counter()
    last_log_tokens = 0

    metrics_logger = None
    if rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        metrics_logger = JsonlLogger(os.path.join(cfg.output_dir, "metrics.jsonl"))

    log(rank, f"Total train batches per rank: {len(train_loader)}")
    log(rank, f"Total optimizer steps: {max_train_steps}")
    log(rank, f"Resume global_step: {global_step}")

    model.train()

    for epoch in range(cfg.epochs):
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            if cfg.enable_throughput_logging:
                local_tokens = int(batch["attention_mask"].sum().item())
                tokens_tensor = torch.tensor([local_tokens], device=device, dtype=torch.long)
                dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
                last_log_tokens += tokens_tensor.item()

            is_sync_step = ((step + 1) % cfg.grad_accum_steps == 0)
            sync_context = nullcontext() if is_sync_step else model.no_sync()

            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = outputs.loss / cfg.grad_accum_steps
                loss.backward()

            if is_sync_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if rank == 0 and global_step % cfg.log_every == 0:
                    real_loss = loss.item() * cfg.grad_accum_steps
                    lr = scheduler.get_last_lr()[0]
                    msg = (
                        f"[epoch {epoch}] step={global_step}/{max_train_steps} "
                        f"loss={real_loss:.4f} lr={lr:.6e} "
                        f"grad_norm={float(grad_norm):.4f}"
                    )
                    metrics = {
                        "type": "train",
                        "epoch": epoch,
                        "step": global_step,
                        "loss": real_loss,
                        "lr": lr,
                        "grad_norm": float(grad_norm),
                    }

                    if cfg.enable_throughput_logging:
                        now = time.perf_counter()
                        elapsed = now - last_log_time
                        tokens_per_sec = last_log_tokens / elapsed if elapsed > 0 else 0.0
                        msg += f" tokens/sec={tokens_per_sec:.2f}"
                        metrics["tokens_per_sec"] = tokens_per_sec
                        last_log_time = now
                        last_log_tokens = 0

                    print(msg, flush=True)
                    metrics_logger.log(metrics)

                if rank == 0 and global_step % cfg.save_every == 0:
                    save_checkpoint(cfg, model, tokenizer, optimizer, scheduler, global_step, rank)

                if global_step % cfg.eval_every == 0:
                    eval_loss, eval_ppl = evaluate(model, eval_loader, device)
                    if rank == 0:
                        print(
                            f"[eval] step={global_step} eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.4f}",
                            flush=True,
                        )
                        metrics_logger.log({
                            "type": "eval",
                            "epoch": epoch,
                            "step": global_step,
                            "eval_loss": eval_loss,
                            "eval_ppl": eval_ppl,
                        })

    if rank == 0:
        final_dir = os.path.join(cfg.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        model.module.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        torch.save({"global_step": global_step}, os.path.join(final_dir, "trainer_state.pt"))
        print(f"Training finished. Final checkpoint saved to {final_dir}", flush=True)
