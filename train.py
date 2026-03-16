import argparse

from src.config import load_config, apply_overrides
from src.dist_utils import setup_distributed, cleanup_distributed, log, set_seed
from src.data import build_tokenizer, build_dataloaders
from src.model import build_model
from src.trainer import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    # 可选覆盖项
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--eval_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    local_rank, rank, world_size, device = setup_distributed()
    set_seed(cfg.seed + rank)

    log(rank, f"Loaded config from {args.config}")
    log(rank, f"Model: {cfg.model_name}")
    log(rank, f"Output dir: {cfg.output_dir}")
    log(rank, f"World size: {world_size}")

    tokenizer = build_tokenizer(cfg, rank)
    train_loader, train_sampler, eval_loader = build_dataloaders(cfg, tokenizer, rank, world_size)
    model = build_model(cfg, device, rank)

    train(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        train_sampler=train_sampler,
        eval_loader=eval_loader,
        device=device,
        rank=rank,
    )

    cleanup_distributed()


if __name__ == "__main__":
    main()
