from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, default_data_collator

from .download import ensure_dataset_split, ensure_model_assets
from .dist_utils import log


def build_tokenizer(cfg, rank):
    model_source = ensure_model_assets(cfg.model_name, rank)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _tokenize_dataset(ds, tokenizer, cfg, desc: str):
    def tokenize_fn(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length",
            return_attention_mask=True,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    ds = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds.column_names,
        desc=desc,
    )
    ds.set_format(type="torch")
    return ds


def build_dataloaders(cfg, tokenizer, rank, world_size):
    log(rank, f"Loading train split: {cfg.train_split}")
    train_ds = ensure_dataset_split(cfg.dataset_name, cfg.train_split, rank)
    train_ds = _tokenize_dataset(train_ds, tokenizer, cfg, desc="Tokenizing train")

    log(rank, f"Loading eval split: {cfg.eval_split}")
    eval_ds = ensure_dataset_split(cfg.dataset_name, cfg.eval_split, rank)
    eval_ds = _tokenize_dataset(eval_ds, tokenizer, cfg, desc="Tokenizing eval")

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    eval_sampler = DistributedSampler(
        eval_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.micro_batch_size,
        sampler=train_sampler,
        collate_fn=default_data_collator,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.micro_batch_size,
        sampler=eval_sampler,
        collate_fn=default_data_collator,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )

    return train_loader, train_sampler, eval_loader
