import math
import random

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

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
        # Keep natural sequence lengths here and pad only inside each batch.
        # This is the common production setup before moving to full packing.
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_length,
            return_attention_mask=True,
        )
        out["length"] = [len(input_ids) for input_ids in out["input_ids"]]
        return out

    ds = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds.column_names,
        desc=desc,
    )
    ds = ds.filter(
        lambda example: example["length"] > 0,
        desc=f"{desc} (drop empty samples)",
    )
    return ds


class DistributedBucketSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        drop_last: bool = False,
        shuffle: bool = True,
        bucket_multiplier: int = 50,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.bucket_size = max(batch_size * bucket_multiplier, batch_size)
        self.epoch = 0
        self.lengths = list(dataset["length"])

        if self.drop_last:
            self.num_samples = len(self.lengths) // self.num_replicas
        else:
            self.num_samples = math.ceil(len(self.lengths) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
        else:
            rng = None

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]
        else:
            indices = indices[:self.total_size]

        rank_indices = indices[self.rank:self.total_size:self.num_replicas]

        batches = []
        for start in range(0, len(rank_indices), self.bucket_size):
            bucket = rank_indices[start:start + self.bucket_size]
            bucket.sort(key=lambda idx: self.lengths[idx], reverse=True)
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start:batch_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle and rng is not None:
            rng.shuffle(batches)

        flattened = [idx for batch in batches for idx in batch]
        return iter(flattened[:self.num_samples])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def build_dataloaders(cfg, tokenizer, rank, world_size):
    # Dynamic per-batch padding is the standard baseline in production training:
    # it keeps labels masked on pad positions while avoiding global max_length pad.
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    log(rank, f"Loading train split: {cfg.train_split}")
    train_ds = ensure_dataset_split(cfg.dataset_name, cfg.train_split, rank)
    raw_train_size = len(train_ds)
    train_ds = _tokenize_dataset(train_ds, tokenizer, cfg, desc="Tokenizing train")
    dropped_train = raw_train_size - len(train_ds)
    if dropped_train > 0:
        log(rank, f"Dropped {dropped_train} empty train samples after tokenization")

    log(rank, f"Loading eval split: {cfg.eval_split}")
    eval_ds = ensure_dataset_split(cfg.dataset_name, cfg.eval_split, rank)
    raw_eval_size = len(eval_ds)
    eval_ds = _tokenize_dataset(eval_ds, tokenizer, cfg, desc="Tokenizing eval")
    dropped_eval = raw_eval_size - len(eval_ds)
    if dropped_eval > 0:
        log(rank, f"Dropped {dropped_eval} empty eval samples after tokenization")

    train_sampler = DistributedBucketSampler(
        train_ds,
        batch_size=cfg.micro_batch_size,
        num_replicas=world_size,
        rank=rank,
        seed=cfg.seed,
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
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.micro_batch_size,
        sampler=eval_sampler,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )

    return train_loader, train_sampler, eval_loader
