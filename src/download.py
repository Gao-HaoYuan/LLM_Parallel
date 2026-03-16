from __future__ import annotations

import re
import shutil
from pathlib import Path

import torch.distributed as dist
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dist_utils import log


ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
MODELS_DIR = ASSETS_DIR / "models"
DATASETS_DIR = ASSETS_DIR / "datasets"


def ensure_model_assets(model_name: str, rank: int) -> str:
    model_dir = MODELS_DIR / model_name

    if _has_model_assets(model_dir):
        return str(model_dir)

    if rank == 0:
        download_model_assets(model_name, rank=rank, force=_asset_dir_exists(model_dir))

    _barrier_if_needed()
    return str(model_dir)


def ensure_dataset_split(dataset_name: str, split: str, rank: int):
    dataset_dir = DATASETS_DIR / dataset_name

    if not _has_dataset_assets(dataset_dir):
        if rank == 0:
            download_dataset_assets(dataset_name, rank=rank, force=_asset_dir_exists(dataset_dir))
        _barrier_if_needed()

    dataset = load_from_disk(str(dataset_dir))
    return _select_split(dataset, split)


def download_model_assets(model_name: str, rank: int = 0, force: bool = False) -> str:
    model_dir = MODELS_DIR / model_name
    temp_dir = model_dir.with_name(f"{model_dir.name}.tmp")

    if _has_model_assets(model_dir) and not force:
        log(rank, f"Model assets already exist: {model_dir}")
        return str(model_dir)

    # Download into a temporary directory first so other processes never observe
    # a half-written model directory as if it were ready to use.
    log(rank, f"Downloading model assets for {model_name} -> {model_dir}")
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    _reset_dir(temp_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    tokenizer.save_pretrained(temp_dir)
    model.save_pretrained(temp_dir)

    if not _has_model_assets(temp_dir):
        raise RuntimeError(f"Downloaded model assets are incomplete: {temp_dir}")

    _replace_dir(temp_dir, model_dir)
    return str(model_dir)


def download_dataset_assets(dataset_name: str, rank: int = 0, force: bool = False) -> str:
    dataset_dir = DATASETS_DIR / dataset_name
    temp_dir = dataset_dir.with_name(f"{dataset_dir.name}.tmp")

    if _has_dataset_assets(dataset_dir) and not force:
        log(rank, f"Dataset assets already exist: {dataset_dir}")
        return str(dataset_dir)

    # Same idea as model downloads: only replace the final directory after the
    # dataset has been fully written and passes the completeness check.
    log(rank, f"Downloading dataset {dataset_name} -> {dataset_dir}")
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    _reset_dir(temp_dir)

    dataset = load_dataset(dataset_name)
    dataset.save_to_disk(str(temp_dir))

    if not _has_dataset_assets(temp_dir):
        raise RuntimeError(f"Downloaded dataset assets are incomplete: {temp_dir}")

    _replace_dir(temp_dir, dataset_dir)
    return str(dataset_dir)


def _select_split(dataset, split: str):
    # split supported patterns:
    # - "train"                  -> use the full named split
    # - "train[:1000]"           -> take rows [0, 1000)
    # - "train[100:200]"         -> take rows [100, 200)
    # - "validation[500:]"       -> take rows [500, len(split))
    # This keeps the config syntax close to Hugging Face split notation while
    # operating on datasets that have already been saved locally.
    match = re.fullmatch(r"([^\[]+)(?:\[(.*)\])?", split)
    if match is None:
        raise ValueError(f"Unsupported split format: {split}")

    split_name, slice_spec = match.groups()
    ds = dataset[split_name]

    if slice_spec is None:
        return ds

    slice_match = re.fullmatch(r"(\d*)?:(\d*)?", slice_spec)
    if slice_match is None:
        raise ValueError(
            f"Unsupported split slice format: {split}. "
            "Only integer slices like train[:1000] or train[100:200] are supported."
        )

    start_text, end_text = slice_match.groups()
    start = int(start_text) if start_text else 0
    end = int(end_text) if end_text else len(ds)
    indices = range(start, min(end, len(ds)))
    return ds.select(indices)


def _has_model_assets(model_dir: Path) -> bool:
    required_files = [
        model_dir / "config.json",
        model_dir / "tokenizer_config.json",
        model_dir / "tokenizer.json",
    ]
    has_weight_file = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("pytorch_model*.bin"))
    return model_dir.is_dir() and all(path.exists() for path in required_files) and has_weight_file


def _has_dataset_assets(dataset_dir: Path) -> bool:
    required_files = [
        dataset_dir / "dataset_dict.json",
    ]
    split_dirs = [
        dataset_dir / "train",
        dataset_dir / "validation",
    ]
    return dataset_dir.is_dir() and all(path.exists() for path in required_files) and any(
        split_dir.exists() for split_dir in split_dirs
    )


def _asset_dir_exists(path: Path) -> bool:
    return path.exists()


def _reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _replace_dir(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    src.replace(dst)


def _barrier_if_needed():
    # These helpers are called inside the DDP startup path, not only from the
    # standalone download script. Rank 0 may be downloading while other ranks
    # are already trying to read from disk, so they must wait here.
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
