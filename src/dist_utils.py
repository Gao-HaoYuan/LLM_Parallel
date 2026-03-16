import os
import torch
import torch.distributed as dist


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return local_rank, rank, world_size, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def log(rank: int, msg: str):
    if rank == 0:
        print(msg, flush=True)


def is_main_process(rank: int) -> bool:
    return rank == 0