import math
import torch
import torch.distributed as dist


@torch.no_grad()
def evaluate(model, eval_loader, device):
    model.eval()

    loss_sum = torch.zeros(1, device=device)
    count = torch.zeros(1, device=device)

    for batch in eval_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss.detach()

        loss_sum += loss
        count += 1

    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)

    avg_loss = (loss_sum / count).item()
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    model.train()
    return avg_loss, ppl