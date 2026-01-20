import torch
import os
import time
import webdataset as wds
from pathlib import Path
import torch.distributed as dist
from siglip import loss_fn, get_lr
from utils import filter_sample, preprocess, save_checkpoint, model_import
from DDPManager import DDPManager
from functools import partial
from val import eval_model

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.manual_seed(1337)
if device == "cuda":
    torch.cuda.manual_seed(1337)

url = "https://huggingface.co/datasets/pixparse/cc12m-wds/resolve/main/cc12m-train-{0000..0003}.tar"
val_url = "https://huggingface.co/datasets/pixparse/cc12m-wds/resolve/main/cc12m-train-{0004..0007}.tar"
train_url = "https://huggingface.co/datasets/pixparse/cc12m-wds/resolve/main/cc12m-train-{0000..2170}.tar"
eval_url = "https://huggingface.co/datasets/pixparse/cc12m-wds/resolve/main/cc12m-train-{2171..2172}.tar"
test_url = "https://huggingface.co/datasets/pixparse/cc12m-wds/resolve/main/cc12m-train-{2173..2175}.tar"


train_cache_dir = Path("train_cache_dir")
train_cache_dir.mkdir(parents=True, exist_ok=True)
trainset = (
    wds.WebDataset(  # type: ignore
        url,
        shardshuffle=False,
        resampled=True,
        cache_dir=train_cache_dir,
        nodesplitter=wds.split_by_node,  # type: ignore
    )  # type: ignore
    .shuffle(1000)
    .decode("pil")
    .select(filter_sample)
    .batched(512)
    .map(preprocess)
)


trainloader = wds.WebLoader(  # type: ignore
    trainset, batch_size=None, num_workers=1, pin_memory=True, prefetch_factor=2
)  # type: ignore
trainloader = trainloader.unbatched().shuffle(100).batched(32)


val_cache_dir = Path("val_cache_dir")
val_cache_dir.mkdir(parents=True, exist_ok=True)
evalset = (
    wds.WebDataset(  # type: ignore
        val_url,
        shardshuffle=False,
        cache_dir=val_cache_dir,
        nodesplitter=wds.split_by_node,  # pyright: ignore[reportAttributeAccessIssue]
    )  # type: ignore
    .decode("pil")
    .select(filter_sample)
    .batched(64)
    .map(preprocess)
)
evalloader = wds.WebLoader(  # type: ignore
    evalset, batch_size=None, num_workers=4, pin_memory=True, prefetch_factor=4
)  # type: ignore


model = model_import(device)
# torch.cuda.empty_cache()

ddp = DDPManager()
device = ddp.device
model = ddp.wrap_model(model)
# model = torch.compile(model)
raw_model = model.module if ddp.is_distributed else model  # type: ignore

num_samples = 10240000
global_batch_size = 4096
mini_batch_size = 1024
num_epochs = 2
world_size = ddp.world_size

grad_accum_steps = int(global_batch_size / (mini_batch_size * world_size))

steps_per_epoch = num_samples // global_batch_size
max_steps = num_epochs * steps_per_epoch
warmup_steps = 0  # int(0.15 * max_steps)
trainloader_steps = max_steps * grad_accum_steps

max_lr = 1e-4
min_lr = max_lr * 0.1  # might be 0.01

trainloader = trainloader.with_epoch(trainloader_steps)
val_loader = evalloader.with_epoch(8)
optimizer = model.configure_optimizers(weight_decay=0.05, learning_rate=max_lr)  # type: ignore


def train():
    prefilled_lr = partial(get_lr, warmup_steps, max_steps, max_lr, min_lr)
    optimizer.zero_grad(set_to_none=True)
    opt_step = 0
    t0 = time.time()

    try:
        for step, batch in enumerate(trainloader):
            if opt_step >= 10:
                break
            update_step = (step + 1) % grad_accum_steps == 0

            if ddp.is_distributed:
                model.require_backward_grad_sync = update_step  # type: ignore

            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            logits = model(input_ids, attention_mask, pixel_values)
            loss = loss_fn(logits) / grad_accum_steps
            loss.backward()

            if not update_step:
                continue

            loss_to_log = loss.detach() * grad_accum_steps
            if ddp.is_distributed:
                dist.all_reduce(loss_to_log, op=dist.ReduceOp.AVG)

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore
            lr = prefilled_lr(opt_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if ddp.is_master:
                logit_scale = raw_model.logit_scale.exp()  # type: ignore
                pairs = mini_batch_size * grad_accum_steps * world_size
                pairs_per_sec = pairs / dt
                log_str = (
                    f"opt_step={opt_step} | step={step + 1} | "
                    f"loss={loss_to_log.item():.6f} | lr={lr:.6f} | "
                    f"norm={norm.item():.4f} | dt={dt:.4f}s | "
                    f"pairs/sec={pairs_per_sec:.2f} | logit_scale={logit_scale.item():.4f}"
                )
                print(log_str)
                with open("logs/training.log", "a") as f:
                    f.write(f"{log_str}\n")

            if opt_step % 5 == 0:
                if opt_step >= 5:
                    save_checkpoint(raw_model, optimizer, opt_step, lr)
                model.eval()  # type: ignore
                val_loss, val_r1, val_r5, val_margin = eval_model(
                    model, val_loader, device, loss_fn, ddp.is_distributed
                )
                ddp.barrier()
                if ddp.is_master:
                    val_str = (
                        f"[VAL] loss={val_loss:.6f} | R@1={val_r1:.4f} | "
                        f"R@5={val_r5:.4f} | margin={val_margin:.4f} | opt_step={opt_step}"
                    )
                    print(val_str)
                    with open("logs/validation.log", "a") as f:
                        f.write(f"{val_str}\n")
                model.train()  # type: ignore

    except Exception as e:
        if ddp.is_master:
            print(f"Error at step {step}: {e}")  # type: ignore
            save_checkpoint(raw_model, optimizer, opt_step, lr)  # type: ignore
        raise


if __name__ == "__main__":
    train()
    ddp.cleanup()
