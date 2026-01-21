import os
import time
import torch
import webdataset as wds
from pathlib import Path
from val import eval_model
from functools import partial
import torch.distributed as dist
from DDPManager import DDPManager
from siglip import loss_fn, get_lr
from torch.cuda.amp import autocast
from utils import filter_sample, preprocess, save_checkpoint, model_import


TRAIN_URL = "https://huggingface.co/datasets/pixparse/cc12m-wds/resolve/main/cc12m-train-{0000..2170}.tar"
VAL_URL = "https://huggingface.co/datasets/pixparse/cc12m-wds/resolve/main/cc12m-train-{2171..2172}.tar"

PREPROCESS_BATCH_SIZE = 8192
GLOBAL_BATCH_SIZE = 4096
TRAIN_BATCH_SIZE = 512
EVAL_BATCH_SIZE = 1024

NUM_SAMPLES = 10240000
NUM_EPOCHS = 2
MAX_LR = 3e-4
MIN_LR = MAX_LR * 0.1
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.10

TRAIN_NUM_WORKERS = 16
EVAL_NUM_WORKERS = 2
SHUFFLE_BUFFER = 8192
LOADER_SHUFFLE_BUFFER = 2048
PREFETCH_FACTOR = 3

VAL_EPOCH_SIZE = 8
CHECKPOINT_INTERVAL = 250

SEED = 1337


def train(
    trainloader,
    val_loader,
    model,
    raw_model,
    optimizer,
    ddp,
    device,
    world_size,
    grad_accum_steps,
    max_steps,
):
    warmup_steps = int(WARMUP_RATIO * max_steps)
    prefilled_lr = partial(get_lr, warmup_steps, max_steps, MAX_LR, MIN_LR)
    optimizer.zero_grad(set_to_none=True)
    opt_step = 0
    t0 = time.time()

    try:
        for step, batch in enumerate(trainloader):
            update_step = (step + 1) % grad_accum_steps == 0

            if ddp.is_distributed:
                model.require_backward_grad_sync = update_step  # type: ignore

            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16): # type: ignore

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
                pairs = TRAIN_BATCH_SIZE * grad_accum_steps * world_size
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

            if opt_step % CHECKPOINT_INTERVAL == 0:
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
                t0 = time.time()

    except Exception as e:
        if ddp.is_master:
            print(f"Error at step {step}: {e}")  # type: ignore
            save_checkpoint(raw_model, optimizer, opt_step, lr)  # type: ignore
        raise


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if device == "cuda":
        if not torch.cuda.is_bf16_supported():
            print("WARNING: BF16 not supported on this GPU, falling back to FP32")
        else:
            print("BF16 supported and will be used for training")

    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed(SEED)

    # Training dataset
    train_cache_dir = Path("train_cache_dir")
    train_cache_dir.mkdir(parents=True, exist_ok=True)
    trainset = (
        wds.WebDataset(  # type: ignore
            TRAIN_URL,
            shardshuffle=False,
            resampled=True,
            cache_dir=train_cache_dir,
            nodesplitter=wds.split_by_node,  # type: ignore
        )  # type: ignore
        .shuffle(SHUFFLE_BUFFER)
        .decode("pil")
        .select(filter_sample)
        .batched(PREPROCESS_BATCH_SIZE)
        .map(preprocess)
    )

    trainloader = wds.WebLoader(  # type: ignore
        trainset,
        batch_size=None,
        num_workers=TRAIN_NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
    )  # type: ignore
    trainloader = (
        trainloader.unbatched().shuffle(LOADER_SHUFFLE_BUFFER).batched(TRAIN_BATCH_SIZE)
    )

    # Validation dataset
    val_cache_dir = Path("val_cache_dir")
    val_cache_dir.mkdir(parents=True, exist_ok=True)
    evalset = (
        wds.WebDataset(  # type: ignore
            VAL_URL,
            shardshuffle=False,
            cache_dir=val_cache_dir,
            nodesplitter=wds.split_by_node,  # type: ignore
        )  # type: ignore
        .decode("pil")
        .select(filter_sample)
        .batched(EVAL_BATCH_SIZE)
        .map(preprocess)
    )
    evalloader = wds.WebLoader(  # type: ignore
        evalset,
        batch_size=None,
        num_workers=EVAL_NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
    )  # type: ignore

    model = model_import(device)

    ddp = DDPManager()
    device = ddp.device
    model = ddp.wrap_model(model)
    raw_model = model.module if ddp.is_distributed else model  # type: ignore

    world_size = ddp.world_size
    grad_accum_steps = int(GLOBAL_BATCH_SIZE / (TRAIN_BATCH_SIZE * world_size))
    steps_per_epoch = NUM_SAMPLES // GLOBAL_BATCH_SIZE
    max_steps = NUM_EPOCHS * steps_per_epoch

    warmup_steps = int(0.15 * max_steps)
    trainloader_steps = max_steps * grad_accum_steps

    trainloader = trainloader.with_epoch(trainloader_steps)
    val_loader = evalloader.with_epoch(VAL_EPOCH_SIZE)
    optimizer = model.configure_optimizers(
        weight_decay=WEIGHT_DECAY, learning_rate=MAX_LR
    )  # type: ignore

    print("Starting training...")
    print(f"  Global batch size: {GLOBAL_BATCH_SIZE}")
    print(f"  Mini-batch size: {TRAIN_BATCH_SIZE}")
    print(f"  Gradient accumulation steps: {grad_accum_steps}")
    print(f"  World size: {world_size}")
    print(f"  Max steps: {max_steps}")

    train(
        trainloader=trainloader,
        val_loader=val_loader,
        model=model,
        raw_model=raw_model,
        optimizer=optimizer,
        ddp=ddp,
        device=device,
        world_size=world_size,
        grad_accum_steps=grad_accum_steps,
        max_steps=max_steps,
    )
    ddp.cleanup()


if __name__ == "__main__":
    main()
