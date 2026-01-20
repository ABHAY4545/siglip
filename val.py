import torch
import torch.distributed as dist


@torch.no_grad()
def metrics_retrieval(logits):
    B = logits.size(0)
    targets = torch.arange(B, device=logits.device)
    ranks = torch.argsort(logits, dim=-1, descending=True)
    correct = ranks == targets[:, None]
    r1 = correct[:, :1].any(dim=-1).float().mean()
    r5 = correct[:, :5].any(dim=-1).float().mean()
    diag = logits.diag()
    mask = ~torch.eye(B, device=logits.device, dtype=torch.bool)
    off_diag = logits[mask]
    margin = diag.mean() - off_diag.mean()
    return r1.item(), r5.item(), margin.item()


@torch.no_grad()
def eval_model(model, val_loader, device, loss_fn, ddp_is_distributed):
    image_embeds = []
    text_embeds = []

    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        image_features, text_features = model(
            input_ids, attention_mask, pixel_values, return_features=True
        )
        image_embeds.append(image_features)
        text_embeds.append(text_features)

    image_embeds = torch.cat(image_embeds, dim=0)
    text_embeds = torch.cat(text_embeds, dim=0)

    if ddp_is_distributed:
        world_size = dist.get_world_size()
        gathered_image = [torch.zeros_like(image_embeds) for _ in range(world_size)]
        gathered_text = [torch.zeros_like(text_embeds) for _ in range(world_size)]
        dist.all_gather(gathered_image, image_embeds)
        dist.all_gather(gathered_text, text_embeds)
        image_embeds = torch.cat(gathered_image, dim=0)
        text_embeds = torch.cat(gathered_text, dim=0)

    model_module = model.module if hasattr(model, "module") else model
    logit_scale = model_module.logit_scale.exp().clamp(max=100)
    val_logits = (image_embeds @ text_embeds.T) * logit_scale + model_module.logit_bias

    r1, r5, diag_margin = metrics_retrieval(val_logits)
    loss = loss_fn(val_logits).item()
    return loss, r1, r5, diag_margin


if __name__ == "__main__":
    raise RuntimeError(
        "This file is not meant to be run directly.\n"
        "Import the functions you need instead."
    )
