import torch
from pathlib import Path
from siglip import SigLipModel
from transformers import AutoProcessor, AutoModel
from model_surgery import model_surgery

model_id = "google/siglip2-base-patch16-256"
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)


def save_checkpoint(raw_model, optimizer, opt_step, lr, path="checkpoints"):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "opt_step": opt_step,
        "learning_rate": lr,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
    }
    save_path = path / f"siglip_step_{opt_step}.pt"
    torch.save(checkpoint, save_path)
    print(f"✅ Checkpoint saved at step {opt_step} to {save_path}")


def load_model(checkpoint_path, max_lr):
    model = SigLipModel(256, 0.15)
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.lock_text_tower()
    optimizer = model.configure_optimizers(weight_decay=0.05, learning_rate=max_lr)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model = torch.compile(model)
    return model, optimizer


def preprocess(samples, **kwargs):
    texts = [caption for caption in samples["txt"]]
    images = [img for img in samples["jpg"]]
    inputs = processor(
        text=texts,
        images=images,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
        **kwargs,
    )

    return {
        "pixel_values": inputs["pixel_values"],
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }


def filter_sample(sample, min_size=256):
    meta = sample.get("json")
    if meta is None:
        return None
    if meta.get("status") != "success":
        return None
    w, h = meta.get("width"), meta.get("height")
    if w is None or h is None:
        return None
    if w < min_size or h < min_size:
        return None
    return sample


def model_import(device):
    hf_model = AutoModel.from_pretrained(model_id)
    model = SigLipModel(img_size=256, dropout_rate=0.1).to(device)
    model_surgery(model, hf_model, 12, vision_model=False)
    model.lock_text_tower()
    return model


if __name__ == "__main__":
    raise RuntimeError(
        "This file is not meant to be run directly.\n"
        "Import the functions you need instead."
    )
