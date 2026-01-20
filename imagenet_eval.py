import torch
import torch.nn.functional as F
from transformers import SiglipProcessor
from tqdm.auto import tqdm
from torchvision import datasets, transforms


@torch.no_grad()
def evaluate_zero_shot_imagenet(
    model,  # your SigLIP vision encoder (SiglipVisionModel or equivalent)
    processor,  # SiglipProcessor
    imagenet_val_path,
    device="cuda",
    batch_size=256,
    prompt_template="a photo of a {}",  # classic CLIP-style template
):
    model.eval()
    model.to(device)

    # Standard ImageNet normalization for SigLIP (mean/std 0.5)
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),  # your input size
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )

    val_dataset = datasets.ImageFolder(imagenet_val_path, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Get the 1000 ImageNet class names (you need this list!)
    # Download from: https://storage.googleapis.com/bit-bucket/imagenet_classes.txt
    # or use HF datasets: from datasets import load_dataset; ds = load_dataset("imagenet-1k")["validation"]
    class_names = [
        ...
    ]  # list of 1000 strings, e.g. ["tench", "goldfish", ..., "laptop"]

    # Prompt engineering — very important!
    text_inputs = [prompt_template.format(name) for name in class_names]

    # Precompute text embeddings (only once!)
    text_tokens = processor(text=text_inputs, return_tensors="pt", padding=True)
    text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
    text_embeds = model.get_text_features(**text_tokens)  # (1000, dim)
    text_embeds = F.normalize(text_embeds, dim=-1)

    correct = 0
    total = 0

    for images, labels in tqdm(val_loader, desc="Zero-shot eval"):
        images = images.to(device)

        # Process images (SigLIP expects pixel_values)
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(
            device
        )

        image_embeds = model.get_image_features(pixel_values=pixel_values)
        image_embeds = F.normalize(image_embeds, dim=-1)

        # Similarity = dot product
        logits = image_embeds @ text_embeds.T  # (B, 1000)
        preds = logits.argmax(dim=1)

        correct += (preds.cpu() == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Zero-shot ImageNet top-1: {accuracy:.2f}%")
    return accuracy


processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-256")


def run(model, step, device="cuda"):
    # Every 500–1000 steps (adjust to your total_steps ~6000)
    if step % 750 == 0 and step > 0:
        _ = evaluate_zero_shot_imagenet(
            model.vision_model,  # ← your image encoder part!
            processor,
            "/path/to/imagenet/val",
            device=device,
            batch_size=256,
        )
    # print(acc)
    # log acc somewhere (wandb, tensorboard, print)


if __name__ == "__main__":
    raise RuntimeError(
        "This file is not meant to be run directly.\n"
        "Import the functions you need instead."
    )
