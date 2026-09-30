# SigLIP — From-Scratch PyTorch Implementation

A PyTorch reimplementation of the [SigLIP](https://arxiv.org/abs/2303.15343) dual encoder and its pairwise sigmoid objective. Both towers are defined in this repository rather than loaded as a ready-made model. The vision tower is initialized here and trained on image–caption pairs; the text tower receives weights from `google/siglip2-base-patch16-256` and stays frozen. The project has been trained on CC12M for roughly 4–5 epochs.

## Model

- **Vision tower:** 256 × 256 RGB images, 16 × 16 patches, 12 transformer layers, and attention pooling (`siglip_image_tower.py`).
- **Text tower:** 12 transformer layers with pretrained text weights transferred by `model_surgery.py` (`siglip_text_tower.py`).
- **Training objective:** normalized image and text embeddings, a learned logit scale and bias, and SigLIP's pairwise sigmoid loss (`siglip.py`).

The model produces an image–text similarity matrix for a batch. Matching pairs are on its diagonal; all other pairs in the batch act as negatives.

## Setup

Use Python 3.11 or newer and install the dependencies declared in `pyproject.toml`:

```bash
uv sync
```

Training is designed for a GPU with a compatible PyTorch build and BF16 support. The first run needs network access for the Hugging Face text weights and processor, plus the CC12M shards. If you use ROCm, ensure that the PyTorch build in your environment supports your GPU.

## Data and training

`train.py` streams WebDataset shards from the [CC12M dataset on Hugging Face](https://huggingface.co/datasets/pixparse/cc12m-wds). Its configured training split is shards `0000`–`2170`; validation uses `2171`–`2172`. Each sample needs matching `.jpg`, `.txt`, and `.json` entries. The metadata filter keeps samples with `status: "success"` and width and height of at least 256 pixels.

The intended single-GPU entrypoint is:

```bash
mkdir -p logs
uv run python train.py
```

Training settings are constants near the top of `train.py`. The current defaults include a per-device batch of 512, a global batch of 4096 through gradient accumulation, two epochs over a configured 10,240,000 samples, and a warmup followed by cosine learning-rate decay. Adjust these constants for your hardware and training run.

The trainer writes step logs to `logs/training.log`. Every 250 optimizer steps, it saves a checkpoint under `checkpoints/`, evaluates retrieval on the validation shards, and appends loss, R@1, R@5, and margin to `logs/validation.log`. Checkpoints and logs are excluded from Git.

## Repository guide

| File | Purpose |
| --- | --- |
| `train.py` | WebDataset pipeline and training loop |
| `siglip.py` | Dual encoder, loss, and learning-rate schedule |
| `siglip_image_tower.py` | Vision transformer |
| `siglip_text_tower.py` | Text transformer |
| `model_surgery.py` | Transfers pretrained text weights into the custom model |
| `utils.py` | Preprocessing, filtering, and checkpoint helpers |
| `val.py` | Retrieval metrics and validation |
| `DDPManager.py` | Distributed process setup |
| `training.ipynb` | Training notebook |
| `imagenet_eval.py` | Experimental ImageNet evaluation scaffold |
