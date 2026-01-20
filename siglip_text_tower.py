import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask


class SiglipTextEmbeddings(nn.Module):
    position_ids: Tensor

    def __init__(self, vocab_size, d_model, max_seq_len=64):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.register_buffer(
            "position_ids", torch.arange(max_seq_len).unsqueeze(0), persistent=False
        )

    def forward(self, x):
        seq_len = x.shape[1]
        x = self.token_embedding(x)
        positions = self.position_ids[:, :seq_len]
        x = x + self.position_embedding(positions)
        return x


class SiglipMLP(nn.Module):
    def __init__(self, d_model, mlp_ratio):
        super().__init__()
        self.d_model = d_model
        self.mlp_ratio = int(mlp_ratio)
        self.activation_fn = nn.GELU(approximate="tanh")
        self.fc1 = nn.Linear(self.d_model, self.mlp_ratio * self.d_model)
        self.fc2 = nn.Linear(self.mlp_ratio * self.d_model, self.d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class SiglipAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x, attn_bias):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_bias, is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, num_heads, d_model, mlp_ratio):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.self_attn = SiglipAttention(num_heads, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.mlp = SiglipMLP(d_model, mlp_ratio)

    def forward(self, x, attn_mask):
        x = x + self.self_attn(self.layer_norm1(x), attn_mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class SiglipEncoder(nn.Module):
    def __init__(self, num_heads, num_layers, d_model, mlp_ratio):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    num_heads=num_heads, d_model=d_model, mlp_ratio=mlp_ratio
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, attn_mask):
        for block in self.layers:
            x = block(x, attn_mask)
        return x


class SiglipTextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=256000,
        num_layers=12,
        d_model=768,
        num_heads=12,
        mlp_ratio=4.0,
        max_seq_len=64,
    ):
        super().__init__()
        self.embeddings = SiglipTextEmbeddings(
            vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len
        )
        self.encoder = SiglipEncoder(num_heads, num_layers, d_model, mlp_ratio)
        self.final_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.head = nn.Linear(d_model, d_model)

    def forward(self, input_ids, attn_mask):
        x = self.embeddings(input_ids)
        attn_bias = _prepare_4d_attention_mask(
            attn_mask,
            dtype=x.dtype,
            tgt_len=x.shape[1],
        )
        x = self.encoder(x, attn_bias)
        x = self.final_layer_norm(x)
        pooled = x[:, -1, :]
        pooled = self.head(pooled)
        return pooled


if __name__ == "__main__":
    raise RuntimeError(
        "This file is not meant to be run directly.\n"
        "Import the functions you need instead."
    )
