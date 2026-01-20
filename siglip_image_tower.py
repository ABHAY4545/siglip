from weight_init import default_flax_embed_init, lecun_normal_
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, img_size, num_channels, patch_size, d_model):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(
            num_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
        )
        self.position_embedding = nn.Embedding(self.num_patches, d_model)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False,
        )

    def forward(self, x):
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        x = x + self.position_embedding(self.position_ids)
        return x


class SiglipMLP(nn.Module):
    def __init__(self, d_model, mlp_ratio, dropout_rate):
        super().__init__()
        self.d_model = d_model
        self.mlp_ratio = int(mlp_ratio)
        self.dropout_rate = dropout_rate

        self.activation_fn = nn.GELU(approximate="tanh")
        self.fc1 = nn.Linear(self.d_model, self.mlp_ratio * self.d_model)
        self.fc2 = nn.Linear(self.mlp_ratio * self.d_model, self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SiglipAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, num_heads, d_model, mlp_ratio, dropout_rate):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.self_attn = SiglipAttention(num_heads, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.mlp = SiglipMLP(d_model, mlp_ratio, dropout_rate)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class SiglipEncoder(nn.Module):
    def __init__(self, num_heads, num_layers, d_model, mlp_ratio, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    num_heads=num_heads,
                    d_model=d_model,
                    mlp_ratio=mlp_ratio,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x


class SiglipMultiHeadAttentionPoolingHead(nn.Module):
    def __init__(self, num_heads, d_model, mlp_ratio, dropout_rate):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, d_model))
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)
        self.mlp = SiglipMLP(d_model, mlp_ratio, dropout_rate)

    def forward(self, x):
        B = x.shape[0]
        probe = self.probe.expand(B, -1, -1)
        x = self.attention(probe, x, x)[0]
        res = x
        x = self.layernorm(x)
        x = self.mlp(x) + res

        return x[:, 0]


class SiglipVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size,
        dropout_rate,
        num_layers=12,
        num_channels=3,
        patch_size=16,
        d_model=768,
        num_heads=12,
        mlp_ratio=4,
    ):
        super().__init__()

        self.embeddings = SiglipVisionEmbeddings(
            img_size=img_size,
            num_channels=num_channels,
            patch_size=patch_size,
            d_model=d_model,
        )

        self.encoder = SiglipEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
        )

        self.post_layernorm = nn.LayerNorm(d_model, eps=1e-6)

        self.head = SiglipMultiHeadAttentionPoolingHead(
            num_heads=num_heads,
            d_model=d_model,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
        )
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.encoder(x)
        x = self.post_layernorm(x)
        x = self.head(x)
        return x

    def _init_weights(self, m: nn.Module):
        if isinstance(m, SiglipVisionEmbeddings):
            width = int(self.embeddings.patch_embedding.out_channels)
            nn.init.normal_(m.position_embedding.weight, std=1 / math.sqrt((width)))
        elif isinstance(m, nn.Embedding):
            default_flax_embed_init(m.weight)
        elif isinstance(m, SiglipAttention):
            nn.init.xavier_uniform_(m.qkv_proj.weight)
            nn.init.zeros_(m.qkv_proj.bias)
            nn.init.xavier_uniform_(m.out_proj.weight)
            nn.init.zeros_(m.out_proj.bias)
        elif isinstance(m, SiglipMLP):
            nn.init.xavier_uniform_(m.fc1.weight)
            nn.init.zeros_(m.fc1.bias)
            nn.init.xavier_uniform_(m.fc2.weight)
            nn.init.zeros_(m.fc2.bias)
        elif isinstance(m, SiglipMultiHeadAttentionPoolingHead):
            nn.init.xavier_uniform_(m.probe.data)
            nn.init.xavier_normal_(m.attention.in_proj_weight.data)
            nn.init.zeros_(m.attention.in_proj_bias.data)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            lecun_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


if __name__ == "__main__":
    raise RuntimeError(
        "This file is not meant to be run directly.\n"
        "Import the functions you need instead."
    )
