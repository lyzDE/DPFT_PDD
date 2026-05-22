import torch
import torch.nn as nn

class AsymmetricCrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1, ffn_ratio=4):
        super().__init__()

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout1 = nn.Dropout(dropout)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_ratio, dim),
            nn.Dropout(dropout)
        )

    def forward(self, cls_token, x):
        q = self.norm_q(cls_token)
        kv = self.norm_kv(x)

        attn_out, _ = self.attn(query=q, key=kv, value=kv)
        cls_token = cls_token + self.dropout1(attn_out)

        cls_token = cls_token + self.ffn(self.norm_ffn(cls_token))

        return cls_token
class TemporalContextFusion(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=2, dropout=0.1, ffn_ratio=4):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.layers = nn.ModuleList([
            AsymmetricCrossAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                ffn_ratio=ffn_ratio
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)

        for layer in self.layers:
            cls_token = layer(cls_token, x)

        cls_token = self.final_norm(cls_token)
        return cls_token.squeeze(1)
