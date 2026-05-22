import torch
import torch.nn as nn

class TemporalTransformer(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=2, dropout=0.1, ff_mult=2):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        B, T, D = x.shape

        cls = self.cls_token.expand(B, 1, D)
        x = torch.cat([cls, x], dim=1)

        if key_padding_mask is not None:
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        x = self.norm(x)

        cls_out = x[:, 0]

        return cls_out

class TemporalBiLSTM(nn.Module):
    def __init__(self, dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True
        )
        self.proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        h_fw = h_n[-2]
        h_bw = h_n[-1]
        h = torch.cat([h_fw, h_bw], dim=-1)
        feat = self.proj(h)
        feat = self.norm(feat)
        return feat

class TemporalBiGRU(nn.Module):
    def __init__(self, dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True
        )
        self.proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out, h_n = self.gru(x)
        h_fw = h_n[-2]
        h_bw = h_n[-1]
        h = torch.cat([h_fw, h_bw], dim=-1)

        feat = self.proj(h)
        feat = self.norm(feat)
        return feat
