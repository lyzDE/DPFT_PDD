import torch
import torch.nn as nn
import torch.nn.functional as F

class BiCrossAttentionFusionLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1, ffn_mul=2):
        super().__init__()

        self.attn_v = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_p = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_v1 = nn.LayerNorm(dim)
        self.norm_p1 = nn.LayerNorm(dim)

        self.norm_v2 = nn.LayerNorm(dim)
        self.norm_p2 = nn.LayerNorm(dim)

        self.ffn_v = nn.Sequential(
            nn.Linear(dim, dim * ffn_mul),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mul, dim),
            nn.Dropout(dropout),
        )
        self.ffn_p = nn.Sequential(
            nn.Linear(dim, dim * ffn_mul),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mul, dim),
            nn.Dropout(dropout),
        )

    def forward(self, v, p):
        v_q = self.norm_v1(v)
        p_kv = self.norm_p1(p)
        v_attn, _ = self.attn_v(query=v_q, key=p_kv, value=p_kv)
        v = v + v_attn
        v = v + self.ffn_v(self.norm_v2(v))

        p_q = self.norm_p1(p)
        v_kv = self.norm_v1(v)
        p_attn, _ = self.attn_p(query=p_q, key=v_kv, value=v_kv)
        p = p + p_attn
        p = p + self.ffn_p(self.norm_p2(p))

        return v, p

class BidirectionalInteractiveAttention(nn.Module):
    def __init__(self, dim=128, num_heads=4, dropout=0.1, num_layers=2, ffn_mul=4):
        super().__init__()
        self.layers = nn.ModuleList([
            BiCrossAttentionFusionLayer(dim, num_heads, dropout=dropout, ffn_mul=ffn_mul)
            for _ in range(num_layers)
        ])

    def forward(self, v, p):
        for layer in self.layers:
            v, p = layer(v, p)

        fused = v + p
        return fused

class AdditiveAttentionFusion(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1, return_alpha=False):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.Wv = nn.Linear(dim, hidden_dim, bias=True)
        self.Wp = nn.Linear(dim, hidden_dim, bias=False)
        self.w = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.return_alpha = return_alpha

    def forward(self, v, p):
        h = torch.tanh(self.Wv(v) + self.Wp(p))
        h = self.dropout(h)
        score = self.w(h)
        alpha = torch.sigmoid(score)
        fused = alpha * v + (1.0 - alpha) * p

        if self.return_alpha:
            return fused, alpha
        return fused

class SameFrameBiModalFusionLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1, ffn_mul=2):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_v = nn.Linear(dim, dim, bias=True)
        self.k_v = nn.Linear(dim, dim, bias=True)
        self.v_v = nn.Linear(dim, dim, bias=True)

        self.q_p = nn.Linear(dim, dim, bias=True)
        self.k_p = nn.Linear(dim, dim, bias=True)
        self.v_p = nn.Linear(dim, dim, bias=True)

        self.proj_v = nn.Linear(dim, dim)
        self.proj_p = nn.Linear(dim, dim)

        self.drop_attn = nn.Dropout(dropout)
        self.drop_proj = nn.Dropout(dropout)

        self.norm_v1 = nn.LayerNorm(dim)
        self.norm_p1 = nn.LayerNorm(dim)
        self.norm_v2 = nn.LayerNorm(dim)
        self.norm_p2 = nn.LayerNorm(dim)

        self.ffn_v = nn.Sequential(
            nn.Linear(dim, dim * ffn_mul),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mul, dim),
            nn.Dropout(dropout),
        )
        self.ffn_p = nn.Sequential(
            nn.Linear(dim, dim * ffn_mul),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mul, dim),
            nn.Dropout(dropout),
        )

    def _reshape_q(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).unsqueeze(3)

    def _reshape_kv(self, x):
        B, T, S, D = x.shape
        return x.view(B, T, S, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

    def forward(self, v, p):
        B, T, D = v.shape

        v_in = self.norm_v1(v)
        p_in = self.norm_p1(p)

        q_v = self._reshape_q(self.q_v(v_in))

        kv_v_in = torch.stack([v_in, p_in], dim=2)
        k_v = self._reshape_kv(self.k_v(kv_v_in))
        vv_v = self._reshape_kv(self.v_v(kv_v_in))

        attn_logits_v = torch.matmul(q_v * self.scale, k_v.transpose(-2, -1))
        attn_v = F.softmax(attn_logits_v, dim=-1)
        attn_v = self.drop_attn(attn_v)

        out_v = torch.matmul(attn_v, vv_v)
        out_v = out_v.squeeze(3).reshape(B, T, D)
        out_v = self.drop_proj(self.proj_v(out_v))

        v = v + out_v
        v = v + self.ffn_v(self.norm_v2(v))

        v_ref = self.norm_v1(v)
        p_ref = self.norm_p1(p)

        q_p = self._reshape_q(self.q_p(p_ref))

        kv_p_in = torch.stack([p_ref, v_ref], dim=2)
        k_p = self._reshape_kv(self.k_p(kv_p_in))
        vv_p = self._reshape_kv(self.v_p(kv_p_in))

        attn_logits_p = torch.matmul(q_p * self.scale, k_p.transpose(-2, -1))
        attn_p = F.softmax(attn_logits_p, dim=-1)
        attn_p = self.drop_attn(attn_p)

        out_p = torch.matmul(attn_p, vv_p)
        out_p = out_p.squeeze(3).reshape(B, T, D)
        out_p = self.drop_proj(self.proj_p(out_p))

        p = p + out_p
        p = p + self.ffn_p(self.norm_p2(p))

        return v, p, attn_v, attn_p

class FrameSameModalFusion(nn.Module):
    def __init__(self, dim=128, num_heads=4, dropout=0.1, num_layers=2, ffn_mul=4, output_mode="stack"):
        super().__init__()
        assert output_mode in ("stack", "concat", "sum")

        self.output_mode = output_mode
        self.layers = nn.ModuleList([
            SameFrameBiModalFusionLayer(dim, num_heads, dropout=dropout, ffn_mul=ffn_mul)
            for _ in range(num_layers)
        ])

    def forward(self, v, p, return_attn=False):
        attn_v_list, attn_p_list = [], []

        for layer in self.layers:
            v, p, attn_v, attn_p = layer(v, p)
            if return_attn:
                attn_v_list.append(attn_v)
                attn_p_list.append(attn_p)

        if self.output_mode == "stack":
            fused = torch.stack([v, p], dim=2)
        elif self.output_mode == "concat":
            fused = torch.cat([v, p], dim=-1)
        else:
            fused = v + p

        if return_attn:
            return fused, attn_v_list, attn_p_list
        return fused

