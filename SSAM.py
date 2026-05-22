import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialContextAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, cls_tok, patch_tok):
        B, N, C = patch_tok.shape

        q = self.q(cls_tok).reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(patch_tok).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn_logits = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn_logits.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, 1, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn

class SpatialContextAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.1, proj_drop=0.1, ffn_ratio=4):
        super().__init__()

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.attn = SpatialContextAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )

        self.drop_path = nn.Dropout(proj_drop)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(dim * ffn_ratio, dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, cls_tok, patch_tok):
        attn_out, attn = self.attn(
            self.norm_q(cls_tok),
            self.norm_kv(patch_tok)
        )
        cls_tok = cls_tok + self.drop_path(attn_out)

        cls_tok = cls_tok + self.drop_path(
            self.ffn(self.norm_ffn(cls_tok))
        )

        return cls_tok, attn
class VisionEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels=128,
        H=8, W=8,
        num_heads=4,
        dropout=0.1,
        num_refine_layers=2,
        ffn_ratio=4,
    ):
        super().__init__()
        self.H, self.W = H, W
        self.mid_channels = mid_channels
        self.num_refine_layers = num_refine_layers

        self.proj_1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, mid_channels))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.cls_attn_layers = nn.ModuleList([
            SpatialContextAttentionBlock(
                dim=mid_channels,
                num_heads=num_heads,
                attn_drop=dropout,
                proj_drop=dropout,
                ffn_ratio=ffn_ratio
            )
            for _ in range(num_refine_layers)
        ])

    def _to_nchw(self, vision, allowed_hw=(7, 8, 14, 16)):
        if vision.dim() == 5:
            B, T = vision.shape[0], vision.shape[1]

            if vision.shape[-2] in allowed_hw and vision.shape[-1] in allowed_hw:
                C, H, W = vision.shape[2], vision.shape[3], vision.shape[4]
                v = vision.reshape(B * T, C, H, W).contiguous()
                return v, B, T

            if vision.shape[2] in allowed_hw and vision.shape[3] in allowed_hw:
                H, W, C = vision.shape[2], vision.shape[3], vision.shape[4]
                v = vision.permute(0, 1, 4, 2, 3).contiguous().reshape(B * T, C, H, W)
                return v, B, T

            raise ValueError(f"Unsupported 5D vision shape: {vision.shape}")

        elif vision.dim() == 4:
            if vision.shape[-2] in allowed_hw and vision.shape[-1] in allowed_hw:
                return vision.contiguous(), None, None

            if vision.shape[1] in allowed_hw and vision.shape[2] in allowed_hw:
                v = vision.permute(0, 3, 1, 2).contiguous()
                return v, None, None

            raise ValueError(f"Unsupported 4D vision shape: {vision.shape}")
        else:
            raise ValueError(f"Unsupported vision shape: {vision.shape}")

    def forward(self, vision):
        v, B, T = self._to_nchw(vision)
        N, C, H, W = v.shape
        assert (H, W) == (self.H, self.W), f"Expect {(self.H, self.W)}, got {(H, W)}"

        x = self.proj_1x1(v)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)

        patch_tok = x.flatten(2).transpose(1, 2).contiguous()

        cls = self.cls_token.expand(N, -1, -1)

        attn_list = []
        for layer in self.cls_attn_layers:
            cls, attn = layer(cls, patch_tok)
            attn_list.append(attn)

        cls_out = cls.squeeze(1)

        if B is None or T is None:
            return cls_out.unsqueeze(1), attn_list

        cls_feat = cls_out.view(B, T, -1)
        return cls_feat, attn_list

class VisionEncoder_GAP(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels=128,
        H=16, W=16,
        dropout=0.1,
    ):
        super().__init__()
        self.H, self.W = H, W
        self.mid_channels = mid_channels

        self.proj_1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

        self.gap = nn.AdaptiveAvgPool2d(1)

    def _to_nchw(self, vision, allowed_hw=(7, 8, 14, 16)):
        if vision.dim() == 5:
            B, T = vision.shape[0], vision.shape[1]

            if vision.shape[-2] in allowed_hw and vision.shape[-1] in allowed_hw:
                C, H, W = vision.shape[2], vision.shape[3], vision.shape[4]
                v = vision.reshape(B * T, C, H, W).contiguous()
                return v, B, T

            if vision.shape[2] in allowed_hw and vision.shape[3] in allowed_hw:
                H, W, C = vision.shape[2], vision.shape[3], vision.shape[4]
                v = vision.permute(0, 1, 4, 2, 3).contiguous().reshape(B * T, C, H, W)
                return v, B, T

            raise ValueError(f"Unsupported 5D vision shape: {vision.shape}")

        elif vision.dim() == 4:
            if vision.shape[-2] in allowed_hw and vision.shape[-1] in allowed_hw:
                return vision.contiguous(), None, None

            if vision.shape[1] in allowed_hw and vision.shape[2] in allowed_hw:
                v = vision.permute(0, 3, 1, 2).contiguous()
                return v, None, None

            raise ValueError(f"Unsupported 4D vision shape: {vision.shape}")
        else:
            raise ValueError(f"Unsupported vision shape: {vision.shape}")

    def forward(self, vision):
        v, B, T = self._to_nchw(vision)
        N, C, H, W = v.shape
        assert (H, W) == (self.H, self.W)

        x = self.proj_1x1(v)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.gap(x)
        x = x.flatten(1)

        if B is None or T is None:
            return x.unsqueeze(1)

        x = x.view(B, T, -1)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction=reduction)
        self.spatial_attn = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x

class VisionEncoder_CBAM(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels=128,
        H=8, W=8,
        dropout=0.1,
        cbam_reduction=16,
        cbam_spatial_kernel=7,
        pool_type="avg",
    ):
        super().__init__()
        self.H, self.W = H, W
        self.mid_channels = mid_channels
        self.pool_type = pool_type

        self.proj_1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

        self.cbam = CBAM(
            channels=mid_channels,
            reduction=cbam_reduction,
            spatial_kernel=cbam_spatial_kernel
        )

    def _to_nchw(self, vision, allowed_hw=(7, 8, 14, 16)):
        if vision.dim() == 5:
            B, T = vision.shape[0], vision.shape[1]

            if vision.shape[-2] in allowed_hw and vision.shape[-1] in allowed_hw:
                C, H, W = vision.shape[2], vision.shape[3], vision.shape[4]
                v = vision.reshape(B * T, C, H, W).contiguous()
                return v, B, T

            if vision.shape[2] in allowed_hw and vision.shape[3] in allowed_hw:
                H, W, C = vision.shape[2], vision.shape[3], vision.shape[4]
                v = vision.permute(0, 1, 4, 2, 3).contiguous().reshape(B * T, C, H, W)
                return v, B, T

            raise ValueError(f"Unsupported 5D vision shape: {vision.shape}")

        elif vision.dim() == 4:
            if vision.shape[-2] in allowed_hw and vision.shape[-1] in allowed_hw:
                return vision.contiguous(), None, None

            if vision.shape[1] in allowed_hw and vision.shape[2] in allowed_hw:
                v = vision.permute(0, 3, 1, 2).contiguous()
                return v, None, None

            raise ValueError(f"Unsupported 4D vision shape: {vision.shape}")
        else:
            raise ValueError(f"Unsupported vision shape: {vision.shape}")

    def _global_pool(self, x):
        if self.pool_type == "avg":
            out = F.adaptive_avg_pool2d(x, 1).flatten(1)
        elif self.pool_type == "max":
            out = F.adaptive_max_pool2d(x, 1).flatten(1)
        elif self.pool_type == "avgmax":
            avg_out = F.adaptive_avg_pool2d(x, 1).flatten(1)
            max_out = F.adaptive_max_pool2d(x, 1).flatten(1)
            out = avg_out + max_out
        else:
            raise ValueError(f"Unsupported pool_type: {self.pool_type}")
        return out

    def forward(self, vision):
        v, B, T = self._to_nchw(vision)
        N, C, H, W = v.shape
        assert (H, W) == (self.H, self.W), f"Expect {(self.H, self.W)}, got {(H, W)}"

        x = self.proj_1x1(v)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.cbam(x)

        feat = self._global_pool(x)

        if B is None or T is None:
            return feat.unsqueeze(1), x

        cls_feat = feat.view(B, T, -1)
        fmap = x.view(B, T, self.mid_channels, H, W)
        return cls_feat, fmap
