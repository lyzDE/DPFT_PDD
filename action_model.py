
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm
import math
from torch.nn import TransformerEncoderLayer
import numpy as np
import os
import pickle
from TEM_ATTN import TemporalContextFusion
from SSAM import VisionEncoder,VisionEncoder_GAP,VisionEncoder_CBAM

from focal_loss import FocalLoss

from timm.models.layers import DropPath
from BIA import BidirectionalInteractiveAttention,AdditiveAttentionFusion, FrameSameModalFusion
from STA_GCN import STA_GCN
from TAF import TemporalBiLSTM,TemporalBiGRU

class ActionModel_ours(nn.Module):
    def __init__(self, vision_in_channels=768, embed_dim_pose=128,
                 embed_dim_vision=128, embed_dim_tem=128, num_heads_tem=4,
                 dropout=0.1, dropout_cls=0.3,vision_mid_channels=128,
                 ):
        super().__init__()

        self.pose_encoder = STA_GCN(embed_dim=embed_dim_pose,T=5)
        self.pose_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        _pe = torch.zeros(1, 5, 128)
        _pos = torch.arange(5).unsqueeze(1).float()
        _div = torch.exp(torch.arange(0, 128, 2).float() * (-math.log(10000.0) / 128))
        _pe[0, :, 0::2] = torch.sin(_pos * _div)
        _pe[0, :, 1::2] = torch.cos(_pos * _div)
        self.register_buffer('pose_pos_embed', _pe)

        self.vision_encoder = VisionEncoder(
            in_channels=vision_in_channels,
            mid_channels=vision_mid_channels,
            H=16, W=16,
            dropout=dropout
        )

        self.align_v = nn.Identity() if embed_dim_vision == embed_dim_tem else nn.Linear(embed_dim_vision, embed_dim_tem)
        self.align_p = nn.Identity() if embed_dim_pose == embed_dim_tem else nn.Linear(embed_dim_pose, embed_dim_tem)

        self.frame_fusion = BidirectionalInteractiveAttention(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            dropout=dropout,
            num_layers=2
        )

        self.temporal_transformer = TemporalContextFusion(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            num_layers=2,
            dropout=dropout
        )

        self.norm_p = nn.LayerNorm(embed_dim_tem)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim_tem, 128),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(128, 3)
        )

    def forward(self, vision, keypoints, sample=True, return_kl=True, K=1):
        if keypoints.dim() == 3 and keypoints.shape[-1] == 36:
            B, T, _ = keypoints.shape
            pose = keypoints.view(B, T, 18, 2)
        elif keypoints.dim() == 4 and keypoints.shape[-2:] == (18, 2):
            pose = keypoints
        else:
            raise ValueError(f"Unsupported keypoints shape: {keypoints.shape}")

        v_feat, attn_list = self.vision_encoder(vision)
        p_feat = self.pose_encoder(pose)
        p_feat = p_feat + self.pose_pos_embed
        p_feat = self.pose_transformer(p_feat)
        p_feat = self.norm_p(p_feat)

        v = self.align_v(v_feat)
        p = self.align_p(p_feat)
        fused_feat = self.frame_fusion(v, p)
        temporal_feat = self.temporal_transformer(fused_feat)
        logits = self.classifier(temporal_feat)
        return logits

class ActionModel_V1(nn.Module):
    def __init__(self, vision_in_channels=1024, embed_dim_pose=128,
                 embed_dim_vision=128, embed_dim_tem=128, num_heads_tem=4,
                 dropout=0.1, dropout_cls=0.3,vision_mid_channels=128,
                 ):
        super().__init__()

        self.pose_encoder = STA_GCN(embed_dim=embed_dim_pose,T=5)
        self.pose_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        _pe = torch.zeros(1, 5, 128)
        _pos = torch.arange(5).unsqueeze(1).float()
        _div = torch.exp(torch.arange(0, 128, 2).float() * (-math.log(10000.0) / 128))
        _pe[0, :, 0::2] = torch.sin(_pos * _div)
        _pe[0, :, 1::2] = torch.cos(_pos * _div)
        self.register_buffer('pose_pos_embed', _pe)

        self.vision_encoder = VisionEncoder(
            in_channels=vision_in_channels,
            mid_channels=vision_mid_channels,
            H=8, W=8,
            dropout=dropout
        )

        self.align_v = nn.Identity() if embed_dim_vision == embed_dim_tem else nn.Linear(embed_dim_vision, embed_dim_tem)
        self.align_p = nn.Identity() if embed_dim_pose == embed_dim_tem else nn.Linear(embed_dim_pose, embed_dim_tem)

        self.frame_fusion = BidirectionalInteractiveAttention(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            dropout=dropout,
            num_layers=2
        )

        self.temporal_transformer = TemporalContextFusion(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            num_layers=2,
            dropout=dropout
        )

        self.norm_p = nn.LayerNorm(embed_dim_tem)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim_tem, 128),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(128, 3)
        )

    def forward(self, vision, keypoints, sample=True, return_kl=True, K=1):
        if keypoints.dim() == 3 and keypoints.shape[-1] == 36:
            B, T, _ = keypoints.shape
            pose = keypoints.view(B, T, 18, 2)
        elif keypoints.dim() == 4 and keypoints.shape[-2:] == (18, 2):
            pose = keypoints
        else:
            raise ValueError(f"Unsupported keypoints shape: {keypoints.shape}")
        v_feat, attn_list = self.vision_encoder(vision)
        p_feat = self.pose_encoder(pose)
        p_feat = p_feat + self.pose_pos_embed
        p_feat = self.pose_transformer(p_feat)
        p_feat = self.norm_p(p_feat)

        v = self.align_v(v_feat)
        p = self.align_p(p_feat)
        fused_feat = self.frame_fusion(v, p)
        temporal_feat = self.temporal_transformer(fused_feat)
        logits = self.classifier(temporal_feat)
        return logits

class ActionModel_V2(nn.Module):
    def __init__(self, vision_in_channels=768, embed_dim_pose=128,
                 embed_dim_vision=128, embed_dim_tem=128, num_heads_tem=4,
                 dropout=0.1, dropout_cls=0.3,vision_mid_channels=128,
                 ):
        super().__init__()

        self.pose_encoder = STA_GCN(embed_dim=embed_dim_pose,T=5)
        self.pose_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        _pe = torch.zeros(1, 5, 128)
        _pos = torch.arange(5).unsqueeze(1).float()
        _div = torch.exp(torch.arange(0, 128, 2).float() * (-math.log(10000.0) / 128))
        _pe[0, :, 0::2] = torch.sin(_pos * _div)
        _pe[0, :, 1::2] = torch.cos(_pos * _div)
        self.register_buffer('pose_pos_embed', _pe)

        self.vision_encoder = VisionEncoder(
            in_channels=vision_in_channels,
            mid_channels=vision_mid_channels,
            H=14, W=14,
            dropout=dropout
        )

        self.align_v = nn.Identity() if embed_dim_vision == embed_dim_tem else nn.Linear(embed_dim_vision, embed_dim_tem)
        self.align_p = nn.Identity() if embed_dim_pose == embed_dim_tem else nn.Linear(embed_dim_pose, embed_dim_tem)

        self.frame_fusion = BidirectionalInteractiveAttention(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            dropout=dropout,
            num_layers=2
        )

        self.temporal_transformer = TemporalContextFusion(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            num_layers=2,
            dropout=dropout
        )

        self.norm_p = nn.LayerNorm(embed_dim_tem)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim_tem, 128),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(128, 3)
        )

    def forward(self, vision, keypoints, sample=True, return_kl=True, K=1):
        if keypoints.dim() == 3 and keypoints.shape[-1] == 36:
            B, T, _ = keypoints.shape
            pose = keypoints.view(B, T, 18, 2)
        elif keypoints.dim() == 4 and keypoints.shape[-2:] == (18, 2):
            pose = keypoints
        else:
            raise ValueError(f"Unsupported keypoints shape: {keypoints.shape}")

        v_feat, attn_list = self.vision_encoder(vision)
        p_feat = self.pose_encoder(pose)
        p_feat = p_feat + self.pose_pos_embed
        p_feat = self.pose_transformer(p_feat)
        p_feat = self.norm_p(p_feat)

        v = self.align_v(v_feat)
        p = self.align_p(p_feat)
        fused_feat = self.frame_fusion(v, p)
        temporal_feat = self.temporal_transformer(fused_feat)
        logits = self.classifier(temporal_feat)
        return logits

class ActionModel_V3(nn.Module):
    def __init__(self, vision_in_channels=512, embed_dim_pose=128,
                 embed_dim_vision=128, embed_dim_tem=128, num_heads_tem=4,
                 dropout=0.1, dropout_cls=0.3,vision_mid_channels=128,
                 ):
        super().__init__()
        self.pose_encoder = STA_GCN(embed_dim=embed_dim_pose,T=5)
        self.pose_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        _pe = torch.zeros(1, 5, 128)
        _pos = torch.arange(5).unsqueeze(1).float()
        _div = torch.exp(torch.arange(0, 128, 2).float() * (-math.log(10000.0) / 128))
        _pe[0, :, 0::2] = torch.sin(_pos * _div)
        _pe[0, :, 1::2] = torch.cos(_pos * _div)
        self.register_buffer('pose_pos_embed', _pe)

        self.vision_encoder = VisionEncoder(
            in_channels=vision_in_channels,
            mid_channels=vision_mid_channels,
            H=7, W=7,
            dropout=dropout
        )

        self.align_v = nn.Identity() if embed_dim_vision == embed_dim_tem else nn.Linear(embed_dim_vision, embed_dim_tem)
        self.align_p = nn.Identity() if embed_dim_pose == embed_dim_tem else nn.Linear(embed_dim_pose, embed_dim_tem)

        self.frame_fusion = BidirectionalInteractiveAttention(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            dropout=dropout,
            num_layers=2
        )

        self.temporal_transformer = TemporalContextFusion(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            num_layers=2,
            dropout=dropout
        )

        self.norm_p = nn.LayerNorm(embed_dim_tem)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim_tem, 128),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(128, 3)
        )

    def forward(self, vision, keypoints, sample=True, return_kl=True, K=1):
        if keypoints.dim() == 3 and keypoints.shape[-1] == 36:
            B, T, _ = keypoints.shape
            pose = keypoints.view(B, T, 18, 2)
        elif keypoints.dim() == 4 and keypoints.shape[-2:] == (18, 2):
            pose = keypoints
        else:
            raise ValueError(f"Unsupported keypoints shape: {keypoints.shape}")

        v_feat, attn_list = self.vision_encoder(vision)
        p_feat = self.pose_encoder(pose)
        p_feat = p_feat + self.pose_pos_embed
        p_feat = self.pose_transformer(p_feat)
        p_feat = self.norm_p(p_feat)

        v = self.align_v(v_feat)
        p = self.align_p(p_feat)
        fused_feat = self.frame_fusion(v, p)
        temporal_feat = self.temporal_transformer(fused_feat)
        logits = self.classifier(temporal_feat)
        return logits

class ActionModel_V4(nn.Module):
    def __init__(self, vision_in_channels=768, embed_dim_pose=128,
                 embed_dim_vision=128, embed_dim_tem=128, num_heads_tem=4,
                 dropout=0.1, dropout_cls=0.3,vision_mid_channels=128,
                 ):
        super().__init__()

        self.pose_encoder = STA_GCN(embed_dim=embed_dim_pose,T=5)
        self.pose_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        _pe = torch.zeros(1, 5, 128)
        _pos = torch.arange(5).unsqueeze(1).float()
        _div = torch.exp(torch.arange(0, 128, 2).float() * (-math.log(10000.0) / 128))
        _pe[0, :, 0::2] = torch.sin(_pos * _div)
        _pe[0, :, 1::2] = torch.cos(_pos * _div)
        self.register_buffer('pose_pos_embed', _pe)

        self.vision_encoder = VisionEncoder(
            in_channels=vision_in_channels,
            mid_channels=vision_mid_channels,
            H=16, W=16,
            dropout=dropout
        )

        self.align_v = nn.Identity() if embed_dim_vision == embed_dim_tem else nn.Linear(embed_dim_vision, embed_dim_tem)
        self.align_p = nn.Identity() if embed_dim_pose == embed_dim_tem else nn.Linear(embed_dim_pose, embed_dim_tem)

        self.frame_fusion = BidirectionalInteractiveAttention(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            dropout=dropout,
            num_layers=2
        )

        self.temporal_lstm = TemporalBiLSTM(
            dim=embed_dim_tem,
            num_layers=2,
            dropout=dropout
        )

        self.norm_p = nn.LayerNorm(embed_dim_tem)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim_tem, 128),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(128, 3)
        )

    def forward(self, vision, keypoints, sample=True, return_kl=True, K=1):
        if keypoints.dim() == 3 and keypoints.shape[-1] == 36:
            B, T, _ = keypoints.shape
            pose = keypoints.view(B, T, 18, 2)
        elif keypoints.dim() == 4 and keypoints.shape[-2:] == (18, 2):
            pose = keypoints
        else:
            raise ValueError(f"Unsupported keypoints shape: {keypoints.shape}")

        v_feat, attn_list = self.vision_encoder(vision)
        p_feat = self.pose_encoder(pose)
        p_feat = p_feat + self.pose_pos_embed
        p_feat = self.pose_transformer(p_feat)
        p_feat = self.norm_p(p_feat)

        v = self.align_v(v_feat)
        p = self.align_p(p_feat)
        fused_feat = self.frame_fusion(v, p)
        temporal_feat = self.temporal_lstm(fused_feat)
        logits = self.classifier(temporal_feat)
        return logits

class ActionModel_V5(nn.Module):
    def __init__(self, vision_in_channels=768, embed_dim_pose=128,
                 embed_dim_vision=128, embed_dim_tem=128, num_heads_tem=4,
                 dropout=0.1, dropout_cls=0.3,vision_mid_channels=128,
                 ):
        super().__init__()

        self.pose_encoder = STA_GCN(embed_dim=embed_dim_pose,T=5)
        self.pose_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        _pe = torch.zeros(1, 5, 128)
        _pos = torch.arange(5).unsqueeze(1).float()
        _div = torch.exp(torch.arange(0, 128, 2).float() * (-math.log(10000.0) / 128))
        _pe[0, :, 0::2] = torch.sin(_pos * _div)
        _pe[0, :, 1::2] = torch.cos(_pos * _div)
        self.register_buffer('pose_pos_embed', _pe)

        self.vision_encoder = VisionEncoder(
            in_channels=vision_in_channels,
            mid_channels=vision_mid_channels,
            H=16, W=16,
            dropout=dropout
        )

        self.align_v = nn.Identity() if embed_dim_vision == embed_dim_tem else nn.Linear(embed_dim_vision, embed_dim_tem)
        self.align_p = nn.Identity() if embed_dim_pose == embed_dim_tem else nn.Linear(embed_dim_pose, embed_dim_tem)

        self.frame_fusion = BidirectionalInteractiveAttention(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            dropout=dropout,
            num_layers=2
        )

        self.temporal_gru = TemporalBiGRU(
            dim=embed_dim_tem,
            num_layers=2,
            dropout=dropout
        )

        self.norm_p = nn.LayerNorm(embed_dim_tem)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim_tem, 128),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(128, 3)
        )

    def forward(self, vision, keypoints, sample=True, return_kl=True, K=1):
        if keypoints.dim() == 3 and keypoints.shape[-1] == 36:
            B, T, _ = keypoints.shape
            pose = keypoints.view(B, T, 18, 2)
        elif keypoints.dim() == 4 and keypoints.shape[-2:] == (18, 2):
            pose = keypoints
        else:
            raise ValueError(f"Unsupported keypoints shape: {keypoints.shape}")

        v_feat, attn_list = self.vision_encoder(vision)
        p_feat = self.pose_encoder(pose)
        p_feat = p_feat + self.pose_pos_embed
        p_feat = self.pose_transformer(p_feat)
        p_feat = self.norm_p(p_feat)

        v = self.align_v(v_feat)
        p = self.align_p(p_feat)
        fused_feat = self.frame_fusion(v, p)
        temporal_feat = self.temporal_gru(fused_feat)
        logits = self.classifier(temporal_feat)
        return logits

class ActionModel_V6(nn.Module):
    def __init__(self, vision_in_channels=768, embed_dim_pose=128,
                 embed_dim_vision=128, embed_dim_tem=128, num_heads_tem=4,
                 dropout=0.1, dropout_cls=0.3,vision_mid_channels=128,
                 ):
        super().__init__()

        self.pose_encoder = STA_GCN(embed_dim=embed_dim_pose,T=5)
        self.pose_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        _pe = torch.zeros(1, 5, 128)
        _pos = torch.arange(5).unsqueeze(1).float()
        _div = torch.exp(torch.arange(0, 128, 2).float() * (-math.log(10000.0) / 128))
        _pe[0, :, 0::2] = torch.sin(_pos * _div)
        _pe[0, :, 1::2] = torch.cos(_pos * _div)
        self.register_buffer('pose_pos_embed', _pe)

        self.vision_encoder = VisionEncoder_GAP(
            in_channels=vision_in_channels,
            mid_channels=vision_mid_channels,
            H=16, W=16,
            dropout=dropout
        )

        self.align_v = nn.Identity() if embed_dim_vision == embed_dim_tem else nn.Linear(embed_dim_vision, embed_dim_tem)
        self.align_p = nn.Identity() if embed_dim_pose == embed_dim_tem else nn.Linear(embed_dim_pose, embed_dim_tem)

        self.frame_fusion = BidirectionalInteractiveAttention(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            dropout=dropout,
            num_layers=2
        )

        self.temporal_transformer = TemporalContextFusion(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            num_layers=2,
            dropout=dropout
        )
        self.norm_p = nn.LayerNorm(embed_dim_tem)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim_tem, 128),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(128, 3)
        )

    def forward(self, vision, keypoints, sample=True, return_kl=True, K=1):
        if keypoints.dim() == 3 and keypoints.shape[-1] == 36:
            B, T, _ = keypoints.shape
            pose = keypoints.view(B, T, 18, 2)
        elif keypoints.dim() == 4 and keypoints.shape[-2:] == (18, 2):
            pose = keypoints
        else:
            raise ValueError(f"Unsupported keypoints shape: {keypoints.shape}")

        v_feat = self.vision_encoder(vision)
        p_feat = self.pose_encoder(pose)
        p_feat = p_feat + self.pose_pos_embed
        p_feat = self.pose_transformer(p_feat)
        p_feat = self.norm_p(p_feat)

        v = self.align_v(v_feat)
        p = self.align_p(p_feat)
        fused_feat = self.frame_fusion(v, p)
        temporal_feat = self.temporal_transformer(fused_feat)
        logits = self.classifier(temporal_feat)
        return logits

class ActionModel_V7(nn.Module):
    def __init__(self, vision_in_channels=768, embed_dim_pose=128,
                 embed_dim_vision=128, embed_dim_tem=128, num_heads_tem=4,
                 dropout=0.1, dropout_cls=0.3,vision_mid_channels=128,
                 ):
        super().__init__()

        self.pose_encoder = STA_GCN(embed_dim=embed_dim_pose,T=5)
        self.vision_encoder = VisionEncoder(
            in_channels=vision_in_channels,
            mid_channels=vision_mid_channels,
            H=16, W=16,
            dropout=dropout
        )

        self.align_v = nn.Identity() if embed_dim_vision == embed_dim_tem else nn.Linear(embed_dim_vision, embed_dim_tem)
        self.align_p = nn.Identity() if embed_dim_pose == embed_dim_tem else nn.Linear(embed_dim_pose, embed_dim_tem)

        self.frame_fusion = BidirectionalInteractiveAttention(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            dropout=dropout,
            num_layers=2
        )

        self.temporal_transformer = TemporalContextFusion(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            num_layers=2,
            dropout=dropout
        )

        self.norm_p = nn.LayerNorm(embed_dim_tem)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim_tem, 128),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(128, 3)
        )

    def forward(self, vision, keypoints, sample=True, return_kl=True, K=1):
        if keypoints.dim() == 3 and keypoints.shape[-1] == 36:
            B, T, _ = keypoints.shape
            pose = keypoints.view(B, T, 18, 2)
        elif keypoints.dim() == 4 and keypoints.shape[-2:] == (18, 2):
            pose = keypoints
        else:
            raise ValueError(f"Unsupported keypoints shape: {keypoints.shape}")

        v_feat, attn_list = self.vision_encoder(vision)
        p_feat = self.pose_encoder(pose)

        p_feat = self.norm_p(p_feat)

        v = self.align_v(v_feat)
        p = self.align_p(p_feat)
        fused_feat = self.frame_fusion(v, p)
        temporal_feat = self.temporal_transformer(fused_feat)
        logits = self.classifier(temporal_feat)
        return logits

class ActionModel_V8(nn.Module):

    def __init__(
        self,
        vision_in_channels=768,
        embed_dim_pose=128,
        embed_dim_vision=128,
        embed_dim_tem=128,
        num_heads_tem=4,
        dropout=0.1,
        dropout_cls=0.3,
        vision_mid_channels=128,
    ):
        super().__init__()

        self.pose_embed = nn.Linear(36, embed_dim_pose)

        self.pose_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim_pose,
                nhead=4,
                dim_feedforward=embed_dim_pose * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True
            ),
            num_layers=2
        )
        _pe = torch.zeros(1, 5, embed_dim_pose)
        _pos = torch.arange(5).unsqueeze(1).float()
        _div = torch.exp(torch.arange(0, embed_dim_pose, 2).float() * (-math.log(10000.0) / embed_dim_pose))
        _pe[0, :, 0::2] = torch.sin(_pos * _div)
        _pe[0, :, 1::2] = torch.cos(_pos * _div)
        self.register_buffer('pose_pos_embed', _pe)

        self.norm_p = nn.LayerNorm(embed_dim_pose)

        self.vision_encoder = VisionEncoder(
            in_channels=vision_in_channels,
            mid_channels=vision_mid_channels,
            H=16, W=16,
            dropout=dropout
        )

        self.align_v = nn.Identity() if embed_dim_vision == embed_dim_tem \
            else nn.Linear(embed_dim_vision, embed_dim_tem)

        self.align_p = nn.Identity() if embed_dim_pose == embed_dim_tem \
            else nn.Linear(embed_dim_pose, embed_dim_tem)

        self.frame_fusion = BidirectionalInteractiveAttention(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            dropout=dropout,
            num_layers=2
        )

        self.temporal_transformer = TemporalContextFusion(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            num_layers=2,
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim_tem, 128),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(128, 3)
        )

    def forward(self, vision, keypoints):

        if keypoints.dim() == 3 and keypoints.shape[-1] == 36:
            B, T, _ = keypoints.shape
            pose = keypoints
        elif keypoints.dim() == 4 and keypoints.shape[-2:] == (18, 2):
            B, T, _, _ = keypoints.shape
            pose = keypoints.view(B, T, -1)
        else:
            raise ValueError(f"Unsupported keypoints shape: {keypoints.shape}")

        v_feat,attn_list = self.vision_encoder(vision)

        p_feat = self.pose_embed(pose)
        p_feat = p_feat + self.pose_pos_embed
        p_feat = self.pose_transformer(p_feat)
        p_feat = self.norm_p(p_feat)

        v = self.align_v(v_feat)
        p = self.align_p(p_feat)

        fused_feat = self.frame_fusion(v, p)

        temporal_feat = self.temporal_transformer(fused_feat)

        logits = self.classifier(temporal_feat)

        return logits

class ActionModel_V9(nn.Module):
    def __init__(self, vision_in_channels=768, embed_dim_pose=128,
                 embed_dim_vision=128, embed_dim_tem=128, num_heads_tem=4,
                 dropout=0.1, dropout_cls=0.3,vision_mid_channels=128,
                 ):
        super().__init__()

        self.pose_encoder = STA_GCN(embed_dim=embed_dim_pose,T=5)
        self.pose_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        _pe = torch.zeros(1, 5, 128)
        _pos = torch.arange(5).unsqueeze(1).float()
        _div = torch.exp(torch.arange(0, 128, 2).float() * (-math.log(10000.0) / 128))
        _pe[0, :, 0::2] = torch.sin(_pos * _div)
        _pe[0, :, 1::2] = torch.cos(_pos * _div)
        self.register_buffer('pose_pos_embed', _pe)

        self.vision_encoder = VisionEncoder(
            in_channels=vision_in_channels,
            mid_channels=vision_mid_channels,
            H=16, W=16,
            dropout=dropout
        )

        self.align_v = nn.Identity() if embed_dim_vision == embed_dim_tem else nn.Linear(embed_dim_vision, embed_dim_tem)
        self.align_p = nn.Identity() if embed_dim_pose == embed_dim_tem else nn.Linear(embed_dim_pose, embed_dim_tem)

        self.temporal_transformer = TemporalContextFusion(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            num_layers=2,
            dropout=dropout
        )

        self.norm_p = nn.LayerNorm(embed_dim_tem)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim_tem, 128),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(128, 3)
        )

    def forward(self, vision, keypoints, sample=True, return_kl=True, K=1):
        if keypoints.dim() == 3 and keypoints.shape[-1] == 36:
            B, T, _ = keypoints.shape
            pose = keypoints.view(B, T, 18, 2)
        elif keypoints.dim() == 4 and keypoints.shape[-2:] == (18, 2):
            pose = keypoints
        else:
            raise ValueError(f"Unsupported keypoints shape: {keypoints.shape}")

        v_feat, attn_list = self.vision_encoder(vision)
        p_feat = self.pose_encoder(pose)
        p_feat = p_feat + self.pose_pos_embed
        p_feat = self.pose_transformer(p_feat)
        p_feat = self.norm_p(p_feat)

        v = self.align_v(v_feat)
        p = self.align_p(p_feat)
        fused_feat = v + p
        temporal_feat = self.temporal_transformer(fused_feat)
        logits = self.classifier(temporal_feat)
        return logits

class ActionModel_V10(nn.Module):
    def __init__(self, vision_in_channels=768, embed_dim_pose=128,
                 embed_dim_vision=128, embed_dim_tem=128, num_heads_tem=4,
                 dropout=0.1, dropout_cls=0.3,vision_mid_channels=128,
                 ):
        super().__init__()

        self.pose_encoder = STA_GCN(embed_dim=embed_dim_pose,T=5)
        self.pose_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        _pe = torch.zeros(1, 5, 128)
        _pos = torch.arange(5).unsqueeze(1).float()
        _div = torch.exp(torch.arange(0, 128, 2).float() * (-math.log(10000.0) / 128))
        _pe[0, :, 0::2] = torch.sin(_pos * _div)
        _pe[0, :, 1::2] = torch.cos(_pos * _div)
        self.register_buffer('pose_pos_embed', _pe)

        self.vision_encoder = VisionEncoder(
            in_channels=vision_in_channels,
            mid_channels=vision_mid_channels,
            H=16, W=16,
            dropout=dropout
        )

        self.align_v = nn.Identity() if embed_dim_vision == embed_dim_tem else nn.Linear(embed_dim_vision, embed_dim_tem)
        self.align_p = nn.Identity() if embed_dim_pose == embed_dim_tem else nn.Linear(embed_dim_pose, embed_dim_tem)

        self.frame_fusion = AdditiveAttentionFusion(
            dim=embed_dim_tem,
            hidden_dim=embed_dim_tem,
            dropout=dropout
        )

        self.temporal_transformer = TemporalContextFusion(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            num_layers=2,
            dropout=dropout
        )

        self.norm_p = nn.LayerNorm(embed_dim_tem)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim_tem, 128),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(128, 3)
        )

    def forward(self, vision, keypoints, sample=True, return_kl=True, K=1):
        if keypoints.dim() == 3 and keypoints.shape[-1] == 36:
            B, T, _ = keypoints.shape
            pose = keypoints.view(B, T, 18, 2)
        elif keypoints.dim() == 4 and keypoints.shape[-2:] == (18, 2):
            pose = keypoints
        else:
            raise ValueError(f"Unsupported keypoints shape: {keypoints.shape}")

        v_feat, attn_list = self.vision_encoder(vision)
        p_feat = self.pose_encoder(pose)
        p_feat = p_feat + self.pose_pos_embed
        p_feat = self.pose_transformer(p_feat)
        p_feat = self.norm_p(p_feat)

        v = self.align_v(v_feat)
        p = self.align_p(p_feat)
        fused_feat = self.frame_fusion(v, p)
        temporal_feat = self.temporal_transformer(fused_feat)
        logits = self.classifier(temporal_feat)
        return logits

class ActionModel_V11(nn.Module):
    def __init__(self, vision_in_channels=768, embed_dim_pose=128,
                 embed_dim_vision=128, embed_dim_tem=128, num_heads_tem=4,
                 dropout=0.1, dropout_cls=0.3,vision_mid_channels=128,
                 ):
        super().__init__()

        self.pose_encoder = STA_GCN(embed_dim=embed_dim_pose,T=5)
        self.pose_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        _pe = torch.zeros(1, 5, 128)
        _pos = torch.arange(5).unsqueeze(1).float()
        _div = torch.exp(torch.arange(0, 128, 2).float() * (-math.log(10000.0) / 128))
        _pe[0, :, 0::2] = torch.sin(_pos * _div)
        _pe[0, :, 1::2] = torch.cos(_pos * _div)
        self.register_buffer('pose_pos_embed', _pe)

        self.temporal_transformer = TemporalContextFusion(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            num_layers=2,
            dropout=dropout
        )

        self.norm_p = nn.LayerNorm(embed_dim_tem)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim_tem, 128),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(128, 3)
        )

    def forward(self, vision, keypoints, sample=True, return_kl=True, K=1):
        if keypoints.dim() == 3 and keypoints.shape[-1] == 36:
            B, T, _ = keypoints.shape
            pose = keypoints.view(B, T, 18, 2)
        elif keypoints.dim() == 4 and keypoints.shape[-2:] == (18, 2):
            pose = keypoints
        else:
            raise ValueError(f"Unsupported keypoints shape: {keypoints.shape}")

        p_feat = self.pose_encoder(pose)
        p_feat = p_feat + self.pose_pos_embed
        p_feat = self.pose_transformer(p_feat)
        p_feat = self.norm_p(p_feat)

        temporal_feat = self.temporal_transformer(p_feat)
        logits = self.classifier(temporal_feat)
        return logits

class ActionModel_V12(nn.Module):
    def __init__(self, vision_in_channels=768, embed_dim_pose=128,
                 embed_dim_vision=128, embed_dim_tem=128, num_heads_tem=4,
                 dropout=0.1, dropout_cls=0.3, vision_mid_channels=128,
                 ):
        super().__init__()

        self.vision_encoder = VisionEncoder(
            in_channels=vision_in_channels,
            mid_channels=vision_mid_channels,
            H=16, W=16,
            dropout=dropout
        )

        self.temporal_transformer = TemporalContextFusion(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            num_layers=2,
            dropout=dropout
        )

        self.norm_p = nn.LayerNorm(embed_dim_tem)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim_tem, 128),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(128, 3)
        )

    def forward(self, vision, keypoints, sample=True, return_kl=True, K=1):

        v_feat, attn_list = self.vision_encoder(vision)

        temporal_feat = self.temporal_transformer(v_feat)
        logits = self.classifier(temporal_feat)
        return logits

class ActionModel_V13(nn.Module):
    def __init__(self, vision_in_channels=768, embed_dim_pose=128,
                 embed_dim_vision=128, embed_dim_tem=128, num_heads_tem=4,
                 dropout=0.1, dropout_cls=0.3, vision_mid_channels=128,
                 ):
        super().__init__()

        self.pose_encoder = STA_GCN(embed_dim=embed_dim_pose, T=5)
        self.pose_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        _pe = torch.zeros(1, 5, 128)
        _pos = torch.arange(5).unsqueeze(1).float()
        _div = torch.exp(torch.arange(0, 128, 2).float() * (-math.log(10000.0) / 128))
        _pe[0, :, 0::2] = torch.sin(_pos * _div)
        _pe[0, :, 1::2] = torch.cos(_pos * _div)
        self.register_buffer('pose_pos_embed', _pe)

        self.vision_encoder = VisionEncoder_CBAM(
            in_channels=vision_in_channels,
            mid_channels=vision_mid_channels,
            H=16, W=16,
            dropout=dropout
        )

        self.align_v = nn.Identity() if embed_dim_vision == embed_dim_tem else nn.Linear(embed_dim_vision,
                                                                                         embed_dim_tem)
        self.align_p = nn.Identity() if embed_dim_pose == embed_dim_tem else nn.Linear(embed_dim_pose, embed_dim_tem)

        self.frame_fusion = BidirectionalInteractiveAttention(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            dropout=dropout,
            num_layers=2
        )

        self.temporal_transformer = TemporalContextFusion(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            num_layers=2,
            dropout=dropout
        )
        self.norm_p = nn.LayerNorm(embed_dim_tem)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim_tem, 128),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(128, 3)
        )

    def forward(self, vision, keypoints, sample=True, return_kl=True, K=1):
        if keypoints.dim() == 3 and keypoints.shape[-1] == 36:
            B, T, _ = keypoints.shape
            pose = keypoints.view(B, T, 18, 2)
        elif keypoints.dim() == 4 and keypoints.shape[-2:] == (18, 2):
            pose = keypoints
        else:
            raise ValueError(f"Unsupported keypoints shape: {keypoints.shape}")

        v_feat, _ = self.vision_encoder(vision)
        p_feat = self.pose_encoder(pose)
        p_feat = p_feat + self.pose_pos_embed
        p_feat = self.pose_transformer(p_feat)
        p_feat = self.norm_p(p_feat)

        v = self.align_v(v_feat)
        p = self.align_p(p_feat)
        fused_feat = self.frame_fusion(v, p)
        temporal_feat = self.temporal_transformer(fused_feat)
        logits = self.classifier(temporal_feat)
        return logits

class ActionModel_V14(nn.Module):
    def __init__(self, vision_in_channels=768, embed_dim_pose=128,
                 embed_dim_vision=128, embed_dim_tem=128, num_heads_tem=4,
                 dropout=0.1, dropout_cls=0.3, vision_mid_channels=128,
                 ):
        super().__init__()

        self.pose_encoder = STA_GCN(embed_dim=embed_dim_pose, T=5)
        self.pose_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        _pe = torch.zeros(1, 5, 128)
        _pos = torch.arange(5).unsqueeze(1).float()
        _div = torch.exp(torch.arange(0, 128, 2).float() * (-math.log(10000.0) / 128))
        _pe[0, :, 0::2] = torch.sin(_pos * _div)
        _pe[0, :, 1::2] = torch.cos(_pos * _div)
        self.register_buffer('pose_pos_embed', _pe)

        self.vision_encoder = VisionEncoder(
            in_channels=vision_in_channels,
            mid_channels=vision_mid_channels,
            H=16, W=16,
            dropout=dropout
        )

        self.align_v = nn.Identity() if embed_dim_vision == embed_dim_tem else nn.Linear(embed_dim_vision,
                                                                                         embed_dim_tem)
        self.align_p = nn.Identity() if embed_dim_pose == embed_dim_tem else nn.Linear(embed_dim_pose, embed_dim_tem)

        self.frame_fusion = FrameSameModalFusion(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            dropout=dropout,
            num_layers=2,
            output_mode="concat"
        )

        self.temporal_transformer = TemporalContextFusion(
            dim=embed_dim_tem*2,
            num_heads=num_heads_tem,
            num_layers=2,
            dropout=dropout
        )

        self.norm_p = nn.LayerNorm(embed_dim_tem)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim_tem* 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(128, 3)
        )

    def forward(self, vision, keypoints, sample=True, return_kl=True, K=1):
        if keypoints.dim() == 3 and keypoints.shape[-1] == 36:
            B, T, _ = keypoints.shape
            pose = keypoints.view(B, T, 18, 2)
        elif keypoints.dim() == 4 and keypoints.shape[-2:] == (18, 2):
            pose = keypoints
        else:
            raise ValueError(f"Unsupported keypoints shape: {keypoints.shape}")

        v_feat, attn_list = self.vision_encoder(vision)
        p_feat = self.pose_encoder(pose)
        p_feat = p_feat + self.pose_pos_embed
        p_feat = self.pose_transformer(p_feat)
        p_feat = self.norm_p(p_feat)

        v = self.align_v(v_feat)
        p = self.align_p(p_feat)
        fused_feat = self.frame_fusion(v, p)
        temporal_feat = self.temporal_transformer(fused_feat)
        logits = self.classifier(temporal_feat)
        return logits

class ActionModel_V15(nn.Module):
    def __init__(self, vision_in_channels=768, embed_dim_pose=128,
                 embed_dim_vision=128, embed_dim_tem=128, num_heads_tem=4,
                 dropout=0.1, dropout_cls=0.2, vision_mid_channels=128,
                 ):
        super().__init__()

        self.pose_encoder = STA_GCN(embed_dim=embed_dim_pose, T=5)
        self.pose_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        _pe = torch.zeros(1, 5, 128)
        _pos = torch.arange(5).unsqueeze(1).float()
        _div = torch.exp(torch.arange(0, 128, 2).float() * (-math.log(10000.0) / 128))
        _pe[0, :, 0::2] = torch.sin(_pos * _div)
        _pe[0, :, 1::2] = torch.cos(_pos * _div)
        self.register_buffer('pose_pos_embed', _pe)

        self.vision_encoder = VisionEncoder(
            in_channels=vision_in_channels,
            mid_channels=vision_mid_channels,
            H=14, W=14,
            dropout=dropout
        )

        self.align_v = nn.Identity() if embed_dim_vision == embed_dim_tem else nn.Linear(embed_dim_vision,
                                                                                         embed_dim_tem)
        self.align_p = nn.Identity() if embed_dim_pose == embed_dim_tem else nn.Linear(embed_dim_pose, embed_dim_tem)

        self.frame_fusion = BidirectionalInteractiveAttention(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            dropout=dropout,
            num_layers=2
        )

        self.temporal_transformer = TemporalContextFusion(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            num_layers=2,
            dropout=dropout
        )
        self.norm_p = nn.LayerNorm(embed_dim_tem)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim_tem, 128),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(128, 3)
        )

    def forward(self, vision, keypoints, sample=True, return_kl=True, K=1):
        if keypoints.dim() == 3 and keypoints.shape[-1] == 36:
            B, T, _ = keypoints.shape
            pose = keypoints.view(B, T, 18, 2)
        elif keypoints.dim() == 4 and keypoints.shape[-2:] == (18, 2):
            pose = keypoints
        else:
            raise ValueError(f"Unsupported keypoints shape: {keypoints.shape}")

        v_feat, attn_list = self.vision_encoder(vision)
        p_feat = self.pose_encoder(pose)
        p_feat = p_feat + self.pose_pos_embed
        p_feat = self.pose_transformer(p_feat)
        p_feat = self.norm_p(p_feat)

        v = self.align_v(v_feat)
        p = self.align_p(p_feat)
        fused_feat = self.frame_fusion(v, p)
        temporal_feat = self.temporal_transformer(fused_feat)
        logits = self.classifier(temporal_feat)
        return logits

class ActionModel_V16(nn.Module):
    def __init__(self, vision_in_channels=1280, embed_dim_pose=128,
                 embed_dim_vision=128, embed_dim_tem=128, num_heads_tem=4,
                 dropout=0.1, dropout_cls=0.2, vision_mid_channels=128,
                 ):
        super().__init__()

        self.pose_encoder = STA_GCN(embed_dim=embed_dim_pose, T=5)
        self.pose_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        _pe = torch.zeros(1, 5, 128)
        _pos = torch.arange(5).unsqueeze(1).float()
        _div = torch.exp(torch.arange(0, 128, 2).float() * (-math.log(10000.0) / 128))
        _pe[0, :, 0::2] = torch.sin(_pos * _div)
        _pe[0, :, 1::2] = torch.cos(_pos * _div)
        self.register_buffer('pose_pos_embed', _pe)

        self.vision_encoder = VisionEncoder(
            in_channels=vision_in_channels,
            mid_channels=vision_mid_channels,
            H=7, W=7,
            dropout=dropout
        )

        self.align_v = nn.Identity() if embed_dim_vision == embed_dim_tem else nn.Linear(embed_dim_vision,
                                                                                         embed_dim_tem)
        self.align_p = nn.Identity() if embed_dim_pose == embed_dim_tem else nn.Linear(embed_dim_pose, embed_dim_tem)

        self.frame_fusion = BidirectionalInteractiveAttention(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            dropout=dropout,
            num_layers=2
        )

        self.temporal_transformer = TemporalContextFusion(
            dim=embed_dim_tem,
            num_heads=num_heads_tem,
            num_layers=2,
            dropout=dropout
        )
        self.norm_p = nn.LayerNorm(embed_dim_tem)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim_tem, 128),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(128, 3)
        )

    def forward(self, vision, keypoints, sample=True, return_kl=True, K=1):
        if keypoints.dim() == 3 and keypoints.shape[-1] == 36:
            B, T, _ = keypoints.shape
            pose = keypoints.view(B, T, 18, 2)
        elif keypoints.dim() == 4 and keypoints.shape[-2:] == (18, 2):
            pose = keypoints
        else:
            raise ValueError(f"Unsupported keypoints shape: {keypoints.shape}")

        v_feat, attn_list = self.vision_encoder(vision)
        p_feat = self.pose_encoder(pose)
        p_feat = p_feat + self.pose_pos_embed
        p_feat = self.pose_transformer(p_feat)
        p_feat = self.norm_p(p_feat)

        v = self.align_v(v_feat)
        p = self.align_p(p_feat)
        fused_feat = self.frame_fusion(v, p)
        temporal_feat = self.temporal_transformer(fused_feat)
        logits = self.classifier(temporal_feat)
        return logits

def train_model(
    model,
    dataloaders,
    class_weights,
    num_epochs=30,
    lr=1e-4,
    save_dir=None,
    model_name="model",
    grad_clip=1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = FocalLoss(alpha=class_weights, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = -1.0
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}\n" + "-" * 30)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for features, keypoints, labels in tqdm(dataloaders[phase]):
                features = features.to(device, non_blocking=True)
                keypoints = keypoints.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                B = labels.size(0)
                total_samples += B

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    logits = model(features, keypoints)
                    loss = criterion(logits, labels)

                    preds = logits.argmax(dim=1)

                    if phase == "train":
                        loss.backward()
                        if grad_clip is not None and grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()

                running_loss += loss.item() * B
                running_corrects += (preds == labels).sum().item()

            epoch_loss = running_loss / max(1, total_samples)
            epoch_acc = running_corrects / max(1, total_samples)
            print(f"{phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                if save_dir is not None:
                    import os
                    best_path = os.path.join(save_dir, f"best_{model_name}.pth")
                    torch.save(model.state_dict(), best_path)
                    print(f"🏆 New best model saved: {best_path}  (Val Acc={best_val_acc:.4f})")

    print("✅ Training finished.")
    return model

def test_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for features, keypoints, labels in dataloader:
            features = features.to(device)
            keypoints = keypoints.to(device)
            labels = labels.to(device)

            logits = model(features, keypoints)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_preds, 0),
        np.concatenate(all_labels, 0),
        np.concatenate(all_probs, 0),
    )

