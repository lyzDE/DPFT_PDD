import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, embed_dim, reduction=4, T=5):
        super().__init__()
        self.embed_dim = embed_dim
        self.reduction = reduction
        self.T = T

        self.spatial_conv = nn.Conv1d(embed_dim, 1, kernel_size=1)
        self.temporal_conv = nn.Conv1d(embed_dim, 1, kernel_size=1)

        self.channel_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // reduction, embed_dim),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B_T, C, N = x.shape
        T = self.T

        if B_T % T != 0:
            raise ValueError(f"B*T={B_T} is not divisible by T={T}")

        B = B_T // T
        x4 = x.view(B, T, C, N)

        x_s = x4.mean(dim=1)
        s = self.sigmoid(self.spatial_conv(x_s))
        s = s.repeat_interleave(T, dim=0)

        x_t = x4.mean(dim=3).permute(0, 2, 1)
        t = self.sigmoid(self.temporal_conv(x_t))
        t = t.permute(0, 2, 1).unsqueeze(-1)
        t = t.repeat(1, 1, 1, N).view(B_T, 1, N)

        x_c = x4.mean(dim=(1, 3))
        c = self.channel_fc(x_c)
        c = c.unsqueeze(1).unsqueeze(-1).repeat(1, T, 1, N)
        c = c.view(B_T, C, N)

        return x * s * t * c
class MultiSubsetGCNLayer(nn.Module):
    def __init__(self, in_dim, embed_dim, adj_subsets, num_joints=18, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_joints = num_joints

        self.num_subsets = adj_subsets.size(0)
        self.feature_transforms = nn.ModuleList([
            nn.Linear(in_dim, embed_dim, bias=bias) for _ in range(self.num_subsets)
        ])

        self.edge_weights = nn.Parameter(torch.ones(self.num_subsets, num_joints, num_joints))

        self.register_buffer('adj_subsets', adj_subsets)

    def normalize_adj(self, A):
        deg = A.sum(dim=-1, keepdim=True)
        deg_inv = torch.pow(deg + 1e-6, -1.0)
        return A * deg_inv

    def forward(self, x):
        output = 0
        for i in range(self.num_subsets):
            A = self.adj_subsets[i] * self.edge_weights[i]
            A_norm = self.normalize_adj(A)

            weighted_x = torch.matmul(A_norm, x)
            output += self.feature_transforms[i](weighted_x)

        return output

class STMA_GCN_Unit(nn.Module):
    def __init__(self, in_dim, embed_dim, adj_subsets, kernel_size=3, dropout=0.3, T=5):
        super().__init__()
        self.T = T
        self.gcn = MultiSubsetGCNLayer(in_dim, embed_dim, adj_subsets)
        self.attention = AttentionModule(embed_dim, T=T)
        self.tcn = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0)),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.res_conv = nn.Conv2d(in_dim, embed_dim, kernel_size=1) if in_dim != embed_dim else None

    def forward(self, x):
        B_T, N, C_in = x.shape
        B = B_T // self.T

        out = self.gcn(x)
        out = out.permute(0, 2, 1)
        out = self.attention(out)

        out = out.view(B, self.T, -1, N).permute(0, 2, 1, 3)
        out = self.tcn(out)

        res = x.view(B, self.T, N, C_in).permute(0, 3, 1, 2)
        if self.res_conv is not None: res = self.res_conv(res)

        out = F.relu(out + res)
        return out.permute(0, 2, 3, 1).contiguous().view(B_T, N, -1)

class STA_GCN (nn.Module):
    def __init__(self, in_dim=2, embed_dim=128, hidden_dim1=32, hidden_dim2=64, dropout=0.3, T=5):
        super().__init__()
        self.T = T
        adj_subsets = self._build_multimodal_adj()

        self.layer1 = STMA_GCN_Unit(in_dim, hidden_dim1, adj_subsets, dropout=dropout, T=T)
        self.layer2 = STMA_GCN_Unit(hidden_dim1, hidden_dim2, adj_subsets, dropout=dropout, T=T)
        self.layer3 = STMA_GCN_Unit(hidden_dim2, embed_dim, adj_subsets, dropout=dropout, T=T)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def _build_multimodal_adj(self):
        N = 18
        hierarchy = {17: 0, 0: 1, 5: 1, 6: 1, 11: 1, 12: 1, 1: 2, 2: 2, 7: 2, 8: 2, 13: 2, 14: 2, 3: 3, 4: 3, 9: 3,
                     10: 3, 15: 3, 16: 3}
        edges = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 17), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14),
                 (14, 16), (17, 5), (17, 6), (17, 11), (17, 12)]
        A_root, A_in, A_out = torch.eye(N), torch.zeros(N, N), torch.zeros(N, N)
        for i, j in edges:
            if hierarchy[i] < hierarchy[j]:
                A_in[j, i], A_out[i, j] = 1, 1
            elif hierarchy[i] > hierarchy[j]:
                A_in[i, j], A_out[j, i] = 1, 1
            else:
                A_in[i, j] = A_in[j, i] = 1
        return torch.stack([A_root, A_in, A_out])

    def forward(self, x):
        B, T, N, C = x.shape
        raw_coords = x.view(B * T, N, C)

        x_feat = self.layer1(raw_coords)
        x_feat = self.layer2(x_feat)
        x_feat = self.layer3(x_feat)

        x_feat = x_feat.permute(0, 2, 1)
        x_feat = self.global_pool(x_feat).squeeze(-1)

        return x_feat.view(B, T, -1)
