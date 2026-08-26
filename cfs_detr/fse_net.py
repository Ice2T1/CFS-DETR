"""FSE-Net components used by CFS-DETR."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import C2f, Conv


class UniversalEdgeEnhancer(nn.Module):
    """Universal edge enhancer (UEE)."""

    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = x - self.pool(x)
        edge = self.out_conv(edge)
        return x + edge


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1),
                          torch.mean(x, 1).unsqueeze(1)), dim=1)


class ESMSpatialGate(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 3, act=False)
        self.dw1 = nn.Sequential(
            Conv(channel, channel, 5, s=1, d=2, g=channel, act=nn.GELU()),
            Conv(channel, channel, 7, s=1, d=3, g=channel, act=nn.GELU()),
        )
        self.dw2 = Conv(channel, channel, 3, g=channel, act=nn.GELU())

    def forward(self, x):
        spatial_weight = self.spatial(self.compress(x))
        return self.dw1(x) * spatial_weight + self.dw2(x)


class ESMLocalAttention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(channel, 1, 1))
        self.b = nn.Parameter(torch.ones(channel, 1, 1))

    def forward(self, x):
        centered = x - torch.mean(x, dim=(2, 3), keepdim=True)
        return self.a * centered * x + self.b * x


class EdgeSelectionModule(nn.Module):
    """Edge selection module (ESM)."""

    def __init__(self, channel):
        super().__init__()
        self.spatial_gate = ESMSpatialGate(channel)
        self.local_attention = ESMLocalAttention(channel)
        self.a = nn.Parameter(torch.zeros(channel, 1, 1))
        self.b = nn.Parameter(torch.ones(channel, 1, 1))

    def forward(self, x):
        out = self.local_attention(self.spatial_gate(x))
        return self.a * out + self.b * x


class FiberSEMEdgeEnhancer(nn.Module):
    """Fiber SEM edge enhancer (FSEE)."""

    def __init__(self, inc, bins=(3, 6, 9, 12)):
        super().__init__()
        branch_channels = inc // len(bins)
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                Conv(inc, branch_channels, 1),
                Conv(branch_channels, branch_channels, 3, g=branch_channels),
            ) for bin_size in bins
        ])
        self.edge_enhancers = nn.ModuleList([
            UniversalEdgeEnhancer(branch_channels) for _ in bins
        ])
        self.local_conv = Conv(inc, inc, 3)
        self.esm = EdgeSelectionModule(inc * 2)
        self.final_conv = Conv(inc * 2, inc)

    def forward(self, x):
        size = x.shape[2:]
        outputs = [self.local_conv(x)]
        for feature, enhancer in zip(self.features, self.edge_enhancers):
            branch = F.interpolate(feature(x), size, mode='bilinear', align_corners=True)
            outputs.append(enhancer(branch))
        return self.final_conv(self.esm(torch.cat(outputs, dim=1)))


class FSENetBlock(C2f):
    """CSP wrapper used to construct the FSE-Net backbone."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            FiberSEMEdgeEnhancer(self.c) for _ in range(n)
        )
