"""Cross-Scale Invariant Module (CIM)."""

import math

import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = float(coefficient)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coefficient * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.0

    @staticmethod
    def coefficient(epoch):
        if isinstance(epoch, torch.Tensor):
            epoch = epoch.detach().item()
        epoch = max(float(epoch), 0.0)
        return 2.0 / (1.0 + math.exp(-epoch)) - 1.0

    def forward(self, x, epoch=0.0):
        self.alpha = self.coefficient(epoch)
        return GradientReversalFunction.apply(x, self.alpha)


class CrossScaleInvariantModule(nn.Module):
    """CIM containing a gradient reversal layer and domain classifier."""

    def __init__(self, in_channels=512, hidden_dim=1024, num_domains=2):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, x, epoch=0.0):
        x = self.grl(x, epoch)
        x = self.features(x).flatten(1)
        return self.classifier(x)
