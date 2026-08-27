import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class EndogenousFeatureDecoupling(nn.Module):
    """
    轻量化内生偏移特征解耦模块
    专门用于解耦TransModule生成的内生偏移中的不变特征和变换特征
    
    轻量化设计：
    1. 增加reduction比例减少特征分析器参数
    2. 简化特征提取器结构，移除深度卷积
    3. 共享部分参数减少重复计算
    4. 移除输出投影层，直接使用融合特征
    5. 保持核心解耦功能和残差连接
    """
    
    def __init__(self, channels: int, reduction: int = 32, decoupling_strength: float = 0.5):
        """
        Args:
            channels: 输入通道数
            reduction: 通道压缩比例 (默认32，比原来16更大)
            decoupling_strength: 解耦强度 [0, 1]
        """
        super().__init__()
        
        self.channels = channels
        self.reduction = reduction
        self.decoupling_strength = decoupling_strength
        
        # 轻量化特征分析器 - 区分不变和变换特征
        self.feature_analyzer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 2, 1, bias=False),  # 2个分支的权重
            nn.Softmax(dim=1)
        )
        
        # 共享的基础特征提取器 - 减少重复参数
        self.shared_extractor = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 轻量化不变特征提取器 - 专注于形状和结构
        self.invariant_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 轻量化变换特征提取器 - 专注于颜色和噪声
        self.variant_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 轻量化特征融合器 - 减少通道数
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W] (可能包含内生偏移)
            
        Returns:
            output: 融合后的增强特征 [B, C, H, W]
        """
        # 分析特征特性，确定解耦权重
        feature_weights = self.feature_analyzer(x)  # [B, 2, 1, 1]
        
        # 共享特征提取
        shared_feat = self.shared_extractor(x)
        
        # 提取不变特征 - 形状和结构信息（全局上下文）
        invariant_attn = self.invariant_branch(shared_feat)
        invariant_feat = x * invariant_attn
        
        # 提取变换特征 - 颜色和噪声信息（局部特征）
        variant_attn = self.variant_branch(shared_feat)
        variant_feat = x * variant_attn
        
        # 自适应解耦 - 根据特征特性调整解耦强度
        decoupling_mask = torch.sigmoid(torch.tensor(self.decoupling_strength * 10, device=x.device, dtype=x.dtype))
        
        # 应用解耦权重
        invariant_feat = invariant_feat * feature_weights[:, 0:1, :, :] * decoupling_mask
        variant_feat = variant_feat * feature_weights[:, 1:2, :, :] * decoupling_mask
        
        # 特征融合
        concat_feat = torch.cat([invariant_feat, variant_feat], dim=1)
        fused_feat = self.fusion(concat_feat)
        
        # 残差连接 - 保持原始信息
        output = fused_feat + x
        
        return output