import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFeatureDecoupling(nn.Module):
    """
    简单有效的特征解耦模块
    专注于分离形状特征和颜色/噪声特征
    """
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        
        self.channels = channels
        self.reduction = reduction
        
        # 形状特征提取器 - 使用边缘检测
        self.shape_extractor = nn.Sequential(
            # 边缘检测卷积
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels//4, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # 全局平均池化捕获形状信息
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 颜色/噪声特征提取器
        self.color_extractor = nn.Sequential(
            # 1x1卷积捕获颜色信息
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # 空间注意力捕获噪声
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels//4, bias=False),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion = nn.Conv2d(channels * 2, channels, 1, bias=False)
        
    def forward(self, x):
        # 提取形状特征
        shape_attn = self.shape_extractor(x)
        shape_feat = x * shape_attn
        
        # 提取颜色/噪声特征
        color_attn = self.color_extractor(x)
        color_feat = x * color_attn
        
        # 融合特征
        concat_feat = torch.cat([shape_feat, color_feat], dim=1)
        fused_feat = self.fusion(concat_feat)
        
        # 残差连接
        output = fused_feat + x
        
        return output

class LightweightDecoupling(nn.Module):
    """
    超轻量级特征解耦模块
    最小化计算开销
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.channels = channels
        
        # 简单的形状检测
        self.shape_detect = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 简单的颜色检测
        self.color_detect = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 融合
        self.fusion = nn.Conv2d(channels * 2, channels, 1, bias=False)
        
    def forward(self, x):
        # 形状特征
        shape_attn = self.shape_detect(x)
        shape_feat = x * shape_attn
        
        # 颜色特征
        color_attn = self.color_detect(x)
        color_feat = x * color_attn
        
        # 融合
        concat_feat = torch.cat([shape_feat, color_feat], dim=1)
        fused_feat = self.fusion(concat_feat)
        
        # 残差
        output = fused_feat + x
        
        return output


# 工厂函数
def create_simple_decoupling(module_type: str = "simple", channels: int = 256, **kwargs):
    """创建简单特征解耦模块"""
    if module_type == "simple":
        return SimpleFeatureDecoupling(channels, **kwargs)
    elif module_type == "adaptive":
        return AdaptiveFeatureDecoupling(channels, **kwargs)
    elif module_type == "lightweight":
        return LightweightDecoupling(channels, **kwargs)
    else:
        raise ValueError(f"Unknown module type: {module_type}") 