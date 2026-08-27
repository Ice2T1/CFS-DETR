import torch
import torch.nn as nn


class OriModule(nn.Module):
    """
    OriModule - 简单的透明层，直接传递输入到输出
    用于在模型中作为占位符或调试用途
    """
    
    def __init__(self):
        """初始化透明层"""
        super().__init__()
    
    def forward(self, x):
        """
        前向传播 - 直接返回输入
        
        Args:
            x: 输入张量
            
        Returns:
            输出张量（与输入相同）
        """
        return x
