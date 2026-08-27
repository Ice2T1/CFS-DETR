import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
import math
import random


class TransModule(nn.Module):
    
    def __init__(self, c, endogenous_prob=0.05, aging_level=0.5):
        """
        Advanced Endogenous Shift Module for training.
        Combines frequency domain and spatial domain endogenous shifts in series.

        Args:
            c (int): Number of input channels (usually 3 for RGB).
            endogenous_prob (float): Probability of applying endogenous shift [0, 1].
            aging_level (float): Aging factor for endogenous shift [0, 1].
        """
        super(TransModule, self).__init__()
        self.c = c
        self.endogenous_prob = endogenous_prob  # 增加概率到1.0
        self.aging_level = aging_level  # 保持0.5
        
        # 内生偏移相关参数
        self.register_buffer('aging_factor', torch.tensor(aging_level))
        
    def _apply_endogenous_frequency_shift(self, x):
        """频域内生偏移 - 模拟传感器频响特性变化"""
        if not self.training or random.random() > self.endogenous_prob:
            return x
            
        B, C, H, W = x.shape
        device = x.device
        aging_level = torch.sigmoid(self.aging_factor).item()
        
        # FFT变换
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        
        # 创建频域滤波器 - 模拟传感器频响特性
        y = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        Y, X = torch.meshgrid(y, x_coords, indexing='ij')
        D = torch.sqrt(X ** 2 + Y ** 2)
        
        # 低通滤波器 - 模拟传感器老化导致的频响下降 (减少对线条的弱化)
        cutoff_freq = 0.7 - aging_level * 0.3  # 提高截止频率，减少对高频的过度衰减
        low_pass = torch.exp(-D ** 2 / (2 * cutoff_freq ** 2))
        
        # 添加频域噪声 - 模拟传感器噪声 (减少噪声)
        freq_noise = torch.randn_like(x_fft_shift) * (0.02 + aging_level * 0.05)
        
        # 应用频域变换
        x_fft_shift = x_fft_shift * low_pass.unsqueeze(0).unsqueeze(0) + freq_noise
        
        # 逆FFT
        x_ishift = torch.fft.ifftshift(x_fft_shift, dim=(-2, -1))
        result = torch.fft.ifft2(x_ishift, dim=(-2, -1)).real
        
        return result.clamp(0, 1)
    
    def _apply_endogenous_wavelet_shift(self, x):
        """小波域内生偏移 - 模拟多尺度特征变化"""
        if not self.training or random.random() > self.endogenous_prob:
            return x
            
        B, C, H, W = x.shape
        device = x.device
        aging_level = torch.sigmoid(self.aging_factor).item()
        
        # 使用简化的频域处理替代复杂的小波变换
        # 使用FFT进行频域分解
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        
        # 创建频域掩码
        y = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        Y, X = torch.meshgrid(y, x_coords, indexing='ij')
        D = torch.sqrt(X ** 2 + Y ** 2)
        
        # 低频掩码 (模拟低频分量)
        low_freq_mask = torch.exp(-D ** 2 / (2 * 0.3 ** 2))
        
        # 高频掩码 (模拟高频分量)
        high_freq_mask = 1.0 - low_freq_mask
        
        # 分离低频和高频分量
        x_low = x_fft_shift * low_freq_mask.unsqueeze(0).unsqueeze(0)
        x_high = x_fft_shift * high_freq_mask.unsqueeze(0).unsqueeze(0)
        
        # 在小波域应用内生偏移 (减少对线条的弱化)
        # 低频分量 - 模拟整体质量下降
        x_low = x_low * (1.0 - aging_level * 0.3)  # 减少低频衰减
        
        # 高频分量 - 模拟细节丢失 (保留更多线条)
        x_high = x_high * (1.0 - aging_level * 0.4)  # 减少高频衰减，保留线条
        
        # 添加频域噪声 (减少噪声)
        noise_std = 0.02 + aging_level * 0.05
        x_low += torch.randn_like(x_low) * noise_std
        x_high += torch.randn_like(x_high) * noise_std
        
        # 重新组合
        x_combined = x_low + x_high
        
        # 逆FFT
        x_ishift = torch.fft.ifftshift(x_combined, dim=(-2, -1))
        result = torch.fft.ifft2(x_ishift, dim=(-2, -1)).real
        
        return result.clamp(0, 1)
    
    def _apply_endogenous_spatial_shift(self, x):
        """空间域内生偏移 - 模拟传感器响应变化"""
        if not self.training or random.random() > self.endogenous_prob:
            return x
            
        B, C, H, W = x.shape
        device = x.device
        aging_level = torch.sigmoid(self.aging_factor).item()
        
        result = x.clone()
        
        # 1. 传感器响应非线性变化 (减少偏红效果)
        gamma_shift = 1.0 + aging_level * 0.5  # 进一步减少gamma变化强度
        result = torch.pow(result + 1e-8, gamma_shift)
        
        # 2. 通道间串扰 - 模拟传感器串扰 (恢复串扰强度)
        cross_talk = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
        cross_talk += torch.randn(B, 3, 3, device=device) * aging_level * 0.3  # 恢复串扰强度
        
        # 应用串扰
        x_reshaped = result.view(B, C, -1)  # (B, C, H*W)
        x_transformed = torch.bmm(cross_talk, x_reshaped)  # (B, C, H*W)
        result = x_transformed.view(B, C, H, W)
        
        # 3. 空间不均匀性 - 模拟传感器不均匀响应 (减少不均匀性)
        spatial_mask = torch.randn(B, C, H, W, device=device) * aging_level * 0.2 + 1.0  # 减少不均匀性
        result = result * spatial_mask
        
        # 4. 暗电流偏移 (减少暗电流)
        dark_current = torch.rand(B, C, 1, 1, device=device) * aging_level * 0.05  # 减少暗电流
        result = result + dark_current
        
        return result.clamp(0, 1)

    def forward(self, x):
        """
        应用高级内生偏移转换 - 串联处理，保持增强参数
        
        Args:
            x: 输入图片 tensor, shape (B, C, H, W), 值域 [0, 1]
            
        Returns:
            转换后的图片 tensor, 相同shape
        """
        # 串联处理：先频域，再空间域
        result = x
        
        # 1. 频域内生偏移 (频域处理)
        result = self._apply_endogenous_frequency_shift(result)
        result = self._apply_endogenous_wavelet_shift(result)
        
        # 2. 空间域内生偏移 (空间域处理)
        result = self._apply_endogenous_spatial_shift(result)
        
        return result.clamp(0, 1)
    
    def update_aging_factor(self, epoch, total_epochs):
        """更新老化因子 - 模拟随时间的老化过程"""
        self.aging_factor.data = torch.tensor(epoch / total_epochs)
    
    def set_aging_level(self, aging_level):
        """设置老化程度"""
        self.aging_factor.data = torch.tensor(aging_level)