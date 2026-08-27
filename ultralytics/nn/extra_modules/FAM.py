import torch
import torch.nn as nn
import torch.nn.functional as F

class FAM2D(nn.Module):
    """
    2D版本的FAM模块，适用于CNN架构
    通过计算两组特征的相似性，增强相似特征（不变特征），减弱非相似特征（变化特征）
    """
    def __init__(self, inc, ouc):
        super(FAM2D, self).__init__()
        
        # inc是输入通道数列表，ouc是输出通道数
        self.in_channels = inc[0] if isinstance(inc, (list, tuple)) else inc
        self.out_channels = ouc
        
        # 特征投影层
        self.conv_spt = nn.Conv2d(self.in_channels, self.in_channels, 1, bias=False)
        self.conv_qry = nn.Conv2d(self.in_channels, self.in_channels, 1, bias=False)
        
        # 融合层
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(self.in_channels * 2, self.in_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def correlation_matrix(self, spt_fg_fts, qry_fg_fts):
        """
        计算2D特征图之间的相关性矩阵
        """
        # 归一化特征
        spt_fg_fts_norm = F.normalize(spt_fg_fts, p=2, dim=1)
        qry_fg_fts_norm = F.normalize(qry_fg_fts, p=2, dim=1)
        
        # 计算余弦相似度
        cosine_similarity = torch.sum(spt_fg_fts_norm * qry_fg_fts_norm, dim=1, keepdim=True)
        
        return cosine_similarity

    def forward(self, x, band='normal'):
        """
        通过相似性计算增强不变特征，减弱变化特征
        
        Args:
            x (list): 包含两个特征张量的列表 [spt_fg_fts, qry_fg_fts]
            band (str): 频带类型，'inhibit' 或其他

        Returns:
            torch.Tensor: 融合后的特征 [B, out_channels, H, W]
        """
        # 处理输入：x可能是列表或元组
        if isinstance(x, (list, tuple)) and len(x) == 2:
            spt_fg_fts, qry_fg_fts = x
        else:
            raise ValueError(f"FAM2D expects 2 input tensors, got {len(x) if isinstance(x, (list, tuple)) else 1}")
        
        # 特征投影
        spt_proj = self.relu(self.conv_spt(spt_fg_fts))
        qry_proj = self.relu(self.conv_qry(qry_fg_fts))
        
        # 计算相似度矩阵
        similarity_matrix = self.sigmoid(self.correlation_matrix(spt_fg_fts, qry_fg_fts))
        
        # 根据频带类型调整权重
        if band == 'inhibit':
            weighted_spt = (1 - similarity_matrix) * spt_proj
            weighted_qry = (1 - similarity_matrix) * qry_proj
        else:
            weighted_spt = similarity_matrix * spt_proj
            weighted_qry = similarity_matrix * qry_proj
        
        # 拼接融合
        combined = torch.cat((weighted_spt, weighted_qry), dim=1)
        fused_tensor = self.conv_fusion(combined)
        
        return fused_tensor