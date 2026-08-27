import torch
import torch.nn as nn
import torch.nn.functional as F

class FAM2D_Original(nn.Module):
    """
    别人修改的FAM2D模块版本
    """
    def __init__(self, in_channels=64, out_channels=None, seq_len=5000):
        super(FAM2D_Original, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len
        
        # 空间特征投影层
        self.fc_spt = nn.Sequential(
            nn.Linear(seq_len, seq_len // 10),
            nn.ReLU(),
            nn.Linear(seq_len // 10, seq_len),
        )
        
        # 查询特征投影层
        self.fc_qry = nn.Sequential(
            nn.Linear(seq_len, seq_len // 10),
            nn.ReLU(),
            nn.Linear(seq_len // 10, seq_len),
        )
        
        # 融合层
        self.fc_fusion = nn.Sequential(
            nn.Linear(seq_len * 2, seq_len // 5),
            nn.ReLU(),
            nn.Linear(seq_len // 5, 2 * seq_len),
        )
        
        self.sigmoid = nn.Sigmoid()

    def correlation_matrix(self, spt_fg_fts, qry_fg_fts):
        qry_fg_fts = F.normalize(qry_fg_fts, p=2, dim=1)
        spt_fg_fts = F.normalize(spt_fg_fts, p=2, dim=1)
        # 计算支持集和查询集特征的逐元素相乘并在维度1上求和，得到余弦相似度矩阵
        cosine_similarity = torch.sum(spt_fg_fts * qry_fg_fts, dim=1, keepdim=True)
        return cosine_similarity

    def forward(self, spt_fg_fts, qry_fg_fts, band='none'):
        # 特征通过全连接层并使用Relu激活函数
        spt_proj = F.relu(self.fc_spt(spt_fg_fts))
        qry_proj = F.relu(self.fc_qry(qry_fg_fts))
        
        similarity_matrix = self.sigmoid(self.correlation_matrix(spt_fg_fts, qry_fg_fts))
        
        if band == 'inhibit':
            weighted_spt = (1 - similarity_matrix) * spt_proj
            weighted_qry = (1 - similarity_matrix) * qry_proj
        else:
            weighted_spt = similarity_matrix * spt_proj
            weighted_qry = similarity_matrix * qry_proj
        
        combined = torch.cat((weighted_spt, weighted_qry), dim=2)
        fused_tensor = F.relu(self.fc_fusion(combined))
        return fused_tensor 