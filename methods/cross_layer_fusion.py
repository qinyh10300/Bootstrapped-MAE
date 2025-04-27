import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossLayerFusion(nn.Module):
    def __init__(self, layer_dim, layer_num):
        super(CrossLayerFusion, self).__init__()
        self.fc = nn.Linear(layer_dim * layer_num, layer_dim)
    
    def forward(self, layer_outputs):
        fused_features = torch.cat(layer_outputs, dim=-1)
        
        # 使用线性层将 seq_len 转换为 layer_dim
        fused_features = fused_features.permute(1, 2, 0)  # 调整维度为 [batch_size, feature_dim, seq_len]
        output = self.fc(fused_features)  # 形状：[batch_size, feature_dim, layer_dim]
        output = output.permute(0, 2, 1)  # 调整维度为 [layer_dim, batch_size, feature_dim]

        return output