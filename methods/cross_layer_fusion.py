import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossLayerFusion(nn.Module):
    def __init__(self, seq_len, layer_num):
        super(CrossLayerFusion, self).__init__()
        self.fc = nn.Linear(seq_len * layer_num, seq_len)
        # print(seq_len * layer_num, seq_len)
        # exit(0)
    
    def forward(self, layer_outputs):
        fused_features = torch.cat(layer_outputs, dim=1)
        # print(fused_features.shape)
        
        fused_features = fused_features.permute(0, 2, 1)  # 调整维度为 [batch_size, embed_dim, seq_len * 3]
        # print(fused_features.shape)
        output = self.fc(fused_features)  # 形状：[batch_size, embed_dim, seq_len]
        output = output.permute(0, 2, 1)  # 调整维度为 [batch_size, seq_len, embed_dim]
        # print(output)

        return output