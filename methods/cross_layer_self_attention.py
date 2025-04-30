import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossLayerSelfAttention(nn.Module):
    def __init__(self, layer_dim, layer_num, embed_dim, num_heads=3):
        super(CrossLayerSelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.fc_seq_to_layer = nn.Linear(layer_dim * layer_num, layer_dim)  

    def forward(self, layer_outputs):
        # 假设 layer_outputs 是一个包含多个层输出的列表
        # 转换为 [seq_len, batch_size, feature_dim] 的格式
        layer_outputs = torch.cat(layer_outputs, dim=1)  # 形状：[seq_len, batch_size, feature_dim]
        layer_outputs = layer_outputs.permute(1, 0, 2)  # 形状：[seq_len, batch_size, feature_dim]

        # 跨层自注意力
        # print(layer_outputs.shape)
        attn_output, _ = self.attn(layer_outputs, layer_outputs, layer_outputs)  # 形状：[seq_len, batch_size, feature_dim]

        # 使用线性层将 seq_len 转换为 layer_dim
        attn_output = attn_output.permute(1, 2, 0)  # 调整维度为 [batch_size, feature_dim, seq_len]
        output = self.fc_seq_to_layer(attn_output)  # 形状：[batch_size, feature_dim, layer_dim]
        output = output.permute(0, 2, 1)  # 调整维度为 [batch_size, layer_dim, feature_dim]

        return output