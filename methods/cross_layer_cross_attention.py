import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossLayerCrossAttention(nn.Module):
    def __init__(self, layer_dim, layer_num, embed_dim, num_heads=3):
        assert layer_num > 1, "layer_num must be greater than 1 in CrossLayerCrossAttention"
        self.layer_num = layer_num
        super(CrossLayerCrossAttention, self).__init__()
        self.attn_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim, num_heads) for _ in range(layer_num)])
        self.fc = nn.Linear(layer_dim, layer_dim)

    def forward(self, layer_outputs):
        layer_outputs = [seq.permute(1, 0, 2) for seq in layer_outputs]  # 每个序列都变为 [seq_len, batch_size, embed_dim]

        attn_outputs = []
        for i in range(self.layer_num):
            # 从其他序列中获取要计算的序列
            other_sequences = [seq for j, seq in enumerate(layer_outputs) if j != i]
            # 将其他序列拼接在一起
            context = torch.cat(other_sequences, dim=0)  # 形状：[sum(seq_len_other), batch_size, embed_dim]

            # 执行交叉注意力
            attn_output, _ = self.attn_layers[i](layer_outputs[i], context, context)
            attn_outputs.append(attn_output)

        # 加法融合所有交叉注意力的输出
        fused_output = sum(attn_outputs)
        # print(fused_output.shape)

        # 最后通过一个全连接层映射到最终的维度
        fused_output = fused_output.permute(1, 2, 0)   # 调整输出维度为 [batch_size, embed_dim, seq_len]
        fused_output = self.fc(fused_output)

        return fused_output.permute(0, 2, 1)  # 返回 [batch_size, seq_len, embed_dim]