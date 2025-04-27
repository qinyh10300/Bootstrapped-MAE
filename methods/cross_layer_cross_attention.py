import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossLayerCrossAttention(nn.Module):
    def __init__(self, layer_dim, num_heads, max_sequences=4):
        super(CrossLayerCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.layer_dim = layer_dim

        # 初始化多头注意力层，最多支持 max_sequences 个序列
        self.attn_layers = nn.ModuleList([nn.MultiheadAttention(self.layer_dim, self.num_heads) for _ in range(max_sequences)])

        self.fc = nn.Linear(self.layer_dim, self.layer_dim)

    def forward(self, layer_outputs):
        """
        :param layer_outputs: 一个包含多个序列的列表，每个序列的形状为 [batch_size, seq_len, embed_dim]
        :return: 融合后的序列，形状为 [batch_size, seq_len, embed_dim]
        """
        num_sequences = len(layer_outputs)
        assert num_sequences >= 2, "至少需要两个序列进行交叉注意力操作"

        # 将输入序列调整为 [seq_len, batch_size, embed_dim] 格式
        layer_outputs = [seq.permute(1, 0, 2) for seq in layer_outputs]  # 每个序列都变为 [seq_len, batch_size, embed_dim]

        attn_outputs = []
        for i in range(num_sequences):
            # 从其他序列中获取要计算的序列
            other_sequences = [seq for j, seq in enumerate(layer_outputs) if j != i]
            # 将其他序列拼接在一起
            context = torch.cat(other_sequences, dim=0)  # 形状：[sum(seq_len_other), batch_size, embed_dim]

            # 执行交叉注意力
            attn_output, _ = self.attn_layers[i](layer_outputs[i], context, context)
            attn_outputs.append(attn_output)

        # 加法融合所有交叉注意力的输出
        fused_output = sum(attn_outputs)

        # 最后通过一个全连接层映射到最终的维度
        fused_output = self.fc(fused_output)

        # 调整输出维度为 [batch_size, seq_len, embed_dim]
        return fused_output.permute(1, 0, 2)  # 返回 [batch_size, seq_len, embed_dim]