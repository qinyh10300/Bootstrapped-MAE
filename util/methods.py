import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedLayerFusion(nn.Module):
    def __init__(self, weights):
        super(FixedLayerFusion, self).__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, layer_outputs):
        normalized_weights = F.softmax(self.weights, dim=0) 
        fused_features = sum(w * layer for w, layer in zip(normalized_weights, layer_outputs))  # shape = [batch_size, layer_dim]
        return fused_features

class AdaptiveLayerFusion(nn.Module):
    def __init__(self, layer_num):
        super(AdaptiveLayerFusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(layer_num))  # 可优化的权重参数，初始化为全1

    def forward(self, layer_outputs):
        normalized_weights = F.softmax(self.weights, dim=0) 
        fused_features = sum(w * layer for w, layer in zip(normalized_weights, layer_outputs))  # shape = [batch_size, layer_dim]
        return fused_features
    
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

class GatedFusionDynamic(nn.Module):
    def __init__(self, seq_len, layer_num):
        super(GatedFusionDynamic, self).__init__()
        
        assert layer_num > 1, "layer_num must be greater than 1 in GatedFusionDynamic"
        self.layer_num = layer_num
        self.fc_gate = nn.ModuleList([nn.Linear(seq_len, 1) for _ in range(layer_num)])  # 门控生成每个序列的权重
        self.fc_output = nn.Linear(seq_len, seq_len)

    def forward(self, layer_outputs):
        assert len(layer_outputs) == self.layer_num, "The number of input layer_outputs must match layer_num."
        
        gates = []
        for i in range(self.layer_num):
            layer_outputs[i] = layer_outputs[i].permute(0, 2, 1)
            gate = torch.sigmoid(self.fc_gate[i](layer_outputs[i]))  # 计算每个序列的门控
            gates.append(gate)
        
        # 初始化融合后的序列
        fused_seq = torch.zeros_like(layer_outputs[0])  # 假设所有输入序列维度相同
        
        # 融合每对序列的门控信息
        for i in range(self.layer_num):
            for j in range(i + 1, self.layer_num):
                # 使用门控来融合 seq_i 和 seq_j
                gate_ij = (gates[i] + gates[j]) / 2  # 对每对序列计算门控平均值
                fused_seq += gate_ij * (layer_outputs[i] + layer_outputs[j])  # 融合

        self.fc_output(fused_seq)
        fused_seq = fused_seq.permute(0, 2, 1)

        return fused_seq
    
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