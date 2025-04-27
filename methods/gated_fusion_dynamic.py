import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFusionDynamic(nn.Module):
    def __init__(self, layer_dim, layer_num):
        super(GatedFusionDynamic, self).__init__()
        
        self.layer_num = layer_num
        self.fc_gate = nn.ModuleList([nn.Linear(layer_dim, 1) for _ in range(layer_num)])  # 门控生成每个序列的权重
        self.fc_output = nn.Linear(layer_dim, layer_dim)

    def forward(self, layer_outputs):
        assert len(layer_outputs) == self.layer_num, "The number of input layer_outputs must match layer_num."
        
        gates = []
        for i in range(self.layer_num):
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

        return self.fc_output(fused_seq)