import torch
import torch.nn as nn
import torch.nn.functional as F

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