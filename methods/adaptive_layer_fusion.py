import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveLayerFusion(nn.Module):
    def __init__(self, layer_num):
        super(AdaptiveLayerFusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(layer_num))  # 可优化的权重参数，初始化为全1

    def forward(self, layer_outputs):
        normalized_weights = F.softmax(self.weights, dim=0) 
        fused_features = sum(w * layer for w, layer in zip(normalized_weights, layer_outputs))  # shape = [batch_size, layer_dim]
        return fused_features