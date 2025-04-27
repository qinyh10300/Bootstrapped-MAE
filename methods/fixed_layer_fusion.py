import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedLayerFusion(nn.Module):
    def __init__(self, weights):
        super(FixedLayerFusion, self).__init__()
        self.weights = torch.tensor(weights)

    def forward(self, layer_outputs):
        normalized_weights = F.softmax(self.weights, dim=0) 
        fused_features = sum(w * layer for w, layer in zip(normalized_weights, layer_outputs))  # shape = [batch_size, layer_dim]
        return fused_features