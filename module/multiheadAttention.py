import torch.nn as nn
from module.scaledDotProduct import scaledDotProduct


class MultiHeadAttention(nn.Module):
    def __init__(self, DModel, NumHeads):
        super().__init__()
        self.DModel = DModel
        self.NumHeads = NumHeads
        self.HeadDim = DModel // NumHeads
        self.QkvLayer = nn.Linear(DModel, 3 * DModel)
        self.LinearLayer = nn.Linear(DModel, DModel)
    
    def forward(self, X, Mask=None):
        BatchSize, SequenceLength, DModel = X.size()
        Qkv = self.QkvLayer(X)
        Qkv = Qkv.reshape(BatchSize, SequenceLength, self.NumHeads, 3 * self.HeadDim)
        Qkv = Qkv.permute(0, 2, 1, 3)
        Q, K, V = Qkv.chunk(3, dim=-1)
        Values, Attention = scaledDotProduct(Q, K, V, Mask)
        Values = Values.permute(0, 2, 1, 3).reshape(BatchSize, SequenceLength, self.NumHeads * self.HeadDim)
        Out = self.LinearLayer(Values)
        return Out