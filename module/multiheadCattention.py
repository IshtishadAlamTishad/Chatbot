import torch.nn as nn  
from module.scaledDotProduct import scaledDotProduct


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, DModel, NumHeads):
        super().__init__()
        self.DModel = DModel
        self.NumHeads = NumHeads
        self.HeadDim = DModel // NumHeads
        self.KvLayer = nn.Linear(DModel, 2 * DModel)
        self.QLayer = nn.Linear(DModel, DModel)
        self.LinearLayer = nn.Linear(DModel, DModel)
    
    def forward(self, X, Y, Mask=None):
        BatchSize, SequenceLength, DModel = X.size()
        Kv = self.KvLayer(X)
        Q = self.QLayer(Y)
        Kv = Kv.reshape(BatchSize, SequenceLength, self.NumHeads, 2 * self.HeadDim)
        Q = Q.reshape(BatchSize, SequenceLength, self.NumHeads, self.HeadDim)
        Kv = Kv.permute(0, 2, 1, 3)
        Q = Q.permute(0, 2, 1, 3)
        K, V = Kv.chunk(2, dim=-1)
        Values, Attention = scaledDotProduct(Q, K, V, Mask)
        Values = Values.permute(0, 2, 1, 3).reshape(BatchSize, SequenceLength, DModel)
        Out = self.LinearLayer(Values)
        return Out
