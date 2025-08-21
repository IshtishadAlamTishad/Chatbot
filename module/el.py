import torch.nn as nn
from module.multiheadAttention import MultiHeadAttention
from module.pff import PositionwiseFeedForward
from module.LN import LayerNormalization

class EncoderLayer(nn.Module):
    def __init__(self, DModel, FfnHidden, NumHeads, DropProb):
        super().__init__()
        self.Attention = MultiHeadAttention(DModel=DModel, NumHeads=NumHeads)
        self.Norm1 = LayerNormalization(ParametersShape=[DModel])
        self.Dropout1 = nn.Dropout(p=DropProb)
        self.Ffn = PositionwiseFeedForward(DModel=DModel, Hidden=FfnHidden, DropProb=DropProb)
        self.Norm2 = LayerNormalization(ParametersShape=[DModel])
        self.Dropout2 = nn.Dropout(p=DropProb)

    def forward(self, X, SelfAttentionMask):
        ResidualX = X.clone()
        X = self.Attention(X, Mask=SelfAttentionMask)
        X = self.Dropout1(X)
        X = self.Norm1(X + ResidualX)
        ResidualX = X.clone()
        X = self.Ffn(X)
        X = self.Dropout2(X)
        X = self.Norm2(X + ResidualX)
        return X
