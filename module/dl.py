import torch.nn as nn
from module.multiheadAttention import MultiHeadAttention
from module.pff import PositionwiseFeedForward
from module.LN import LayerNormalization
from module.multiheadCattention import MultiHeadCrossAttention


class DecoderLayer(nn.Module):
    def __init__(self, DModel, FfnHidden, NumHeads, DropProb):
        super().__init__()
        self.SelfAttention = MultiHeadAttention(DModel=DModel, NumHeads=NumHeads)
        self.LayerNorm1 = LayerNormalization(ParametersShape=[DModel])
        self.Dropout1 = nn.Dropout(p=DropProb)
        self.EncoderDecoderAttention = MultiHeadCrossAttention(DModel=DModel, NumHeads=NumHeads)
        self.LayerNorm2 = LayerNormalization(ParametersShape=[DModel])
        self.Dropout2 = nn.Dropout(p=DropProb)
        self.Ffn = PositionwiseFeedForward(DModel=DModel, Hidden=FfnHidden, DropProb=DropProb)
        self.LayerNorm3 = LayerNormalization(ParametersShape=[DModel])
        self.Dropout3 = nn.Dropout(p=DropProb)

    def forward(self, X, Y, SelfAttentionMask, CrossAttentionMask):
        Y_ = Y.clone()
        Y = self.SelfAttention(Y, Mask=SelfAttentionMask)
        Y = self.Dropout1(Y)
        Y = self.LayerNorm1(Y + Y_)
        Y_ = Y.clone()
        Y = self.EncoderDecoderAttention(X, Y, Mask=CrossAttentionMask)
        Y = self.Dropout2(Y)
        Y = self.LayerNorm2(Y + Y_)
        Y_ = Y.clone()
        Y = self.Ffn(Y)
        Y = self.Dropout3(Y)
        Y = self.LayerNorm3(Y + Y_)
        return Y
