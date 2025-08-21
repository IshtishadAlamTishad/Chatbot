import torch.nn as nn
from module.se import SentenceEmbedding
from module.el import EncoderLayer
from module.sqEnc import SequentialEncoder

class Encoder(nn.Module):
    def __init__(self, DModel, FfnHidden, NumHeads, DropProb, NumLayers, MaxSequenceLength, LanguageToIndex, StartToken, EndToken, PaddingToken):
        super().__init__()
        self.SentenceEmbedding = SentenceEmbedding(MaxSequenceLength, DModel, LanguageToIndex, StartToken, EndToken, PaddingToken)
        self.Layers = SequentialEncoder(*[EncoderLayer(DModel, FfnHidden, NumHeads, DropProb) for _ in range(NumLayers)])

    def forward(self, X, SelfAttentionMask, StartToken, EndToken):
        X = self.SentenceEmbedding(X, StartToken, EndToken)
        X = self.Layers(X, SelfAttentionMask)
        return X
