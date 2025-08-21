import torch.nn as nn
from module.se import SentenceEmbedding
from module.dl import DecoderLayer
from module.seqDec import SequentialDecoder

class Decoder(nn.Module):
    
    def __init__(self, DModel, FfnHidden, NumHeads, DropProb, NumLayers, MaxSequenceLength, LanguageToIndex, StartToken, EndToken, PaddingToken):
        
        super().__init__()
        self.SentenceEmbedding = SentenceEmbedding(MaxSequenceLength, DModel, LanguageToIndex, StartToken, EndToken, PaddingToken)
        self.Layers = SequentialDecoder(*[DecoderLayer(DModel, FfnHidden, NumHeads, DropProb) for _ in range(NumLayers)])

    def forward(self, X, Y, SelfAttentionMask, CrossAttentionMask, StartToken, EndToken):
        
        Y = self.SentenceEmbedding(Y, StartToken, EndToken)
        Y = self.Layers(X, Y, SelfAttentionMask, CrossAttentionMask)
        
        return Y