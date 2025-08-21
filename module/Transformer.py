import torch.nn as nn
from module.encoder import Encoder
from module.decoder import Decoder
from module.cuda import getDevice


class Transformer(nn.Module):
    def __init__(self, DModel, FfnHidden, NumHeads, DropProb, NumLayers, MaxSequenceLength, VocabSize, LanguageToIndex, StartToken, EndToken, PaddingToken):
        super().__init__()
        self.Encoder = Encoder(DModel, FfnHidden, NumHeads, DropProb, NumLayers, MaxSequenceLength, LanguageToIndex, StartToken, EndToken, PaddingToken)
        self.Decoder = Decoder(DModel, FfnHidden, NumHeads, DropProb, NumLayers, MaxSequenceLength, LanguageToIndex, StartToken, EndToken, PaddingToken)
        self.Linear = nn.Linear(DModel, VocabSize)
        self.Device = getDevice

    def forward(self, X, Y, EncoderSelfAttentionMask=None, DecoderSelfAttentionMask=None, DecoderCrossAttentionMask=None, EncStartToken=False, EncEndToken=False, DecStartToken=True, DecEndToken=False):
        X = self.Encoder(X, EncoderSelfAttentionMask, StartToken=EncStartToken, EndToken=EncEndToken)
        Out = self.Decoder(X, Y, DecoderSelfAttentionMask, DecoderCrossAttentionMask, StartToken=DecStartToken, EndToken=DecEndToken)
        Out = self.Linear(Out)
        return Out