import torch
import torch.nn as nn
from module.postionalEncoding import PE
from module.cuda import getDevice

class SentenceEmbedding(nn.Module):
    def __init__(self, MaxSequenceLength, DModel, LanguageToIndex, StartToken, EndToken, PaddingToken):
        super().__init__()
        self.VocabSize = len(LanguageToIndex)
        self.MaxSequenceLength = MaxSequenceLength
        self.Embedding = nn.Embedding(self.VocabSize, DModel)
        self.LanguageToIndex = LanguageToIndex
        self.PositionEncoder = PE(DModel, MaxSequenceLength)
        self.Dropout = nn.Dropout(p=0.1)
        self.StartToken = StartToken
        self.EndToken = EndToken
        self.PaddingToken = PaddingToken
    
    def BatchTokenize(self, Batch, StartToken, EndToken):
        def Tokenize(Sentence, StartToken, EndToken):
            SentenceWordIndices = [self.LanguageToIndex.get(Token, self.LanguageToIndex[self.PaddingToken]) for Token in list(Sentence)]
            if StartToken:
                SentenceWordIndices.insert(0, self.LanguageToIndex[self.StartToken])
            if EndToken:
                SentenceWordIndices.append(self.LanguageToIndex[self.EndToken])
            for _ in range(len(SentenceWordIndices), self.MaxSequenceLength):
                SentenceWordIndices.append(self.LanguageToIndex[self.PaddingToken])
            return torch.tensor(SentenceWordIndices)

        Tokenized = []
        for SentenceNum in range(len(Batch)):
            Tokenized.append(Tokenize(Batch[SentenceNum], StartToken, EndToken))
        Tokenized = torch.stack(Tokenized)
        return Tokenized.to(getDevice())
    
    def forward(self, X, StartToken, EndToken):
        X = self.BatchTokenize(X, StartToken, EndToken)
        X = self.Embedding(X)
        Pos = self.PositionEncoder()
        X = self.Dropout(X + Pos)
        return X
