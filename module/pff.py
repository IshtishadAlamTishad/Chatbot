import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):

    def __init__(self, DModel, Hidden, DropProb=0.1):
        super().__init__()
        self.Linear1 = nn.Linear(DModel, Hidden)
        self.Linear2 = nn.Linear(Hidden, DModel)
        self.Relu = nn.ReLU()
        self.Dropout = nn.Dropout(p=DropProb)

    def forward(self, X):
        X = self.Linear1(X)
        X = self.Relu(X)
        X = self.Dropout(X)
        X = self.Linear2(X)
        return X