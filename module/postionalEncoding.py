import torch
import torch.nn as nn


def getDevice():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PE(nn.Module):
    def __init__(self, DModel, MaxSequenceLength):
        super().__init__()
        self.MaxSequenceLength = MaxSequenceLength
        self.DModel = DModel

    def forward(self):
        EvenI = torch.arange(0, self.DModel, 2).float()
        Denominator = torch.pow(10000, EvenI/self.DModel)
        Position = torch.arange(self.MaxSequenceLength).reshape(self.MaxSequenceLength, 1)
        EvenPE = torch.sin(Position / Denominator)
        OddPE = torch.cos(Position / Denominator)
        Stacked = torch.stack([EvenPE, OddPE], dim=2)
        PE = torch.flatten(Stacked, start_dim=1, end_dim=2)
        return PE.to(getDevice())