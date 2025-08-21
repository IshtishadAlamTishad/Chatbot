import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, ParametersShape, Eps=1e-5):
        super().__init__()
        self.ParametersShape = ParametersShape
        self.Eps = Eps
        self.Gamma = nn.Parameter(torch.ones(ParametersShape))
        self.Beta = nn.Parameter(torch.zeros(ParametersShape))

    def forward(self, Inputs):
        Dims = [-(i + 1) for i in range(len(self.ParametersShape))]
        Mean = Inputs.mean(dim=Dims, keepdim=True)
        Var = ((Inputs - Mean) ** 2).mean(dim=Dims, keepdim=True)
        Std = (Var + self.Eps).sqrt()
        Y = (Inputs - Mean) / Std
        Out = self.Gamma * Y + self.Beta
        return Out
