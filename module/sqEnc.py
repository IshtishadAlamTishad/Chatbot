import torch.nn as nn


class SequentialEncoder(nn.Sequential):
    def forward(self, *Inputs):
        X, SelfAttentionMask = Inputs
        for Module in self._modules.values():
            X = Module(X, SelfAttentionMask)
        return X
