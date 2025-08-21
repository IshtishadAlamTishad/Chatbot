import torch.nn as nn

class SequentialDecoder(nn.Sequential):
    def forward(self, *Inputs):
        X, Y, SelfAttentionMask, CrossAttentionMask = Inputs
        for Module in self._modules.values():
            Y = Module(X, Y, SelfAttentionMask, CrossAttentionMask)
        return Y