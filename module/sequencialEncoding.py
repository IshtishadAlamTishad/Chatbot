import torch.nn as nn
import torch.nn.functional as F

class sequencialEncoding(nn.Sequential):
    
    def forward(self, *inputs):
        x,saMask = inputs
        for module in self._modules.values():
            out = module(x,saMask)

        return out