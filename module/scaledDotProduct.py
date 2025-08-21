import torch
import torch.nn.functional as F
import math

def scaledDotProduct(q,k,v,mask=None):

    dk = q.size()[-1]

    scaled = torch.matmul(q,k.transpose(-1,-2))/math.sqrt(dk)
    if mask is not None:
        scaled += mask
    
    a = F.softmax(scaled,dim=-1)
    v = torch.matmul(a,v)
    
    return v,a