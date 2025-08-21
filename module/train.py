import torch
import torch.nn as nn
import torch.optim as optim
from module.cuda import getDevice
import numpy as np

def TrainTransformer(TransformerModel, Dataset, LanguageToIndex, Epochs=10, BatchSize=2, LearningRate=0.0001):
    TransformerModel.train()
    Optimizer = optim.Adam(TransformerModel.parameters(), lr=LearningRate)
    Criterion = nn.CrossEntropyLoss(ignore_index=LanguageToIndex['<PAD>'])
    
    for Epoch in range(Epochs):
        TotalLoss = 0
        np.random.shuffle(Dataset)
        for I in range(0, len(Dataset), BatchSize):
            Batch = Dataset[I:I + BatchSize]
            Inputs = torch.stack([X[0] for X in Batch]).to(getDevice)
            Targets = torch.stack([X[1] for X in Batch]).to(getDevice)
            
            SeqLen = Inputs.size(1)
            SelfAttentionMask = torch.triu(torch.ones(SeqLen, SeqLen), diagonal=1).bool().to(getDevice)
            SelfAttentionMask = SelfAttentionMask * -1e9
            
            Optimizer.zero_grad()
            Output = TransformerModel(Inputs, Targets[:, :-1], 
                                   EncoderSelfAttentionMask=None,
                                   DecoderSelfAttentionMask=SelfAttentionMask[:, :, :Targets.size(1)-1, :Targets.size(1)-1],
                                   DecoderCrossAttentionMask=None,
                                   EncStartToken=False,
                                   EncEndToken=False,
                                   DecStartToken=True,
                                   DecEndToken=False)
            
            Loss = Criterion(Output.view(-1, TransformerModel.Linear.out_features), Targets[:, 1:].contiguous().view(-1))
            Loss.backward()
            Optimizer.step()
            TotalLoss += Loss.item()
        
        print(f"Epoch {Epoch+1}/{Epochs}, Loss: {TotalLoss / (len(Dataset) // BatchSize)}")
    
    torch.save(TransformerModel.state_dict(), 'model/TransformerModel.pth')
    print("Model saved! trainedModel.pth")


    