import torch
import torch.nn as nn
import torch.optim as optim
from module.cuda import getDevice
import numpy as np

def trainModel(model, Dataset, LanguageToIndex, Epochs=10, BatchSize=2, LearningRate=0.0001):
    model.train()
    Optimizer = optim.Adam(model.parameters(), lr=LearningRate)
    Criterion = nn.CrossEntropyLoss(ignore_index=LanguageToIndex['<PAD>'])
    
    for Epoch in range(Epochs):
        TotalLoss = 0
        np.random.shuffle(Dataset)
        
        for i in range(0, len(Dataset),BatchSize):

            Batch = Dataset[i:i + BatchSize]
            Inputs = torch.stack([X[0] for X in Batch]).to(getDevice())  # Shape: [BatchSize, SeqLen]
            Targets = torch.stack([X[1] for X in Batch]).to(getDevice())  # Shape: [BatchSize, SeqLen]
            
            SeqLen = Inputs.size(1)  
            SelfAttentionMask = torch.triu(torch.ones(SeqLen, SeqLen), diagonal=1).bool().to(getDevice())
            SelfAttentionMask = SelfAttentionMask * -1e9
            
            Optimizer.zero_grad()
            Output = model(Inputs, Targets[:, :-1],
                                     EncoderSelfAttentionMask=None,
                                     DecoderSelfAttentionMask=SelfAttentionMask[:, :Targets.size(1)-1],
                                     DecoderCrossAttentionMask=None,
                                     EncStartToken=False,
                                     EncEndToken=False,
                                     DecStartToken=True,
                                     DecEndToken=False)  #Shape: [BatchSize,SeqLen-1,VocabSize]
            
            Loss = Criterion(Output.view(-1, model.Linear.out_features), 
                           Targets[:, 1:].contiguous().view(-1))
            Loss.backward()
            Optimizer.step()
            TotalLoss += Loss.item()
        
        print(f"Epoch {Epoch+1}/{Epochs}, Loss: {TotalLoss / (len(Dataset) // BatchSize)}")
    
    torch.save(model.state_dict(), 'model/trainedModel.pth')
    print("Model saved! -> model/trainedModel.pth")