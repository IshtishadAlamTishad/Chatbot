import torch
import torch.nn.functional as F
import string
from module.cuda import getDevice


def Test(TransformerModel, LanguageToIndex, IndexToLanguage, MaxSequenceLength, TestQueries):
    TransformerModel.eval()
    Results = []
    with torch.no_grad():
        
        for Query in TestQueries:
            QueryTokens = [LanguageToIndex.get(Token, LanguageToIndex['<PAD>']) for Token in Query.lower().translate(str.maketrans('', '', string.punctuation)).split()]
            
            if len(QueryTokens) > MaxSequenceLength - 2:
                QueryTokens = QueryTokens[:MaxSequenceLength - 2]
            
            QueryTokens = [LanguageToIndex['<START>']] + QueryTokens + [LanguageToIndex['<END>']] + [LanguageToIndex['<PAD>']] * (MaxSequenceLength - len(QueryTokens) - 2)
            InputTensor = torch.tensor([QueryTokens]).to(getDevice())
            TargetTensor = torch.tensor([[LanguageToIndex['<START>']] + [LanguageToIndex['<PAD>']] * (MaxSequenceLength - 1)]).to(getDevice())
            
            SeqLen = InputTensor.size(1)
            SelfAttentionMask = torch.triu(torch.ones(SeqLen, SeqLen), diagonal=1).bool().to(getDevice())
            SelfAttentionMask = SelfAttentionMask * -1e9
            
            Output = TransformerModel(InputTensor, TargetTensor[:, :-1],
                                   EncoderSelfAttentionMask=None,
                                   DecoderSelfAttentionMask=SelfAttentionMask[:, :, :TargetTensor.size(1)-1, :TargetTensor.size(1)-1],
                                   DecoderCrossAttentionMask=None,
                                   EncStartToken=False,
                                   EncEndToken=False,
                                   DecStartToken=True,
                                   DecEndToken=False)
            
            OutputProbs = F.softmax(Output, dim=-1)
            PredictedIndices = torch.argmax(OutputProbs, dim=-1)
            PredictedTokens = [IndexToLanguage.get(Idx.item(), '') for Idx in PredictedIndices[0] if Idx.item() in IndexToLanguage and IndexToLanguage[Idx.item()] not in ['<START>', '<END>', '<PAD>']]
            Response = ' '.join(PredictedTokens).strip()
            if not Response:
                Response = "No meaningful response generated."
            Results.append((Query, Response))
    return Results