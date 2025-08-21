import torch
import torch.nn.functional as F
from module.cuda import getDevice

class Chatbot:
    def __init__(self, Db, TransformerModel, LanguageToIndex, IndexToLanguage, MaxSequenceLength, StartToken, EndToken, PaddingToken):
        self.Db = Db
        self.TransformerModel = TransformerModel
        self.LanguageToIndex = LanguageToIndex
        self.IndexToLanguage = IndexToLanguage
        self.MaxSequenceLength = MaxSequenceLength
        self.Memory = []
        self.StartToken = StartToken
        self.EndToken = EndToken
        self.PaddingToken = PaddingToken

    def GenerateResponse(self, RetrievedText, Query):
        InputText = [f"Context: {RetrievedText} Query: {Query}"]
        TargetText = [f"Based on the document: {RetrievedText}"]
        InputTensor = self.TransformerModel.Encoder.SentenceEmbedding.BatchTokenize(InputText, StartToken=True, EndToken=True)
        TargetTensor = self.TransformerModel.Decoder.SentenceEmbedding.BatchTokenize(TargetText, StartToken=True, EndToken=True)
        
        SeqLen = InputTensor.size(1)
        SelfAttentionMask = torch.triu(torch.ones(SeqLen, SeqLen), diagonal=1).bool().to(getDevice())
        SelfAttentionMask = SelfAttentionMask * -1e9
        
        self.TransformerModel.eval()
        
        with torch.no_grad():
            Output = self.TransformerModel(InputText, TargetText,
                                        EncoderSelfAttentionMask=None,
                                        DecoderSelfAttentionMask=SelfAttentionMask,
                                        DecoderCrossAttentionMask=None,
                                        EncStartToken=True,
                                        EncEndToken=True,
                                        DecStartToken=True,
                                        DecEndToken=True)
        
        
        OutputProbs = F.softmax(Output, dim=-1)
        PredictedIndices = torch.argmax(OutputProbs, dim=-1)
        
        PredictedTokens = [self.IndexToLanguage.get(Idx.item(), '') for Idx in PredictedIndices[0] if Idx.item() in self.IndexToLanguage and self.IndexToLanguage[Idx.item()] not in ['<START>', '<END>', '<PAD>']]
        Response = ' '.join(PredictedTokens).strip()
        
        if not Response:
            Result = self.Db.Search(Query)
            if Result:
                Response = f"based on the document: {RetrievedText} (Source: {Result['Source']})"
            else:
                Response = "i could not find relevant information!"
        return Response

    def ask(self, Query):

        self.Memory.append(('user', Query))
        Context = self.Memory[-2][1] if len(self.Memory) > 1 and self.Memory[-2][0] == 'bot' else ''
        FullQuery = Context + ' ' + Query if Context else Query
        Result = self.Db.Search(FullQuery)
        
        if Result:
            Response = self.GenerateResponse(Result['Info'], Query)
        else:
            Response = "i could not find relevant information!"
        
        self.Memory.append(('bot', Response))
        
        return Response