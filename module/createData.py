import torch
import string
from module.readData import ReadDataset

def CreateDataset(DatasetFilePath, LanguageToIndex, MaxSequenceLength):
    QuestionAnswerPairs = ReadDataset(DatasetFilePath)
    Dataset = []
    for Question, Answer in QuestionAnswerPairs:
        QTokens = [LanguageToIndex.get(Token, LanguageToIndex['<PAD>']) for Token in Question.lower().translate(str.maketrans('', '', string.punctuation)).split()]
        ATokens = [LanguageToIndex.get(Token, LanguageToIndex['<PAD>']) for Token in Answer.lower().translate(str.maketrans('', '', string.punctuation)).split()]
        if len(QTokens) > MaxSequenceLength - 2:
            QTokens = QTokens[:MaxSequenceLength - 2]
        if len(ATokens) > MaxSequenceLength - 2:
            ATokens = ATokens[:MaxSequenceLength - 2]
        QTokens = [LanguageToIndex['<START>']] + QTokens + [LanguageToIndex['<END>']] + [LanguageToIndex['<PAD>']] * (MaxSequenceLength - len(QTokens) - 2)
        ATokens = [LanguageToIndex['<START>']] + ATokens + [LanguageToIndex['<END>']] + [LanguageToIndex['<PAD>']] * (MaxSequenceLength - len(ATokens) - 2)
        Dataset.append((torch.tensor(QTokens), torch.tensor(ATokens)))
    return Dataset
