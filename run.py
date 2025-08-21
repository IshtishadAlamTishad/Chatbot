from module.bot import Chatbot
from module.bot import Chatbot
from module.VD import Db
from module.se import LanguageToIndex,IndexToLanguage,MaxSequenceLength,StartToken,EndToken,PaddingToken

model = 'model/trainedModel.pth'

xc = Chatbot(Db,model,LanguageToIndex,IndexToLanguage,MaxSequenceLength,StartToken,EndToken,PaddingToken)

print(xc.ask("What about debt?"))
print(xc.ask("Tell me about infrastructure."))

k = input("Enter your question: ")
print(xc.ask(k))    




