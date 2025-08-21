from module.bot import Chatbot
from module.VD import Db
from module.se import LanguageToIndex, IndexToLanguage, MaxSequenceLength, StartToken, EndToken, PaddingToken

model = 'model/trainedModel.pth'

xc = Chatbot(Db, model, LanguageToIndex, IndexToLanguage, MaxSequenceLength, StartToken, EndToken, PaddingToken)

print(xc.ask("What about debt?"))
print(xc.ask("Tell me about infrastructure."))
print("\nType 'exit' to stop the program.\n")

i = 1
while True:
    k = input("Enter your question: ")
    
    if k.lower() == "exit":
        print("Exited!")
        break

    print(f"{i} → {xc.ask(k)}")
    i += 1
