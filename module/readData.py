import os

def ReadDataset(DatasetFilePath):
    if not os.path.exists(DatasetFilePath):
        raise FileNotFoundError(f"Dataset file {DatasetFilePath} not found.")
    
    ans = []
    with open(DatasetFilePath, 'r', encoding='utf-8') as F:
        for Line in F:
            Line = Line.strip()
            if Line:
                try:
                    Question, Answer = eval(f'[{Line}]')
                    ans.append((Question, Answer))
                except:
                    print(f"Skipping malformed line: {Line}")
    return ans