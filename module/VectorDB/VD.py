import math
import numpy as np
from collections import defaultdict
import string


class VectorDatabase:
    def __init__(self, Sections):
        self.Sections = Sections
        self.Vocab = self.BuildVocab()
        self.Idf = self.ComputeIdf()
        self.Vectors = self.ComputeTfIdf()

    def BuildVocab(self):
        Vocab = set()
        for Section in self.Sections:
            Words = self.Tokenize(Section['Info'])
            Vocab.update(Words)
        return list(Vocab)

    def Tokenize(self, Text):
        Text = Text.lower().translate(str.maketrans('', '', string.punctuation))
        return Text.split()

    def ComputeIdf(self):
        N = len(self.Sections)
        Idf = defaultdict(float)
        for Word in self.Vocab:
            Df = sum(1 for Section in self.Sections if Word in self.Tokenize(Section['Info']))
            Idf[Word] = math.log(N / (1 + Df)) if Df > 0 else 0
        return Idf

    def ComputeTfIdf(self):
        Vectors = []
        for Section in self.Sections:
            Words = self.Tokenize(Section['Info'])
            Tf = defaultdict(float)
            for Word in Words:
                Tf[Word] = Words.count(Word) / len(Words)
            Vector = np.zeros(len(self.Vocab))
            for I, Word in enumerate(self.Vocab):
                Vector[I] = Tf[Word] * self.Idf[Word]
            Vectors.append(Vector)
        return np.array(Vectors)

    def Search(self, Query):
        QWords = self.Tokenize(Query)
        Tf = defaultdict(float)
        
        for Word in QWords:
            Tf[Word] = QWords.count(Word) / len(QWords)
        QVector = np.zeros(len(self.Vocab))
        
        for I, Word in enumerate(self.Vocab):
            if Word in Tf:
                QVector[I] = Tf[Word] * self.Idf[Word]
        
        if np.linalg.norm(QVector) == 0:
            return None
        Similarities = [np.dot(QVector, V) / (np.linalg.norm(QVector) * np.linalg.norm(V)) if np.linalg.norm(V) > 0 else 0 for V in self.Vectors]
        
        BestIdx = np.argmax(Similarities)
        
        return self.Sections[BestIdx]