
import random
from Preprocessor import Processor

'''By Samuel Emilolorun'''
'''
Implements the continuous bag of words model on youtube comments. Trains each word against words that surround it
in a commment
'''
class CBOW:
    def __init__(self, datasetPath, numberOfComments, vectorSize=100, epochs=5, maxSentenceSize=10):
        self.processor = Processor(datasetPath, numberOfComments)
        self.vectorSize = vectorSize
        self.epochs = epochs
        self.maxSentenceSize = maxSentenceSize
        self.maxContextSize = 3
        random.seed(0)



    def buildVocab(self):
        unquieWord = set()
        for comment in self.processor:
            unquieWord.update(comment)
        self.vocab = sorted(unquieWord)

cbow = CBOW("dataset.txt", 61234)

cbow.buildVocab()