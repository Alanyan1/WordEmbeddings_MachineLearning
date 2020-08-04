
import random
import json
from Preprocessor import Processor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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
        self.vocab_oneHotEncoded = self.__oneHotEncode()

    def __oneHotEncode(self):
        integerEncoder = LabelEncoder().fit_transform(self.vocab)
        integerEncoderReshaped = integerEncoder.reshape(len(integerEncoder), 1)
        encoded = OneHotEncoder(sparse=False).fit_transform(integerEncoderReshaped)
        self.wordToPosition = {k: int(v) for k, v in zip(self.vocab, integerEncoder)}
        self.__saveWordToPosition()
        return encoded

    def __saveWordToPosition(self):
        with open("WordPositions", 'w') as file:
            json.dump(self.wordToPosition, file)




cbow = CBOW("dataset.txt", 10)

cbow.buildVocab()