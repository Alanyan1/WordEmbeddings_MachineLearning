
import random
import json
import numpy as np
from Model.Preprocessor import Processor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Model
from keras.layers import Dense, Input, Average
from keras.models import load_model

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
        self.vectors = []
        self.words = []
        self.maxContextSize = 3
        random.seed(0)


    def train(self):
        '''
        Builds the vocab, trains and saves the Artificial Neural Networl,
         after setting up the matrix of features and targets
        '''
        self.__buildVocab()
        classifier = self.__buildNeuralNet()
        x1 = np.array(self.input1)
        x2 = np.array(self.input2)
        x3 = np.array(self.input3)
        y = np.array(self.targets)
        classifier.fit([x1, x2, x3], y, batch_size=1000, epochs=self.epochs)
        classifier.save("CBOW")

    def load(self):
        """
        Loads a trained model and uses it to find a vector representation for every word used to train the model
        This ultimately populates the vectors and words attributes.
        """
        classifier = load_model("CBOW")
        outputLayer = classifier.get_layer("Embedding")
        self.wordVectors = np.transpose(outputLayer.get_weights()[0])
        self.__loadWordToPosition("WordPositions")
        self.__populateVectorNames()
    def __buildVocab(self):
        unquieWord = set()
        for comment in self.processor:
            unquieWord.update(comment)
        self.vocab = sorted(unquieWord)
        self.vocab_oneHotEncoded = self.__oneHotEncode()
        self.__buildMatrixIO()

    def __oneHotEncode(self):
        integerEncoder = LabelEncoder().fit_transform(self.vocab)
        integerEncoderReshaped = integerEncoder.reshape(len(integerEncoder), 1)
        encoded = OneHotEncoder(sparse=False).fit_transform(integerEncoderReshaped)
        self.wordToPosition = {k: int(v) for k, v in zip(self.vocab, integerEncoder)}
        self.__saveWordToPosition()
        return encoded

    def __buildMatrixIO(self):
        '''
        Matrix of features and targets are built
        '''
        self.input1 = []
        self.input2 = []
        self.input3 = []
        self.targets = []
        for comment in self.processor:
            for position in range(len(comment)):
                contexts = self.__getContextList(position, comment)
                if contexts:
                    self.__addToInputOutput(contexts, comment[position])

    def __getContextList(self, position, comment):
        comment = [word for word in comment if word is not comment[position]]
        if len(comment) < self.maxContextSize:
            return []
        return random.sample(comment, self.maxContextSize)

    def __addToInputOutput(self, contexts, target):
        self.input1.append(self.vocab_oneHotEncoded[self.wordToPosition[contexts[0]]])
        self.input2.append(self.vocab_oneHotEncoded[self.wordToPosition[contexts[1]]])
        self.input3.append(self.vocab_oneHotEncoded[self.wordToPosition[contexts[2]]])
        self.targets.append(self.vocab_oneHotEncoded[self.wordToPosition[target]])


    def __buildNeuralNet(self):
        input_dim = len(self.vocab)
        input1 = Input(shape=(input_dim,))
        hidden1 = Dense(activation="relu", input_dim=input_dim,
                             units=self.vectorSize, kernel_initializer="uniform")(input1)
        input2 = Input(shape=(input_dim,))
        hidden2 = Dense(activation="relu", input_dim=input_dim,
                        units=self.vectorSize, kernel_initializer="uniform")(input2)
        input3 = Input(shape=(input_dim,))
        hidden3 = Dense(activation="relu", input_dim=input_dim,
                        units=self.vectorSize, kernel_initializer="uniform")(input3)
        averaged = Average()([hidden1, hidden2, hidden3])
        output = Dense(activation="sigmoid", units=input_dim,
                       kernel_initializer="uniform", name="Embedding")(averaged)
        classifier = Model([input1, input2, input3], output)
        classifier.compile(optimizer="adam", loss="binary_crossentropy")
        return classifier

    def __populateVectorNames(self):
        for word in self.wordToPosition:
            self.vectors.append(self.wordVectors[self.wordToPosition[word]])
            self.words.append(word)
        self.numpyLizeKeyLists()
        print("VECTORS:   ", len(self.vectors))
        print("WORDS: ", len(self.words))



    def __saveWordToPosition(self):
        with open("WordPositions", 'w') as file:
            json.dump(self.wordToPosition, file)

    def __loadWordToPosition(self, vocabPath):
        with open(vocabPath, 'r') as file:
           self.wordToPosition = json.load(file)


    def numpyLizeKeyLists(self):
        self.vectors = np.array(self.vectors)
        self.words = np.array(self.words)