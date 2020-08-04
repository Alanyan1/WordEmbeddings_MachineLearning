from Model.ContextBasedModel import CBOW
from Model.Visual import Visualiser


def run():
    cbow = CBOW("dataset.txt", 5000)
    cbow.train()
    cbow.load()
    visualiser = Visualiser()
    visualiser.plotEmbeddings(cbow)

run()