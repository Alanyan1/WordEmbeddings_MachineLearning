from ContextBasedModel import CBOW
from Visual import Visualiser


def run():
    cbow = CBOW("dataset.txt", 5000)
    cbow.train()
    cbow.load()
    visualiser = Visualiser()
    visualiser.plotEmbeddings(cbow)

run()