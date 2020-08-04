from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px

class Visualiser:
    def __init__(self, model=None):
        self.model = model

    def plotEmbeddings(self, model):
        self.model = model
        reduced = self.__tSNEReduction(model.vectors)
        reduced = reduced.transpose()
        self.__twoDPlot(reduced, model.words)

    def __tSNEReduction(self, vectors):
        tsne = TSNE(n_components=2, random_state=0)
        result = tsne.fit_transform(vectors)
        return result

    def __twoDPlot(self, reduced, words):
        df = pd.DataFrame({"X": reduced[0], "Y": reduced[1], "Names": words})
        fig = px.scatter(df, x="X", y="Y", hover_name="Names",
                         title="Recipe Embeddings Using " + " CBOW Model with Vector size: "
                               + str(self.model.vectors.shape[1]))
        fig.show()