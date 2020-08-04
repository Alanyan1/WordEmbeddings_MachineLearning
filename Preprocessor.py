

class Processor:
    def __init__(self, datasetPath, numberOfComments):
        self.datasetPath = datasetPath
        self.numberOfComments = numberOfComments
        self.currentCommentList = None

    def __iter__(self):
        """
        Yields result of data-preprocessing to a model.
        """
        with open(self.datasetPath, "r") as file:
            for recipeLine in range(self.numberOfComments):
                currentComment = file.readline()
                currentCommentList = currentComment.replace("\n", "").split(",")
                yield currentCommentList



#Example Use
# process = Processor("dataset.txt", 2)
#
# for element in process:
#     print(element)