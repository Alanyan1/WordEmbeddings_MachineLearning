"""
By Samuel Emilolorun
"""
from googleapiclient.discovery import build
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
nltk.download("stopwords")
nltk.download("words")

class DatasetGenerator:
    def __init__(self, youtube_api_key):
        self.youtube = build("youtube", "v3", developerKey=youtube_api_key)


    def buildCommentsDataSet(self, lines, filepath, query="tech"):
        number = lines // (50 * 100) if lines > 5000 else 1
        videoIds = self.getPopluarVideoIds(number, query)
        for videoId in videoIds:
            comments = self.getMostRelevantComments(videoId)
            self.addToFile(comments, filepath)

    def getPopluarVideoIds(self, number, query):
        videoIds = []
        nextPage = None
        for i in range(number):
            print(i)
            request = self.youtube.search().list(part="snippet", q=query, order="relevance", maxResults=50,
                                                 type="video", relevanceLanguage="en", safeSearch="moderate",
                                                 pageToken=nextPage)
            response = request.execute()
            for item in response["items"]:
                videoIds.append(item["id"]["videoId"])
            nextPage = response.get("nextPageToken")
        return videoIds

    def getMostRelevantComments(self, videoId):
        comments = []
        request = self.youtube.commentThreads().list(part="snippet", videoId=videoId, maxResults=100, order="relevance", textFormat="plainText")
        try:
            response = request.execute()
        except Exception as err:
            return []
        for item in response["items"]:
            comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
        return comments


    def addToFile(self, comments, filePath):
        with open(filePath, "a") as file:
            for comment in comments:
                comment = self.clean(comment)
                if comment:
                    file.write(comment + "\n")

    def clean(self, comment):
        comment = comment.lower()
        comment = re.findall("[a-z]+", comment)
        comment = [token for token in comment if len(token) > 2 and wordnet.synsets(token)and token not in stopwords.words("english")]
        return ",".join(comment) if len(comment) > 2 else []


#EXAMPLE USE

# youtube_api_key = "PLEASE ADD A VALID API KEY"
# youtube = DatasetGenerator(youtube_api_key)
# vidIds = youtube.buildCommentsDataSet(100000, "dataset.txt", "finance")
