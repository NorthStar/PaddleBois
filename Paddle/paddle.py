import os.path
import urllib
import scipy
from scipy import spatial
import numpy


# 1
class word2vec:

    word_dict = dict()
    embedding_table = ""

    def __init__(self):
        self.load_files()
        self.word_dict = dict()
        with open("models/word2vec/word_dict", "r") as f:
            for line in f:
                key, value = line.strip().split(" ")
                self.word_dict[key] = value
        self.embedding_table = numpy.loadtxt("models/word2vec/embedding_table", delimiter=",")

    def run(self, s1, s2):
        if s1 in self.word_dict:
            print("contains s1")
        else:
            print(self.word_dict)
            print("doesn't contain")
        i1 = int(self.word_dict[s1])
        i2 = int(self.word_dict[s2])
        return spatial.distance.cosine(self.embedding_table[i1], self.embedding_table[i2])

    def load_files(self):
        if not os.path.isfile("/models/word2vec/inference_topology.pkl"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/04.word2vec/inference_topology.pkl",
                               "models/word2vec/inference_topology.pkl")
            print("Loading models/word2vec/inference_topology.pkl . . .")

        if not os.path.isfile("/models/word2vec/param.tar"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/04.word2vec/param.tar",
                               "models/word2vec/param.tar")
            print("Loading models/word2vec/param.tar . . .")

        if not os.path.isfile("/models/word2vec/word_dict"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/04.word2vec/word_dict",
                               "models/word2vec/word_dict")
            print("Loading models/word2vec/word_dict . . .")

        if not os.path.isfile("/models/word2vec/embedding_table"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/04.word2vec/embedding_table",
                               "models/word2vec/embedding_table")
            print("Loading models/word2vec/embedding_table . . .")

# 2
class image_classification:

    def __init__(self):
        self.load_files()


    def load_files(self):
        if not os.path.isfile("models/image_classification/inference_topology.pkl"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/03.image_classification/inference_topology.pkl",
                               "models/image_classification/inference_topology.pkl")
            print("Loading models/image_classification/inference_topology.pkl . . .")

        if not os.path.isfile("models/image_classification/param.tar"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/03.image_classification/param.tar",
                               "models/image_classification/param.tar")
            print("Loading models/image_classification/param.tar . . .")


# 3
class sentiment_classification:

    def __init__(self):
        self.load_files()

    def load_files(self):
        if not os.path.isfile("models/sentiment_classification/inference_topology.pkl"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/06.understand_sentiment/inference_topology.pkl",
                               "models/sentiment_classification/inference_topology.pkl")
            print("Loading models/sentiment_classification/inference_topology.pkl . . .")

        if not os.path.isfile("models/sentiment_classification/param.tar"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/06.understand_sentiment/param.tar",
                               "models/sentiment_classification/param.tar")
            print("Loading models/sentiment_classification/param.tar . . .")

        if not os.path.isfile("models/sentiment_classification/word_dict.tar"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/06.understand_sentiment/word_dict.tar",
                               "models/sentiment_classification/word_dict.tar")
            print("Loading models/sentiment_classification/word_dict.tar . . .")


# 4
class machine_translation:

    def __init__(self):
        self.load_files()

    def load_files(self):
        if not os.path.isfile("models/machine_translation/inference_topology.pkl"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/08.machine_translation/inference_topology.pkl",
                               "models/machine_translation/inference_topology.pkl")
            print("Loading models/machine_translation/inference_topology.pkl . . .")

        if not os.path.isfile("models/machine_translation/param.tar"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/08.machine_translation/param.tar",
                               "models/machine_translation/param.tar")
            print("Loading models/machine_translation/param.tar . . .")

        if not os.path.isfile("models/machine_translation/src_dict.txt"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/08.machine_translation/src_dict.txt",
                               "models/machine_translation/src_dict.txt")
            print("Loading models/machine_translation/src_dict.txt . . .")

        if not os.path.isfile("models/machine_translation/trg_dict.txt"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/08.machine_translation/trg_dict.txt",
                               "models/machine_translation/trg_dict.txt")
            print("Loading models/machine_translation/trg_dict.txt . . .")


# 5
class recognize_digits:

    def __init__(self):
        self.load_files()

    def load_files(self):
        if not os.path.isfile("models/recognize_digits/inference_topology.pkl"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/inference_topology.pkl",
                               "models/recognize_digits/inference_topology.pkl")
            print("Loading models/recognize_digits/inference_topology.pkl . . .")

        if not os.path.isfile("models/recognize_digits/param.tar"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/param.tar",
                              "models/recognize_digits/param.tar")
            print("Loading models/recognize_digits/param.tar . . .")

