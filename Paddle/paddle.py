import os.path
import urllib
import scipy
from scipy import spatial
import numpy



class word2vec:

    word_dict = dict()
    embedding_table = ""

    def __init__(self):
        if not os.path.isfile("/models/word2vec/inference_topology.pkl"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/04.word2vec/inference_topology.pkl", "models/word2vec/inference_topology.pkl")
        if not os.path.isfile("/models/word2vec/param.tar"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/04.word2vec/param.tar", "models/word2vec/param.tar")
        if not os.path.isfile("/models/word2vec/word_dict"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/04.word2vec/word_dict", "models/word2vec/word_dict")
        if not os.path.isfile("/models/word2vec/embedding_table"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/04.word2vec/embedding_table", "models/word2vec/embedding_table")
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

        


